//! The HTTP surface. The only async part of this crate.
//!
//! Four routes, and every one of them is unauthenticated by design: the
//! authentication *is* the signature. A worker registers by signing a probe and a
//! result by signing the result, and in both cases the coordinator recovers who
//! sent it rather than checking a credential. There is no API key to issue, leak
//! or rotate, and no roster to keep current as members join and leave.
//!
//! `POST /v1/register`  — a signed probe; returns the capability granted.
//! `POST /v1/lease`     — a signed request for work; returns a job or 204.
//! `POST /v1/submit`    — a signed result.
//! `GET  /v1/status`    — fleet counts. This is what alf-gateway polls to fill
//!                        the round/training panels the portal already renders.

use std::collections::HashSet;
use std::sync::Arc;

use axum::extract::State as AxumState;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use ethereum_types::H160;
use parking_lot::Mutex;
use serde::Serialize;

/// Bound on the lease-replay set, mirroring the pool-coordinator's dedup cap. A
/// captured lease request is only usable inside the freshness window, so the set
/// only needs to remember that window's worth of (signer, timestamp) pairs.
const LEASE_SEEN_CAP: usize = 10_000;

use crate::attestation::{self, Attestation};
use crate::job::JobSpec;
use crate::state::State;
use crate::store::Store;
use crate::submission::{recover_submitter, SignedSubmission};

pub struct Coordinator {
    state: Mutex<State>,
    store: Store,
    /// Recently-seen (signer, timestamp) lease requests, to refuse an exact
    /// replay within the freshness window (CP-B-002). In-memory only: the
    /// freshness window bounds replay across a restart, so this need not persist.
    lease_seen: Mutex<HashSet<(H160, u64)>>,
}

impl Coordinator {
    /// Load persisted state, or start empty. A corrupt state file is an error
    /// here rather than a silent fresh start, so the daemon refuses to boot
    /// having forgotten which machines are mid-job.
    pub fn open(store: Store) -> std::io::Result<Self> {
        let state = store.load()?;
        Ok(Self {
            state: Mutex::new(state),
            store,
            lease_seen: Mutex::new(HashSet::new()),
        })
    }

    /// Record that `(who, timestamp)` has asked for a lease. Returns `false` if it
    /// was already recorded — an exact replay of a captured request, which must be
    /// refused. Bounded like the pool-coordinator's dedup set: when full, drop the
    /// oldest-observed half (a brute-force but deterministic eviction).
    fn note_lease_seen(&self, who: H160, timestamp: u64) -> bool {
        let mut set = self.lease_seen.lock();
        if set.len() >= LEASE_SEEN_CAP {
            let drop_count = set.len() / 2;
            let to_remove: Vec<_> = set.iter().take(drop_count).cloned().collect();
            for k in to_remove {
                set.remove(&k);
            }
        }
        set.insert((who, timestamp))
    }

    /// Run `f`, then persist. If persisting fails the change is still live in
    /// memory, and the error is returned so the caller can refuse the request —
    /// reporting success for a lease that would vanish on restart is worse than
    /// reporting failure for one that was granted.
    fn mutate<T>(&self, f: impl FnOnce(&mut State) -> T) -> std::io::Result<T> {
        let mut g = self.state.lock();
        let out = f(&mut g);
        self.store.save(&g)?;
        Ok(out)
    }

    pub fn snapshot(&self) -> State {
        self.state.lock().clone()
    }

    pub fn add_job(&self, spec: JobSpec) -> std::io::Result<()> {
        self.mutate(|s| s.add_job(spec))
    }
}

// The lease request and its digest come from the shared protocol.
pub use citrate_training_worker::coordinator_protocol::{
    lease_digest, LeaseRequest, LEASE_FRESHNESS_NANOS, LEASE_MESSAGE,
};

#[derive(Debug, Serialize)]
pub struct RegisterResponse {
    pub worker: String,
    pub capability: crate::job::Capability,
}

#[derive(Debug, Serialize)]
pub struct ErrorBody {
    pub error: String,
}

fn err(code: StatusCode, msg: impl Into<String>) -> (StatusCode, Json<ErrorBody>) {
    (code, Json(ErrorBody { error: msg.into() }))
}

type ApiError = (StatusCode, Json<ErrorBody>);

async fn register(
    AxumState(c): AxumState<Arc<Coordinator>>,
    Json(att): Json<Attestation>,
) -> Result<Json<RegisterResponse>, ApiError> {
    let w = attestation::verify(&att).map_err(|e| err(StatusCode::BAD_REQUEST, e.to_string()))?;
    let now = unix_now();
    c.mutate(|s| s.register(&w, now)).map_err(|e| {
        err(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("could not persist: {e}"),
        )
    })?;
    Ok(Json(RegisterResponse {
        worker: format!("{:?}", w.id),
        capability: w.capability,
    }))
}

async fn lease(
    AxumState(c): AxumState<Arc<Coordinator>>,
    Json(req): Json<LeaseRequest>,
) -> Result<(StatusCode, Json<Option<JobSpec>>), ApiError> {
    // CP-B-002: reject a stale or far-future timestamp so a captured request is
    // usable only inside a short window, then recover the signer over the SAME
    // timestamp it signed, then refuse an exact replay within that window.
    if unix_now_nanos().abs_diff(req.timestamp) > LEASE_FRESHNESS_NANOS {
        return Err(err(
            StatusCode::UNAUTHORIZED,
            "lease request timestamp is outside the freshness window",
        ));
    }
    let who = citrate_training_worker::wallet::Wallet::recover_address(
        &lease_digest(req.timestamp),
        &req.signature,
    )
    .map_err(|_| err(StatusCode::UNAUTHORIZED, "signature does not recover"))?;
    if !c.note_lease_seen(who, req.timestamp) {
        return Err(err(
            StatusCode::UNAUTHORIZED,
            "lease request already used (replay)",
        ));
    }
    let now = unix_now();
    let got = c.mutate(|s| s.lease(who, now)).map_err(|e| {
        err(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("could not persist: {e}"),
        )
    })?;
    match got {
        Ok(spec) => Ok((StatusCode::OK, Json(Some(spec)))),
        // Nothing to do is not an error — it is the steady state of a fleet with
        // more machines than queued work, and workers poll on it.
        Err(crate::state::LeaseError::NothingAvailable) => Ok((StatusCode::NO_CONTENT, Json(None))),
        Err(e) => Err(err(StatusCode::FORBIDDEN, e.to_string())),
    }
}

async fn submit(
    AxumState(c): AxumState<Arc<Coordinator>>,
    Json(sub): Json<SignedSubmission>,
) -> Result<StatusCode, ApiError> {
    let who = recover_submitter(&sub).map_err(|e| err(StatusCode::UNAUTHORIZED, e.to_string()))?;
    let now = unix_now();
    c.mutate(|s| s.submit(who, &sub.job, sub.payload.clone(), now))
        .map_err(|e| {
            err(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("could not persist: {e}"),
            )
        })?
        .map_err(|e| err(StatusCode::CONFLICT, e.to_string()))?;
    Ok(StatusCode::ACCEPTED)
}

#[derive(Debug, Serialize)]
pub struct StatusResponse {
    pub counts: crate::state::Counts,
    /// Named so a consumer cannot mistake this for live settlement. Nothing is
    /// paid until the settlement tolerance is set from fleet divergence data.
    pub settlement: &'static str,
}

async fn status(AxumState(c): AxumState<Arc<Coordinator>>) -> Json<StatusResponse> {
    // CP-B-009: expire on a CLONE, not on the shared state. Calling
    // `expire_leases` on the live state here mutates it (rewrites
    // JobStatus, inserts into `failed_by`, can quarantine a job) while
    // bypassing `Coordinator::mutate`, so the mutation was never
    // persisted — an unauthenticated GET silently diverged in-memory
    // state from the crash-atomic store, and a restart in that window
    // resurrected dead leases. Computing the display counts on a
    // snapshot keeps `/v1/status` a true read: shared state (and thus
    // the store) is never touched, so memory and disk stay in agreement.
    let mut view = c.snapshot();
    view.expire_leases(unix_now());
    Json(StatusResponse {
        counts: view.counts(),
        settlement: "shadow",
    })
}

pub fn router(c: Arc<Coordinator>) -> Router {
    Router::new()
        .route("/v1/register", post(register))
        .route("/v1/lease", post(lease))
        .route("/v1/submit", post(submit))
        .route("/v1/status", get(status))
        .with_state(c)
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn unix_now_nanos() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}
