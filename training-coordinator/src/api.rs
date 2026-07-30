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

use std::sync::Arc;

use axum::extract::State as AxumState;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use parking_lot::Mutex;
use serde::Serialize;

use crate::attestation::{self, Attestation};
use crate::job::JobSpec;
use crate::state::State;
use crate::store::Store;
use crate::submission::{recover_submitter, SignedSubmission};

pub struct Coordinator {
    state: Mutex<State>,
    store: Store,
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
        })
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
    lease_digest, LeaseRequest, LEASE_MESSAGE,
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
    let who =
        citrate_training_worker::wallet::Wallet::recover_address(&lease_digest(), &req.signature)
            .map_err(|_| err(StatusCode::UNAUTHORIZED, "signature does not recover"))?;
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
    let mut s = c.state.lock();
    // Expire on read so a status page never shows a lease that is already dead.
    s.expire_leases(unix_now());
    Json(StatusResponse {
        counts: s.counts(),
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
