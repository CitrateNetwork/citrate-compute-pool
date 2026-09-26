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
//! `POST /v1/heartbeat` — a signed "still working"; extends the lease by one
//!                        renewal window, up to the job's deadline.
//! `POST /v1/submit`    — a signed result.
//! `GET  /v1/status`    — fleet counts. This is what alf-gateway polls to fill
//!                        the round/training panels the portal already renders.

use std::collections::HashSet;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;

use axum::extract::{ConnectInfo, State as AxumState};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use ethereum_types::H160;
use parking_lot::Mutex;
use serde::Serialize;

/// Bound on the lease/heartbeat seen-set. A signed request is only usable
/// inside the freshness window, so the set only needs that window's
/// (signer, timestamp) pairs, and only for registered signers (a replayed
/// request from an unknown key does nothing). Sized for a full registry
/// polling at its fastest; out-of-window entries are pruned first and an
/// in-window entry is never evicted (the request is refused instead).
const LEASE_SEEN_CAP: usize = 250_000;

/// PBA-L3b-001: new identities admitted per hour, across all sources. Refreshing
/// an existing registration is free; only a key the coordinator has never seen
/// spends from this budget. A volunteer fleet of tens of machines never comes
/// near it; a script minting a key per lease cycle does.
pub const NEW_REGISTRATIONS_PER_HOUR: u64 = 60;
/// Burst allowance for [`NEW_REGISTRATIONS_PER_HOUR`] (a lab bringing a rack
/// online at once).
pub const NEW_REGISTRATION_BURST: u64 = 30;

/// Token bucket for new-identity registrations. In memory: a restart refills it,
/// which is no worse than the burst.
#[derive(Debug)]
struct RegistrationBudget {
    tokens: u64,
    /// Start of the current refill interval. `None` until the first request,
    /// so intervals are measured from that request rather than from epoch
    /// minute boundaries (which made the burst depend on the wall clock).
    last_refill: Option<u64>,
}

impl RegistrationBudget {
    fn new() -> Self {
        Self {
            tokens: NEW_REGISTRATION_BURST,
            last_refill: None,
        }
    }

    /// Take one token at `now` (unix seconds). `false` means over budget.
    fn try_take(&mut self, now: u64) -> bool {
        let per_token = 3_600 / NEW_REGISTRATIONS_PER_HOUR;
        let last = *self.last_refill.get_or_insert(now);
        let earned = now.saturating_sub(last) / per_token;
        if earned > 0 {
            self.tokens = self
                .tokens
                .saturating_add(earned)
                .min(NEW_REGISTRATION_BURST);
            self.last_refill = Some(last.saturating_add(earned * per_token));
        }
        if self.tokens == 0 {
            return false;
        }
        self.tokens -= 1;
        true
    }
}

use crate::attestation::{self, Attestation};
use crate::job::JobSpec;
use crate::state::{LeaseError, Policy, RegisterError, State};
use crate::store::Store;
use crate::submission::{recover_submitter, SignedSubmission};

/// Why a registration was refused inside the state lock.
enum RegFail {
    Replay,
    TooSoon,
    Budget,
    Registry(RegisterError),
}

/// Why a signed request was refused by the seen-set.
#[derive(Debug, PartialEq)]
enum Seen {
    Replay,
    Full,
}

pub struct Coordinator {
    state: Mutex<State>,
    store: Store,
    /// Recently-seen (signer, timestamp) lease requests, to refuse an exact
    /// replay within the freshness window (CP-B-002). In-memory only: the
    /// freshness window bounds replay across a restart, so this need not persist.
    lease_seen: Mutex<HashSet<(H160, u64)>>,
    /// PBA-L3b-001: budget for never-seen identities.
    new_registrations: Mutex<RegistrationBudget>,
}

impl Coordinator {
    /// Load persisted state, or start empty. A corrupt state file is an error
    /// here rather than a silent fresh start, so the daemon refuses to boot
    /// having forgotten which machines are mid-job.
    pub fn open(store: Store) -> std::io::Result<Self> {
        Self::open_with(store, Policy::default())
    }

    /// As [`Coordinator::open`], with an operator [`Policy`] (PBA-L3b-001).
    pub fn open_with(store: Store, policy: Policy) -> std::io::Result<Self> {
        let mut state = store.load()?;
        state.policy = policy;
        // A policy tightened across a restart applies to leases already out.
        let revoked = state.revoke_out_of_policy();
        if !revoked.is_empty() {
            tracing::info!(
                ?revoked,
                "requeued leases the current tier policy no longer allows"
            );
            store.save(&state)?;
        }
        state.take_dirty();
        Ok(Self {
            state: Mutex::new(state),
            store,
            lease_seen: Mutex::new(HashSet::new()),
            new_registrations: Mutex::new(RegistrationBudget::new()),
        })
    }

    /// Record that `(who, timestamp)` has been used. `Replay` if it already
    /// was. When the set is full, entries outside the freshness window are
    /// dropped first; if it is still full of in-window entries the request is
    /// refused (`Full`) rather than forgetting one that could be replayed.
    fn note_seen(&self, who: H160, timestamp: u64, now_nanos: u64) -> Result<(), Seen> {
        let mut set = self.lease_seen.lock();
        if set.contains(&(who, timestamp)) {
            return Err(Seen::Replay);
        }
        if set.len() >= LEASE_SEEN_CAP {
            set.retain(|(_, ts)| ts.abs_diff(now_nanos) <= LEASE_FRESHNESS_NANOS);
            if set.len() >= LEASE_SEEN_CAP {
                return Err(Seen::Full);
            }
        }
        set.insert((who, timestamp));
        Ok(())
    }

    /// Run `f`, then persist. If persisting fails the change is still live in
    /// memory, and the error is returned so the caller can refuse the request —
    /// reporting success for a lease that would vanish on restart is worse than
    /// reporting failure for one that was granted.
    ///
    /// Only persists when `f` changed something that must survive a restart
    /// ([`State::take_dirty`]): a refused request or a poll that only
    /// refreshes `last_seen` costs no serialisation and no fsync.
    fn mutate<T>(&self, f: impl FnOnce(&mut State) -> T) -> std::io::Result<T> {
        let mut g = self.state.lock();
        let out = f(&mut g);
        if g.take_dirty() {
            if let Err(e) = self.store.save(&g) {
                // Not persisted: keep it pending so the next request retries.
                g.mark_dirty();
                return Err(e);
            }
        }
        Ok(out)
    }

    /// [`Coordinator::mutate`] on the blocking pool, so a slow disk (the
    /// save fsyncs) stalls only this request and never the async runtime that
    /// serves every other worker's heartbeat.
    async fn mutate_off_runtime<T: Send + 'static>(
        self: &Arc<Self>,
        f: impl FnOnce(&mut State) -> T + Send + 'static,
    ) -> std::io::Result<T> {
        let me = Arc::clone(self);
        match tokio::task::spawn_blocking(move || me.mutate(f)).await {
            Ok(r) => r,
            Err(e) => Err(std::io::Error::other(e.to_string())),
        }
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
    heartbeat_digest, lease_digest, HeartbeatRequest, HeartbeatResponse, LeaseRequest,
    LEASE_FRESHNESS_NANOS, LEASE_MESSAGE,
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

/// Where a request came from, for the per-source identity cap (PBA-L3b-001).
///
/// The coordinator is deployed loopback-only behind Caddy, which sets
/// `X-Forwarded-For` to the client address (it does not trust an inbound one
/// unless `trusted_proxies` is configured). So the header is believed only when
/// the TCP peer is loopback, and the rightmost entry is used: that is the one the
/// local proxy wrote. A request with no peer information at all is `unknown`,
/// never header-derived. IPv6 is folded to its /64, the unit one host controls.
pub fn source_of(peer: Option<SocketAddr>, headers: &HeaderMap) -> String {
    let ip = match peer {
        Some(p) if !p.ip().is_loopback() => Some(p.ip()),
        Some(_) => headers
            .get("x-forwarded-for")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.rsplit(',').next())
            .and_then(|v| v.trim().parse::<IpAddr>().ok())
            .or(Some(IpAddr::from([127, 0, 0, 1]))),
        None => None,
    };
    match ip {
        Some(IpAddr::V6(v6)) => {
            let s = v6.segments();
            format!("{:x}:{:x}:{:x}:{:x}::/64", s[0], s[1], s[2], s[3])
        }
        Some(IpAddr::V4(v4)) => v4.to_string(),
        None => "unknown".to_string(),
    }
}

async fn register(
    AxumState(c): AxumState<Arc<Coordinator>>,
    peer: Option<ConnectInfo<SocketAddr>>,
    headers: HeaderMap,
    Json(att): Json<Attestation>,
) -> Result<Json<RegisterResponse>, ApiError> {
    // A registration is bound to the moment it was signed, like a lease or a
    // heartbeat: stale or repeated bodies are refused.
    if att.timestamp == 0 {
        return Err(err(
            StatusCode::UPGRADE_REQUIRED,
            "worker too old: registration requires a signed timestamp; upgrade the worker",
        ));
    }
    if unix_now_nanos().abs_diff(att.timestamp) > LEASE_FRESHNESS_NANOS {
        return Err(err(
            StatusCode::UNAUTHORIZED,
            "registration timestamp is outside the freshness window",
        ));
    }
    let w = attestation::verify(&att).map_err(|e| err(StatusCode::BAD_REQUEST, e.to_string()))?;
    let ts = att.timestamp;
    let source = source_of(peer.map(|ConnectInfo(a)| a), &headers);
    let now = unix_now();
    let (c2, w2, source2) = (Arc::clone(&c), w.clone(), source.clone());
    let granted = c
        .mutate_off_runtime(move |s| {
            // Checks that change nothing come first, so a refused request
            // never causes a save.
            if s.registration_seen(w2.id, ts) {
                return Err(RegFail::Replay);
            }
            if let Some(known) = s.workers.get(&w2.id) {
                if now
                    < known
                        .last_registration
                        .saturating_add(s.policy.register_refresh_secs)
                {
                    return Err(RegFail::TooSoon);
                }
            }
            // PBA-L3b-001: a never-seen identity spends from the global budget,
            // but only once its source is known to have room, so a full source
            // cannot drain the budget and lock out every other newcomer.
            if !s.workers.contains_key(&w2.id) && !s.policy.is_trusted(&w2.id) {
                s.admits_new(&w2.id, &source2, now)
                    .map_err(RegFail::Registry)?;
                if !c2.new_registrations.lock().try_take(now) {
                    return Err(RegFail::Budget);
                }
            }
            s.note_registration(w2.id, ts, unix_now_nanos());
            s.register(&w2, &source2, now).map_err(RegFail::Registry)
        })
        .await
        .map_err(|e| {
            err(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("could not persist: {e}"),
            )
        })?
        .map_err(|e| match e {
            RegFail::Budget => err(
                StatusCode::TOO_MANY_REQUESTS,
                "too many new registrations; retry later",
            ),
            RegFail::TooSoon => err(
                StatusCode::TOO_MANY_REQUESTS,
                "registered too recently; retry later",
            ),
            RegFail::Replay => err(
                StatusCode::UNAUTHORIZED,
                "registration already used (replay)",
            ),
            RegFail::Registry(e) => err(StatusCode::TOO_MANY_REQUESTS, e.to_string()),
        })?;
    Ok(Json(RegisterResponse {
        worker: format!("{:?}", w.id),
        capability: granted,
    }))
}

/// Check freshness, recover the signer over `digest`, and refuse an exact
/// replay (CP-B-002). Shared by every signed, timestamp-bound request.
fn authenticate(
    c: &Coordinator,
    timestamp: u64,
    digest: &[u8; 32],
    signature: &[u8],
) -> Result<H160, ApiError> {
    if unix_now_nanos().abs_diff(timestamp) > LEASE_FRESHNESS_NANOS {
        return Err(err(
            StatusCode::UNAUTHORIZED,
            "request timestamp is outside the freshness window",
        ));
    }
    let who = citrate_training_worker::wallet::Wallet::recover_address(digest, signature)
        .map_err(|_| err(StatusCode::UNAUTHORIZED, "signature does not recover"))?;
    // Only a registered signer's requests can do anything, so only they are
    // remembered; an unknown key cannot fill the seen-set.
    if c.state.lock().workers.contains_key(&who) {
        match c.note_seen(who, timestamp, unix_now_nanos()) {
            Ok(()) => {}
            Err(Seen::Replay) => {
                return Err(err(
                    StatusCode::UNAUTHORIZED,
                    "request already used (replay)",
                ))
            }
            Err(Seen::Full) => {
                return Err(err(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "too many requests in flight; retry shortly",
                ))
            }
        }
    }
    Ok(who)
}

fn too_many(until: u64, now: u64, msg: String) -> Response {
    let mut r = err(StatusCode::TOO_MANY_REQUESTS, msg).into_response();
    if let Ok(v) = until.saturating_sub(now).max(1).to_string().parse() {
        r.headers_mut().insert(axum::http::header::RETRY_AFTER, v);
    }
    r
}

async fn lease(
    AxumState(c): AxumState<Arc<Coordinator>>,
    Json(req): Json<LeaseRequest>,
) -> Response {
    // CP-B-002: reject a stale or far-future timestamp so a captured request is
    // usable only inside a short window, then recover the signer over the SAME
    // timestamp it signed, then refuse an exact replay within that window.
    let who = match authenticate(
        &c,
        req.timestamp,
        &lease_digest(req.timestamp),
        &req.signature,
    ) {
        Ok(w) => w,
        Err(e) => return e.into_response(),
    };
    let now = unix_now();
    let got = match c.mutate_off_runtime(move |s| s.lease(who, now)).await {
        Ok(g) => g,
        Err(e) => {
            return err(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("could not persist: {e}"),
            )
            .into_response()
        }
    };
    match got {
        Ok(spec) => (StatusCode::OK, Json(Some(spec))).into_response(),
        // Nothing to do is not an error — it is the steady state of a fleet with
        // more machines than queued work, and workers poll on it.
        Err(LeaseError::NothingAvailable) => {
            (StatusCode::NO_CONTENT, Json(None::<JobSpec>)).into_response()
        }
        Err(e @ LeaseError::CoolingDown { until }) => too_many(until, now, e.to_string()),
        Err(e @ LeaseError::AtLeaseCap) => err(StatusCode::CONFLICT, e.to_string()).into_response(),
        Err(e @ LeaseError::UnknownWorker) => {
            err(StatusCode::FORBIDDEN, e.to_string()).into_response()
        }
    }
}

async fn heartbeat(
    AxumState(c): AxumState<Arc<Coordinator>>,
    Json(req): Json<HeartbeatRequest>,
) -> Result<Json<HeartbeatResponse>, ApiError> {
    let who = authenticate(
        &c,
        req.timestamp,
        &heartbeat_digest(&req.job, req.timestamp),
        &req.signature,
    )?;
    let now = unix_now();
    let job = req.job.clone();
    let expires_at = c
        .mutate_off_runtime(move |s| s.renew(who, &job, now))
        .await
        .map_err(|e| {
            err(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("could not persist: {e}"),
            )
        })?
        .map_err(|e| err(StatusCode::CONFLICT, e.to_string()))?;
    Ok(Json(HeartbeatResponse { expires_at }))
}

async fn submit(
    AxumState(c): AxumState<Arc<Coordinator>>,
    Json(sub): Json<SignedSubmission>,
) -> Result<StatusCode, ApiError> {
    let who = recover_submitter(&sub).map_err(|e| err(StatusCode::UNAUTHORIZED, e.to_string()))?;
    let now = unix_now();
    let (job, payload) = (sub.job.clone(), sub.payload.clone());
    c.mutate_off_runtime(move |s| s.submit(who, &job, payload, now))
        .await
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
        .route("/v1/heartbeat", post(heartbeat))
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

#[cfg(test)]
mod tests {
    use crate::store::Store;

    fn coord(name: &str) -> Coordinator {
        let dir =
            std::env::temp_dir().join(format!("citrate-coord-seen-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Coordinator::open(Store::new(dir.join("state.json"))).unwrap()
    }

    /// The seen-set never evicts an entry still inside the freshness window:
    /// out-of-window entries are dropped first, and if the set is still full
    /// the request is refused instead.
    #[test]
    fn seen_set_keeps_in_window_entries() {
        let c = coord("keep");
        let now = 10 * LEASE_FRESHNESS_NANOS;
        let a = H160::from_low_u64_be(1);
        for i in 0..LEASE_SEEN_CAP as u64 {
            assert_eq!(c.note_seen(a, now - i, now), Ok(()));
        }
        assert_eq!(
            c.note_seen(a, now + 1, now),
            Err(Seen::Full),
            "full of in-window entries"
        );
        // Every earlier entry is still remembered.
        assert_eq!(c.note_seen(a, now, now), Err(Seen::Replay));
        assert_eq!(
            c.note_seen(a, now - (LEASE_SEEN_CAP as u64 - 1), now),
            Err(Seen::Replay)
        );
        // Once they fall out of the window they are pruned and room returns.
        let later = now + 2 * LEASE_FRESHNESS_NANOS;
        assert_eq!(c.note_seen(a, later, later), Ok(()));
        assert!(c.lease_seen.lock().len() < 10);
    }
    use super::*;

    /// PBA-L3b-001: the new-identity budget spends its burst, then earns one
    /// token per `3600 / NEW_REGISTRATIONS_PER_HOUR` seconds, never above the burst.
    #[test]
    fn registration_budget_refills_at_the_configured_rate() {
        let per_token = 3_600 / NEW_REGISTRATIONS_PER_HOUR;
        let mut b = RegistrationBudget::new();
        for _ in 0..NEW_REGISTRATION_BURST {
            assert!(b.try_take(0));
        }
        assert!(!b.try_take(0), "burst exhausted");
        assert!(!b.try_take(per_token - 1), "no token before one interval");
        assert!(b.try_take(per_token), "one token after one interval");
        assert!(!b.try_take(per_token), "and only one");
        // A long idle period refills to the burst, not beyond.
        let later = per_token * 10 * NEW_REGISTRATION_BURST;
        for _ in 0..NEW_REGISTRATION_BURST {
            assert!(b.try_take(later));
        }
        assert!(!b.try_take(later));
    }

    /// The burst does not depend on where the first request falls relative
    /// to a wall-clock minute: a full burst taken across what would have been
    /// a minute boundary still yields exactly the burst.
    #[test]
    fn registration_budget_burst_is_independent_of_the_wall_clock() {
        let per_token = 3_600 / NEW_REGISTRATIONS_PER_HOUR;
        for start in [0, 1, per_token - 1, 1_790_000_039] {
            let mut b = RegistrationBudget::new();
            let mut granted = 0;
            for i in 0..(NEW_REGISTRATION_BURST + 5) {
                // Requests one second apart, crossing epoch minute boundaries.
                if b.try_take(start + i.min(per_token - 1)) {
                    granted += 1;
                }
            }
            assert_eq!(granted, NEW_REGISTRATION_BURST, "start {start}");
        }
    }
}
