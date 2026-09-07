//! The worker's client for the training coordinator.
//!
//! This is what turns a member's machine from a thing that *can* train into a
//! thing that *does*: register once, then poll for work, run it, return a signed
//! result, repeat.
//!
//! ## No credentials
//!
//! There is nothing to configure but a URL and the keystore the worker already
//! loads. Every request proves who is asking by signing a domain-separated digest
//! from [`crate::coordinator_protocol`], and the coordinator recovers the address.
//! A member never receives, stores or rotates an API key, which matters when the
//! members are volunteers rather than an ops team.
//!
//! ## Failure is the normal case
//!
//! A volunteer machine is on domestic wifi, behind a router that reboots, running
//! next to a game. So:
//!
//!   * **204 is not an error.** Having nothing to do is the steady state of a
//!     fleet with more machines than queued work.
//!   * **Network failures back off and retry rather than exiting.** A worker that
//!     dies when the coordinator restarts is a worker a member has to babysit,
//!     and they will not.
//!   * **Backoff is bounded and deterministic** (no jitter, no RNG) so it is
//!     testable, and resets the moment work arrives.
//!
//! ## What it deliberately does not do
//!
//! There is no lease renewal. A long job's lease is sized by the coordinator when
//! the job is authored (`JobSpec::lease_secs`), so a 48-hour rung is leased for
//! longer than it takes rather than being kept alive by heartbeats. Renewal is
//! the better design once jobs outlive their estimates, and it needs a route that
//! does not exist yet — noted rather than half-built.

use std::time::Duration;

use crate::coordinator_protocol::{
    attestation_digest, lease_digest, submission_digest, Attestation, JobId, JobSpec, LeaseRequest,
    RegisterResponse, SignedSubmission,
};
use crate::wallet::Wallet;

#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    #[error("could not reach the coordinator: {0}")]
    Transport(String),
    #[error("coordinator rejected the request ({status}): {body}")]
    Rejected { status: u16, body: String },
    #[error("could not sign: {0}")]
    Signing(String),
    #[error("unexpected response from the coordinator: {0}")]
    Malformed(String),
}

/// Polling cadence. Starts eager and backs off to a cap so an idle fleet does not
/// hammer the coordinator, while a busy one picks work up promptly.
#[derive(Clone, Copy, Debug)]
pub struct Backoff {
    pub initial: Duration,
    pub max: Duration,
}

impl Default for Backoff {
    fn default() -> Self {
        Self {
            initial: Duration::from_secs(5),
            max: Duration::from_secs(300),
        }
    }
}

impl Backoff {
    /// Doubling, capped. Deterministic on purpose — the alternative is jitter,
    /// which would be better for thundering herds and worse for tests, and with a
    /// fleet this size the herd does not exist yet.
    pub fn next(&self, current: Duration) -> Duration {
        std::cmp::min(current.saturating_mul(2), self.max)
    }
}

pub struct CoordinatorClient {
    base: String,
    http: reqwest::Client,
    wallet: Wallet,
    backoff: Backoff,
}

impl CoordinatorClient {
    pub fn new(base_url: impl Into<String>, wallet: Wallet) -> Self {
        Self {
            // Trailing slash would produce `//v1/...`, which some proxies treat
            // as a different path.
            base: base_url.into().trim_end_matches('/').to_string(),
            // CP-B-006: a bare `Client::new()` has NO timeout (a wedged
            // coordinator hangs the poll loop forever) and follows
            // redirects (a 3xx could re-POST a signed body off-gate).
            // Use the redirect-safe, timeout-bounded builder.
            http: crate::outbound::redirect_safe_client(Duration::from_secs(30)),
            wallet,
            backoff: Backoff::default(),
        }
    }

    pub fn with_backoff(mut self, b: Backoff) -> Self {
        self.backoff = b;
        self
    }

    /// The address this worker is known by. Recovered from its own key, so it is
    /// the same identity the coordinator will derive.
    pub fn worker_id(&self) -> ethereum_types::H160 {
        self.wallet.address()
    }

    fn sign(&self, digest: &[u8; 32]) -> Result<Vec<u8>, ClientError> {
        self.wallet
            .sign_digest_recoverable(digest)
            .map(|s| s.to_vec())
            .map_err(|e| ClientError::Signing(e.to_string()))
    }

    async fn post<T: serde::Serialize>(
        &self,
        path: &str,
        body: &T,
    ) -> Result<reqwest::Response, ClientError> {
        self.http
            .post(format!("{}{path}", self.base))
            .json(body)
            .send()
            .await
            .map_err(|e| ClientError::Transport(e.to_string()))
    }

    /// Present a measured probe and be told what this machine may do.
    ///
    /// `probe_json` must be the bytes as produced by nat's `divergence_probe` —
    /// passed through unmodified, because the signature covers them verbatim.
    pub async fn register(&self, probe_json: &str) -> Result<RegisterResponse, ClientError> {
        let att = Attestation {
            probe_json: probe_json.to_string(),
            signature: self.sign(&attestation_digest(probe_json))?,
        };
        let res = self.post("/v1/register", &att).await?;
        let status = res.status();
        if !status.is_success() {
            return Err(ClientError::Rejected {
                status: status.as_u16(),
                body: res.text().await.unwrap_or_default(),
            });
        }
        res.json::<RegisterResponse>()
            .await
            .map_err(|e| ClientError::Malformed(e.to_string()))
    }

    /// Ask for work. `Ok(None)` means there is none — which is a normal answer,
    /// not a failure.
    pub async fn lease(&self) -> Result<Option<JobSpec>, ClientError> {
        // CP-B-002: bind the request to the current time so a captured lease body
        // is not a forever-replayable bearer credential.
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        let req = LeaseRequest {
            timestamp,
            signature: self.sign(&lease_digest(timestamp))?,
        };
        let res = self.post("/v1/lease", &req).await?;
        let status = res.status();
        if status == reqwest::StatusCode::NO_CONTENT {
            return Ok(None);
        }
        if !status.is_success() {
            return Err(ClientError::Rejected {
                status: status.as_u16(),
                body: res.text().await.unwrap_or_default(),
            });
        }
        // A 200 with a null body would mean the coordinator changed its mind about
        // how "nothing" is spelled; treat it as nothing rather than as malformed.
        match res.json::<Option<JobSpec>>().await {
            Ok(j) => Ok(j),
            Err(e) => Err(ClientError::Malformed(e.to_string())),
        }
    }

    /// Return a result, signed so the coordinator can recover who produced it.
    pub async fn submit(&self, job: &JobId, payload: &str) -> Result<(), ClientError> {
        let sub = SignedSubmission {
            job: job.clone(),
            payload: payload.to_string(),
            signature: self.sign(&submission_digest(job, payload))?,
        };
        let res = self.post("/v1/submit", &sub).await?;
        let status = res.status();
        if !status.is_success() {
            return Err(ClientError::Rejected {
                status: status.as_u16(),
                body: res.text().await.unwrap_or_default(),
            });
        }
        Ok(())
    }

    /// Poll until `should_continue` says otherwise, running each job through
    /// `run`.
    ///
    /// `run` returns the payload to submit. An `Err` from it is logged and the
    /// job is *not* submitted, so the lease expires and the coordinator hands the
    /// work to someone else — which is the correct outcome for a machine that
    /// cannot do it, and is why the coordinator tracks `failed_by`.
    pub async fn poll_loop<F, Fut>(
        &self,
        mut run: F,
        should_continue: impl Fn() -> bool,
    ) -> Result<(), ClientError>
    where
        F: FnMut(JobSpec) -> Fut,
        Fut: std::future::Future<Output = anyhow::Result<String>>,
    {
        let mut wait = self.backoff.initial;
        while should_continue() {
            match self.lease().await {
                Ok(Some(job)) => {
                    // Work arrived, so the next idle wait starts eager again.
                    wait = self.backoff.initial;
                    let id = job.id.clone();
                    tracing::info!(job = %id, "leased");
                    match run(job).await {
                        Ok(payload) => match self.submit(&id, &payload).await {
                            Ok(()) => tracing::info!(job = %id, "submitted"),
                            // Nothing to retry against: the lease is either gone
                            // or was never ours. Let it expire and move on.
                            Err(e) => tracing::error!(job = %id, error = %e, "submit failed"),
                        },
                        Err(e) => {
                            tracing::error!(job = %id, error = %e, "job failed; letting the lease expire")
                        }
                    }
                }
                Ok(None) => {
                    tracing::debug!(?wait, "no work");
                    tokio::time::sleep(wait).await;
                    wait = self.backoff.next(wait);
                }
                // A coordinator restart or a flaky link must not end the worker.
                Err(e) => {
                    tracing::warn!(error = %e, ?wait, "coordinator unreachable; will retry");
                    tokio::time::sleep(wait).await;
                    wait = self.backoff.next(wait);
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    include!("coordinator_client_tests.rs");
}
