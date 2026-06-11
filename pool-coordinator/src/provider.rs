//! HTTPS dispatch to a pool member's `/pool-infer` endpoint.
//!
//! Provider Protocol v1 (the formal ADR-006 lands in WP-05.3) defines
//! the request body shape — for slice 1 we send the same OpenAI-shape
//! body the gateway sends to single-provider `/infer`, since a pool
//! member's worker is the same daemon either way.

use std::time::Duration;

use ethereum_types::U256;
use serde::{Deserialize, Serialize};

use crate::error::CoordinatorError;

/// Request body the coordinator POSTs to `member.endpoint/pool-infer`.
#[derive(Debug, Clone, Serialize)]
pub struct PoolInferRequest {
    pub model: String,
    pub prompt: String,
    pub max_tokens: u32,
    pub job_id: u64,
}

/// Response body the member returns. `output` is the model's text
/// completion; the token counts let the coordinator log usage
/// without re-tokenising.
#[derive(Debug, Clone, Deserialize)]
pub struct PoolInferResponse {
    pub output: String,
    #[serde(default)]
    pub input_tokens: Option<u32>,
    #[serde(default)]
    pub output_tokens: Option<u32>,
}

/// Dispatch one request. Surfaces any non-2xx as
/// `CoordinatorError::ProviderFailed` so the daemon's
/// "fail-after-dispatch" path triggers.
pub async fn dispatch_to_member(
    http: &reqwest::Client,
    endpoint: &str,
    body: &PoolInferRequest,
    timeout: Duration,
) -> Result<PoolInferResponse, CoordinatorError> {
    let resp = http
        .post(endpoint)
        .json(body)
        .timeout(timeout)
        .send()
        .await
        .map_err(|e| CoordinatorError::ProviderFailed(format!("transport: {}", e)))?;
    if !resp.status().is_success() {
        return Err(CoordinatorError::ProviderFailed(format!(
            "HTTP {} from {}",
            resp.status(),
            endpoint
        )));
    }
    let decoded = resp
        .json::<PoolInferResponse>()
        .await
        .map_err(|e| CoordinatorError::ProviderFailed(format!("decode: {}", e)))?;
    validate_pool_infer_response(&decoded)?;
    Ok(decoded)
}

/// SECREM-02 6.3 (CITRATE_COMPUTE_POOL-2026-05-31-003): minimum
/// output validation before the coordinator will `completeJob` (and
/// thereby pay the pool). A 2xx with an empty/whitespace `output`
/// is work NOT done — surfacing it as `ProviderFailed` routes
/// `handle_event` down the `failJob` path so the buyer is refunded.
///
/// Semantic validation (output commitments / challenge hooks) is
/// on-chain INFER-S2 scope; this gate only closes the
/// pay-for-empty-work vector.
pub fn validate_pool_infer_response(
    resp: &PoolInferResponse,
) -> Result<(), CoordinatorError> {
    if resp.output.trim().is_empty() {
        return Err(CoordinatorError::ProviderFailed(
            "provider returned empty output (refusing to complete unperformed work)".into(),
        ));
    }
    Ok(())
}

// `_` so dead-code lint doesn't fire on U256 in this slice (it ships
// for the operator's side of the wire that prices the dispatch).
#[allow(dead_code)]
fn _silence_u256(_: U256) {}

#[cfg(test)]
mod tests {
    use super::*;

    fn resp(output: &str) -> PoolInferResponse {
        PoolInferResponse {
            output: output.to_string(),
            input_tokens: Some(1),
            output_tokens: Some(1),
        }
    }

    /// SECREM-02 6.3 (-003): empty / whitespace output must fail
    /// validation; real output must pass.
    #[test]
    fn validate_rejects_empty_and_whitespace_output() {
        assert!(matches!(
            validate_pool_infer_response(&resp("")),
            Err(CoordinatorError::ProviderFailed(_))
        ));
        assert!(matches!(
            validate_pool_infer_response(&resp("   \n\t")),
            Err(CoordinatorError::ProviderFailed(_))
        ));
        assert!(validate_pool_infer_response(&resp("a real completion")).is_ok());
    }
}
