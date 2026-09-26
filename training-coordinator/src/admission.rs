//! Per-signer admission for timestamp-bound requests (lease polls and
//! heartbeats): single use, a per-key rate limit, and a per-key cap on how many
//! used timestamps are remembered.
//!
//! Every limit is per signer, so one key's traffic can only ever get *that*
//! key refused. Used timestamps are kept in a time-ordered index, so dropping
//! the ones that have left the freshness window costs O(expired), never a scan
//! of everything remembered.

use std::collections::{BTreeSet, HashMap};

use ethereum_types::H160;

/// How long a used timestamp must be remembered: a request is accepted up to
/// this far behind the coordinator clock.
pub const WINDOW_PAST_NANOS: u64 =
    citrate_training_worker::coordinator_protocol::LEASE_FRESHNESS_NANOS;

/// How far ahead of the coordinator clock a request may be dated (clock skew).
pub const MAX_FUTURE_SKEW_NANOS: u64 = 30 * 1_000_000_000;

/// Per-key burst: requests a key may make back to back.
pub const PER_KEY_BURST: u64 = 10;

/// Per-key sustained rate: one request per this many nanoseconds. A worker
/// polls at most every few seconds and heartbeats every few minutes.
pub const PER_KEY_REFILL_NANOS: u64 = 2 * 1_000_000_000;

/// Per-key cap on remembered timestamps. At the sustained rate a key cannot
/// reach it inside one window; it only bounds a key's footprint.
pub const PER_KEY_QUOTA: usize = 160;

/// Why a request was refused.
#[derive(Debug, PartialEq, Eq)]
pub enum Refused {
    /// This exact (signer, timestamp) was already used.
    Replay,
    /// This key is sending faster than its rate limit; retry after this many
    /// seconds.
    TooFast { retry_after_secs: u64 },
    /// This key already has its quota of in-window requests remembered.
    OverQuota { retry_after_secs: u64 },
}

#[derive(Debug)]
struct KeyState {
    seen: BTreeSet<u64>,
    tokens: u64,
    last_refill: u64,
}

/// See the module docs.
#[derive(Debug, Default)]
pub struct Admission {
    per_key: HashMap<H160, KeyState>,
    /// (timestamp, signer) of every remembered request, oldest first.
    index: BTreeSet<(u64, H160)>,
}

fn ceil_secs(nanos: u64) -> u64 {
    nanos.div_ceil(1_000_000_000).max(1)
}

impl Admission {
    /// Admit `(who, timestamp)` at `now_nanos`, recording it if admitted.
    pub fn admit(&mut self, who: H160, timestamp: u64, now_nanos: u64) -> Result<(), Refused> {
        self.prune(now_nanos);
        let key = self.per_key.entry(who).or_insert(KeyState {
            seen: BTreeSet::new(),
            tokens: PER_KEY_BURST,
            last_refill: now_nanos,
        });
        if key.seen.contains(&timestamp) {
            return Err(Refused::Replay);
        }
        let earned = now_nanos.saturating_sub(key.last_refill) / PER_KEY_REFILL_NANOS;
        if earned > 0 {
            key.tokens = key.tokens.saturating_add(earned).min(PER_KEY_BURST);
            key.last_refill = key
                .last_refill
                .saturating_add(earned.saturating_mul(PER_KEY_REFILL_NANOS));
        }
        if key.tokens == 0 {
            let next = key.last_refill.saturating_add(PER_KEY_REFILL_NANOS);
            return Err(Refused::TooFast {
                retry_after_secs: ceil_secs(next.saturating_sub(now_nanos)),
            });
        }
        if key.seen.len() >= PER_KEY_QUOTA {
            let oldest = key.seen.first().copied().unwrap_or(now_nanos);
            let frees_at = oldest.saturating_add(WINDOW_PAST_NANOS);
            return Err(Refused::OverQuota {
                retry_after_secs: ceil_secs(frees_at.saturating_sub(now_nanos)),
            });
        }
        key.tokens -= 1;
        key.seen.insert(timestamp);
        self.index.insert((timestamp, who));
        Ok(())
    }

    /// Forget timestamps that have left the window (oldest first, stopping at
    /// the first one still inside it), and keys with nothing left to remember.
    fn prune(&mut self, now_nanos: u64) {
        let cutoff = now_nanos.saturating_sub(WINDOW_PAST_NANOS);
        while let Some(&(ts, who)) = self.index.first() {
            if ts >= cutoff {
                break;
            }
            self.index.pop_first();
            if let Some(k) = self.per_key.get_mut(&who) {
                k.seen.remove(&ts);
                if k.seen.is_empty() {
                    self.per_key.remove(&who);
                }
            }
        }
    }

    /// Remembered timestamps, all keys.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Whether nothing is remembered.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: u64 = 1_000_000_000;

    fn k(i: u64) -> H160 {
        H160::from_low_u64_be(i)
    }

    #[test]
    fn a_used_timestamp_is_refused() {
        let mut a = Admission::default();
        assert_eq!(a.admit(k(1), 100 * S, 100 * S), Ok(()));
        assert_eq!(a.admit(k(1), 100 * S, 100 * S + 1), Err(Refused::Replay));
        assert_eq!(
            a.admit(k(2), 100 * S, 100 * S + 2),
            Ok(()),
            "other key, same ts"
        );
    }

    #[test]
    fn the_rate_limit_is_per_key() {
        let mut a = Admission::default();
        let now = 1_000 * S;
        for i in 0..PER_KEY_BURST {
            assert_eq!(a.admit(k(1), now + i, now), Ok(()));
        }
        assert_eq!(
            a.admit(k(1), now + 99, now),
            Err(Refused::TooFast {
                retry_after_secs: 2
            })
        );
        assert_eq!(a.admit(k(2), now, now), Ok(()), "another key is unaffected");
        // One token per refill interval.
        assert_eq!(a.admit(k(1), now + 100, now + PER_KEY_REFILL_NANOS), Ok(()));
        assert!(matches!(
            a.admit(k(1), now + 101, now + PER_KEY_REFILL_NANOS),
            Err(Refused::TooFast { .. })
        ));
    }

    /// At the per-key rate a key cannot fill its quota inside one window, so
    /// the quota is a footprint bound; drive it directly.
    #[test]
    fn the_quota_is_per_key() {
        let mut q = Admission::default();
        let now = 50_000 * S;
        q.admit(k(7), now, now).unwrap();
        let key = q.per_key.get_mut(&k(7)).unwrap();
        for i in 1..PER_KEY_QUOTA as u64 {
            key.seen.insert(now + i);
        }
        key.tokens = PER_KEY_BURST;
        assert_eq!(
            q.admit(k(7), now + 1_000, now),
            Err(Refused::OverQuota {
                retry_after_secs: WINDOW_PAST_NANOS / S
            })
        );
        assert_eq!(q.admit(k(8), now, now), Ok(()), "another key is unaffected");
        // One below the quota is still admitted.
        let key = q.per_key.get_mut(&k(7)).unwrap();
        key.seen.remove(&(now + 1));
        assert_eq!(q.admit(k(7), now + 1_001, now), Ok(()));
    }

    #[test]
    fn expired_timestamps_and_idle_keys_are_forgotten() {
        let mut a = Admission::default();
        let now = 100 * S;
        a.admit(k(1), now, now).unwrap();
        a.admit(k(2), now + 5 * S, now).unwrap();
        assert_eq!(a.len(), 2);
        // Just inside the window: still remembered.
        let edge = now + WINDOW_PAST_NANOS;
        a.admit(k(3), edge, edge).unwrap();
        assert_eq!(a.admit(k(1), now, edge), Err(Refused::Replay));
        // Past it: forgotten, and the key with it.
        let later = now + WINDOW_PAST_NANOS + 1;
        a.admit(k(3), later, later).unwrap();
        assert!(!a.per_key.contains_key(&k(1)));
        assert!(a.per_key.contains_key(&k(2)), "k2's entry is 5 s younger");
        assert_eq!(a.len(), 3);
        assert!(!a.is_empty());
        assert!(Admission::default().is_empty());
    }
}
