//! The training coordinator — the missing middle of Gate 4.
//!
//! `nat-federated` verifies and aggregates in-process. `training-worker` trains
//! and signs. `citrate-alf-web` recruits. Nothing connected them, because the
//! component that hands out work and collects signed results did not exist. This
//! is that component.
//!
//! It is deliberately **not** the mesh. `training-worker`'s `Transport` (and its
//! libp2p implementation) already carries step-level gossip for a round; this
//! crate handles the coarser question of which machine is doing which job, which
//! is what a fleet of volunteer GPUs actually needs first.
//!
//! ## Shape
//!
//! - [`attestation`] — a machine proves what it is by signing a measured probe.
//!   Identity is the recovered address, so there is no roster.
//! - [`job`] — what work exists and the capability it demands.
//! - [`state`] — the lease/submit/expire state machine. Pure and synchronous:
//!   time is an argument, so expiry is tested directly rather than with sleeps.
//! - [`store`] — crash-atomic persistence, because losing the lease table means
//!   losing track of machines that are mid-job.
//! - [`submission`] — the exact bytes a worker signs, defined once for both
//!   sides.
//! - [`api`] — the HTTP surface, and the only async part.
//!
//! ## What it does not do yet
//!
//! **Settlement is shadow-only.** Results are recorded and nothing is paid. The
//! worker's own `is_live_settlement()` already defaults to `false`, and the
//! cross-backend divergence measurement (nat `divergence_probe`) has established
//! that exact commitment equality cannot be the settlement primitive across
//! heterogeneous hardware — so the tolerance that replaces it has to be set from
//! fleet data this coordinator is what collects. Wiring payment before that
//! number exists would be picking it by guess.

pub mod api;
pub mod attestation;
pub mod job;
pub mod state;
pub mod store;
pub mod submission;

pub use job::{Capability, JobId, JobSpec};
pub use state::{Counts, JobStatus, State};
