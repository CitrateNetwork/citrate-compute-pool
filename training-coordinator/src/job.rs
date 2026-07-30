//! What work exists, and who is allowed to be given it.
//!
//! These types are **re-exported, not redefined**. They live in
//! `citrate_training_worker::coordinator_protocol` because both sides of the wire
//! need them and the worker crate is the one this crate already depends on — so
//! there is exactly one definition of the vocabulary and one definition of each
//! signing digest.
//!
//! That matters more than it looks. Two hand-rolled copies of a digest agree
//! until one changes a separator, and then every honest submission fails to
//! authenticate while presenting as a key problem. Re-exporting makes that class
//! of bug unrepresentable rather than merely unlikely.

pub use citrate_training_worker::coordinator_protocol::{Capability, JobId, JobSpec};
