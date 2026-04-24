//! citrate-training-worker binary entrypoint.
//!
//! S0 ships with only the library + in-process test suite; the bin
//! is a stub that documents the intended CLI but hasn't been wired
//! to a real ChainClient yet (pending S1).
//!
//! Real CLI shape (planned):
//!
//!   citrate-training-worker \
//!       --job-id 42 \
//!       --rpc-url https://rpc.citrate.ai \
//!       --keystore ~/.citrate/wallet.json \
//!       --shard-index 0 \
//!       --backend tch-gpu
//!
//! For now, run tests via `cargo test -p citrate-training-worker`.

fn main() {
    eprintln!(
        "citrate-training-worker stub (S0). \
         Run `cargo test -p citrate-training-worker` to exercise the \
         in-process 3-worker integration test. Real binary wiring \
         lands in WP-07.2 S1."
    );
    std::process::exit(0);
}
