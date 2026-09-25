//! Crash-atomic persistence.
//!
//! The coordinator holds the only record of who is doing what. If it loses that
//! on a restart, every in-flight lease becomes a machine training for two days
//! against a job nobody is expecting a result for.
//!
//! The state is small — tens of jobs, a handful of workers — so this writes the
//! whole thing atomically rather than keeping a journal. Simpler is better here:
//! a whole-state rename has one failure mode and it is well understood, whereas a
//! journal has replay semantics to get right and to audit.
//!
//! Atomic means all four steps, in order:
//!
//!   1. write the new state to a temp file in the SAME directory (a rename across
//!      filesystems is not atomic);
//!   2. fsync the temp file, so its contents are on disk before anything points
//!      at it;
//!   3. rename over the target — atomic on POSIX, so a reader sees the old state
//!      or the new one and never a half-written one;
//!   4. **fsync the directory**, so the rename itself survives power loss. This
//!      step is the one that gets left out, and leaving it out means a crash can
//!      revert to the previous state after a successful-looking save.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::state::State;

pub struct Store {
    path: PathBuf,
}

impl Store {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Load, or return an empty state if nothing has been written yet.
    ///
    /// A corrupt file is an error, never a silent reset: starting fresh from an
    /// unreadable state file would drop every live lease without saying so.
    pub fn load(&self) -> io::Result<State> {
        match fs::read(&self.path) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e)),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(State::default()),
            Err(e) => Err(e),
        }
    }

    pub fn save(&self, state: &State) -> io::Result<()> {
        let dir = self
            .path
            .parent()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "path has no parent"))?;
        fs::create_dir_all(dir)?;

        // PBA-L3b-I02 (CodeQL path-injection lead, refuted): `self.path` comes
        // only from the operator's CITRATE_COORDINATOR_STATE (main.rs); no
        // request data ever reaches it.
        // Same directory, so the rename below stays within one filesystem.
        let tmp = self.path.with_extension("tmp");
        {
            use io::Write;
            let mut f = fs::File::create(&tmp)?;
            f.write_all(&serde_json::to_vec_pretty(state)?)?;
            f.sync_all()?;
        }
        fs::rename(&tmp, &self.path)?;

        // Durability of the rename itself, not just of the bytes.
        fs::File::open(dir)?.sync_all()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::{Capability, JobSpec};

    fn tmpdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("citrate-coord-test-{name}"));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn a_missing_file_loads_as_empty_not_as_an_error() {
        let s = Store::new(tmpdir("missing").join("state.json"));
        assert_eq!(s.load().unwrap().jobs.len(), 0);
    }

    #[test]
    fn state_round_trips_through_disk() {
        let s = Store::new(tmpdir("roundtrip").join("state.json"));
        let mut st = State::default();
        st.add_job(JobSpec::new(
            "j1",
            Capability::H01,
            serde_json::json!({"rung": "64M"}),
        ));
        s.save(&st).unwrap();

        let back = s.load().unwrap();
        assert_eq!(back.jobs.len(), 1);
        assert_eq!(
            back.jobs.values().next().unwrap().spec.requires,
            Capability::H01
        );
    }

    #[test]
    fn saving_twice_leaves_no_temp_file_behind() {
        let dir = tmpdir("notmp");
        let s = Store::new(dir.join("state.json"));
        s.save(&State::default()).unwrap();
        s.save(&State::default()).unwrap();
        let left: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        assert_eq!(left, vec!["state.json".to_string()]);
    }

    /// A truncated or corrupt file must surface, not silently reset. Resetting
    /// would drop every live lease and look like a clean start.
    #[test]
    fn a_corrupt_state_file_is_an_error_not_a_silent_reset() {
        let dir = tmpdir("corrupt");
        let p = dir.join("state.json");
        fs::write(&p, b"{ this is not json").unwrap();
        assert!(Store::new(p).load().is_err());
    }
}
