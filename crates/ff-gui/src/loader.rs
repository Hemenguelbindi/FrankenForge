//! Runs a request (opening a checkpoint, analyzing a tensor) on a worker thread so the window
//! keeps drawing.
//!
//! Each request gets its own channel. Starting a new request drops the previous receiver, so a
//! superseded result can never be delivered: only the latest request is ever applied.

use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::thread;

use crate::failure::Failure;

pub type Outcome<T> = Result<T, Failure>;

struct Pending<K, T> {
    key: K,
    rx: Receiver<Outcome<T>>,
}

struct WakeOnDrop<F: FnOnce()>(Option<F>);

impl<F: FnOnce()> Drop for WakeOnDrop<F> {
    fn drop(&mut self) {
        if let Some(wake) = self.0.take() {
            wake();
        }
    }
}

/// `K` identifies a request (a path, a tensor name); `T` is its result.
pub struct Loader<K, T> {
    pending: Option<Pending<K, T>>,
}

impl<K, T> Default for Loader<K, T> {
    fn default() -> Self {
        Self { pending: None }
    }
}

impl<K: Clone + Send + 'static, T: Send + 'static> Loader<K, T> {
    /// Starts `work` for `key`; `wake` is called when the result is ready.
    pub fn start(
        &mut self,
        key: K,
        work: impl FnOnce(&K) -> Outcome<T> + Send + 'static,
        wake: impl FnOnce() + Send + 'static,
    ) {
        let (tx, rx) = mpsc::channel();
        let worker_key = key.clone();
        thread::spawn(move || {
            // Locals drop in reverse order, also when `work` panics: `tx` first (so `poll` sees the
            // result or the disconnect), then the guard wakes the UI.
            let _wake = WakeOnDrop(Some(wake));
            let tx = tx;
            // A send error only means this request was superseded; the result is dropped on purpose.
            let _ = tx.send(work(&worker_key));
        });
        self.pending = Some(Pending { key, rx });
    }

    pub fn pending(&self) -> Option<&K> {
        self.pending.as_ref().map(|p| &p.key)
    }

    /// Forgets the pending request; its result, if it still arrives, is dropped.
    pub fn clear(&mut self) {
        self.pending = None;
    }

    /// The latest request's outcome once it is ready.
    pub fn poll(&mut self) -> Option<(K, Outcome<T>)> {
        let outcome = match self.pending.as_ref()?.rx.try_recv() {
            Ok(outcome) => outcome,
            Err(TryRecvError::Empty) => return None,
            // The worker died before sending (a panic while working).
            Err(TryRecvError::Disconnected) => Err(Failure::crashed()),
        };
        let pending = self.pending.take()?;
        Some((pending.key, outcome))
    }
}

#[cfg(test)]
#[path = "tests/loader.rs"]
mod tests;
