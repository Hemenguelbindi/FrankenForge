//! What a checkpoint's weights say in words, read from the weights alone (nothing is run):
//! - `nearest`: which tokens a word table places next to a word (cosine of rows);
//! - `neurons`: which tokens each strong feed-forward neuron pushes the next word toward: the
//!   neuron's column of the "down" matrix scored against every row of the word table (projection
//!   onto the vocabulary). Many neurons read as noise; that is shown, never filtered into sense.
//!
//! Tables are streamed in row chunks, so a 4 GiB table never sits in memory.

mod vocab;

use std::ops::Range;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use ff_core::decode::{Decode, decoder};
use ff_core::{Error, Model, ModelFormat, Result, TensorInfo, TensorSource};
use serde::Serialize;

pub use vocab::Vocab;

/// Rows per read: a few MB for usual widths, tens of MB for an 8960-wide per-layer table.
const ROWS: usize = 2048;

/// Progress in bytes read and a stop flag checked between chunks.
#[derive(Clone, Copy)]
pub struct Watch<'a> {
    pub progress: &'a AtomicU64,
    pub stop: &'a AtomicBool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Hit {
    pub id: usize,
    pub text: String,
    pub score: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NeuronWords {
    pub neuron: usize,
    /// Length of what the neuron writes; neurons are listed strongest first.
    pub strength: f32,
    pub words: Vec<Hit>,
}

/// An opened model with what reading it in words needs.
pub struct Lens {
    pub source: Box<dyn TensorSource>,
    pub vocab: Vocab,
    /// See [`word_table`].
    pub table: String,
}

impl Lens {
    /// `path` is a model folder or a file in one; tokenizer.json is taken from that folder.
    pub fn open(path: &Path, format: &dyn ModelFormat) -> Result<Self> {
        let dir = if path.is_dir() { path } else { path.parent().unwrap_or(Path::new("")) };
        let vocab = Vocab::load(dir)?.ok_or_else(|| {
            Error::Incompatible(format!("{}: no tokenizer.json to read words with", dir.display()))
        })?;
        let source = format.open(path)?;
        let table = word_table(source.model(), vocab.len()).ok_or_else(|| {
            Error::Incompatible("no word table (output head or token embedding) found".into())
        })?;
        Ok(Self { source, vocab, table })
    }

    pub fn word(&self, word: &str) -> Result<usize> {
        self.vocab
            .find(word)
            .ok_or_else(|| Error::Incompatible(format!("`{word}` is not one token of this model")))
    }
}

/// The table words are read through: an output head with a row per token, else a token embedding
/// (most small models share one table for both). Per-layer tables are not word tables.
pub fn word_table(model: &Model, vocab: usize) -> Option<String> {
    let rows_fit = |t: &TensorInfo| t.shape.len() == 2 && t.shape[0] >= vocab && vocab > 0;
    let named = |words: &[&str]| {
        model.tensors.values().find(|t| {
            rows_fit(t) && !t.name.contains("per_layer") && t.name.split('.').any(|seg| words.contains(&seg))
        })
    };
    named(&["lm_head"]).or_else(|| named(&["embed_tokens", "wte", "word_embeddings"])).map(|t| t.name.clone())
}

fn floats(
    src: &dyn TensorSource,
    info: &TensorInfo,
    dec: Decode,
    size: usize,
    rows: Range<usize>,
) -> Result<Vec<f32>> {
    let width = info.shape[1];
    let start = (rows.start * width * size) as u64;
    let len = (rows.len() * width * size) as u64;
    let bytes = src.read_range(&info.name, start, len)?;
    Ok(bytes.chunks_exact(size).map(|b| dec(b) as f32).collect())
}

fn readable(src: &dyn TensorSource, name: &str) -> Result<(TensorInfo, Decode, usize)> {
    let info = src.model().tensor(name)?.clone();
    if info.shape.len() != 2 {
        return Err(Error::Incompatible(format!("{name}: not a matrix")));
    }
    let (size, dec) = decoder(info.dtype)
        .ok_or_else(|| Error::Incompatible(format!("{name}: {:?} is not read", info.dtype)))?;
    Ok((info, dec, size))
}

/// Row ids with their scores, best first.
type Scored = Vec<(usize, f32)>;

/// Top `k` rows of `table` (over `cols`) for every query; `None` if stopped.
fn scan(
    src: &dyn TensorSource,
    table: &str,
    cols: Range<usize>,
    queries: &[Vec<f32>],
    cosine: bool,
    k: usize,
    watch: Watch,
) -> Result<Option<Vec<Scored>>> {
    let (info, dec, size) = readable(src, table)?;
    let (rows, width) = (info.shape[0], info.shape[1]);
    let mut best: Vec<Scored> = vec![Vec::new(); queries.len()];
    for r0 in (0..rows).step_by(ROWS) {
        if watch.stop.load(Ordering::Relaxed) {
            return Ok(None);
        }
        let n = ROWS.min(rows - r0);
        let chunk = floats(src, &info, dec, size, r0..r0 + n)?;
        std::thread::scope(|s| {
            for (q, best) in queries.iter().zip(best.iter_mut()) {
                let (chunk, cols) = (&chunk, cols.clone());
                s.spawn(move || {
                    for r in 0..n {
                        let row = &chunk[r * width..(r + 1) * width][cols.clone()];
                        let mut score: f32 = row.iter().zip(q).map(|(a, b)| a * b).sum();
                        if cosine {
                            score /= row.iter().map(|a| a * a).sum::<f32>().sqrt().max(f32::MIN_POSITIVE);
                        }
                        if score.is_finite() {
                            best.push((r0 + r, score));
                        }
                    }
                    keep_top(best, k);
                });
            }
        });
        watch.progress.fetch_add((n * width * size) as u64, Ordering::Relaxed);
    }
    for b in &mut best {
        b.sort_by(|x, y| y.1.total_cmp(&x.1));
    }
    Ok(Some(best))
}

fn keep_top(v: &mut Vec<(usize, f32)>, k: usize) {
    if v.len() > k {
        v.select_nth_unstable_by(k, |x, y| y.1.total_cmp(&x.1));
        v.truncate(k);
    }
}

/// Best first, at most `k`, one per word: `the`, ` the` and `The` are different tokens but the
/// same word, and listing all of them hides the other words.
fn hits(vocab: &Vocab, found: Vec<(usize, f32)>, k: usize) -> Vec<Hit> {
    let mut seen = std::collections::HashSet::new();
    found
        .into_iter()
        .map(|(id, score)| Hit { id, text: vocab.text(id), score })
        .filter(|h| seen.insert(h.text.trim().to_lowercase()))
        .take(k)
        .collect()
}

/// Candidates kept per query so that `k` distinct words remain after merging variants.
fn pool(k: usize) -> usize {
    k * 4
}

/// Tokens whose rows of `table` (over `cols`, e.g. one block's slice of a per-layer table) point
/// the same way as `word`'s row; the word itself comes first with score 1.
pub fn nearest(
    src: &dyn TensorSource,
    table: &str,
    vocab: &Vocab,
    word: usize,
    cols: Range<usize>,
    k: usize,
    watch: Watch,
) -> Result<Option<Vec<Hit>>> {
    let (info, dec, size) = readable(src, table)?;
    if word >= info.shape[0] || cols.end > info.shape[1] || cols.is_empty() {
        return Err(Error::Incompatible(format!("{table}: row {word} or columns {cols:?} out of range")));
    }
    let row = floats(src, &info, dec, size, word..word + 1)?;
    let q: Vec<f32> = row[cols.clone()].to_vec();
    let norm = q.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return Ok(Some(Vec::new()));
    }
    let q: Vec<f32> = q.iter().map(|x| x / norm).collect();
    Ok(scan(src, table, cols, &[q], true, pool(k), watch)?.map(|mut b| hits(vocab, b.remove(0), k)))
}

/// The `n` strongest neurons of `down` (the matrix that writes a feed-forward layer back into the
/// stream) with the `k` tokens each pushes toward. `down` may be stored `[width, inner]` or
/// `[inner, width]`; `width` is the word table's row length.
pub fn neurons(
    src: &dyn TensorSource,
    down: &str,
    table: &str,
    vocab: &Vocab,
    n: usize,
    k: usize,
    watch: Watch,
) -> Result<Option<Vec<NeuronWords>>> {
    let width = readable(src, table)?.0.shape[1];
    let (info, dec, size) = readable(src, down)?;
    let (a, b) = (info.shape[0], info.shape[1]);
    // `columns`: neuron j is column j of a `[width, inner]` matrix; otherwise row j.
    let (columns, inner) = match (a == width, b == width) {
        (true, _) => (true, b),
        (false, true) => (false, a),
        _ => {
            return Err(Error::Incompatible(format!(
                "{down}: {a}x{b} does not match the word width {width}"
            )));
        }
    };
    let rows = |each: &mut dyn FnMut(usize, &[f32])| each_row(src, &info, dec, size, watch, each);
    let mut strength = vec![0f32; inner];
    let finished = rows(&mut |r, row| {
        if columns {
            row.iter().zip(&mut strength).for_each(|(x, s)| *s += x * x);
        } else {
            strength[r] = row.iter().map(|x| x * x).sum();
        }
    })?;
    if !finished {
        return Ok(None);
    }
    let picked = strongest(&strength, n);
    let mut vectors: Vec<Vec<f32>> = vec![Vec::with_capacity(width); picked.len()];
    let finished = rows(&mut |r, row| {
        if columns {
            picked.iter().zip(&mut vectors).for_each(|(&j, v)| v.push(row[j]));
        } else if let Some(p) = picked.iter().position(|&j| j == r) {
            vectors[p] = row.to_vec();
        }
    })?;
    if !finished {
        return Ok(None);
    }
    let Some(found) = scan(src, table, 0..width, &vectors, false, pool(k), watch)? else { return Ok(None) };
    let words = picked.into_iter().zip(found);
    Ok(Some(
        words
            .map(|(j, f)| NeuronWords { neuron: j, strength: strength[j].sqrt(), words: hits(vocab, f, k) })
            .collect(),
    ))
}

/// Calls `each(row index, row)` for every row of a matrix, read in chunks; false if stopped.
fn each_row(
    src: &dyn TensorSource,
    info: &TensorInfo,
    dec: Decode,
    size: usize,
    watch: Watch,
    each: &mut dyn FnMut(usize, &[f32]),
) -> Result<bool> {
    let (rows, width) = (info.shape[0], info.shape[1]);
    for r0 in (0..rows).step_by(ROWS) {
        if watch.stop.load(Ordering::Relaxed) {
            return Ok(false);
        }
        let n = ROWS.min(rows - r0);
        for (r, row) in floats(src, info, dec, size, r0..r0 + n)?.chunks_exact(width).enumerate() {
            each(r0 + r, row);
        }
        watch.progress.fetch_add((n * width * size) as u64, Ordering::Relaxed);
    }
    Ok(true)
}

fn strongest(strength: &[f32], n: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..strength.len()).collect();
    order.sort_by(|x, y| strength[*y].total_cmp(&strength[*x]));
    order.truncate(n);
    order
}

/// Bytes `nearest` reads (the whole table), for a progress bar.
pub fn nearest_bytes(model: &Model, table: &str) -> u64 {
    model.tensors.get(table).map_or(0, |t| t.byte_len() as u64)
}

/// Bytes `neurons` reads: `down` twice, then the table.
pub fn neurons_bytes(model: &Model, down: &str, table: &str) -> u64 {
    2 * nearest_bytes(model, down) + nearest_bytes(model, table)
}

#[cfg(test)]
#[path = "tests/lens.rs"]
mod tests;
