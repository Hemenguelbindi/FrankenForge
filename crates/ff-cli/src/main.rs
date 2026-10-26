//! `frankenforge` CLI: wires the safetensors adapter into the surgery use cases.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, AtomicU64};

use clap::{Parser, Subcommand};
use ff_core::ModelFormat;
use ff_io::Safetensors;
use ff_surgery::{Recipe, build, compare, inspect, lens};

#[derive(Parser)]
#[command(name = "frankenforge", version, about = "Inspect and recombine language-model checkpoints")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Print a model's tensor layout (file or directory of shards)
    Inspect { path: PathBuf },
    /// Compare two models: tensors, blocks, dtype/shape, tokenizers (metadata only)
    Compare {
        a: PathBuf,
        b: PathBuf,
        /// Print the report as JSON instead of text
        #[arg(long)]
        json: bool,
        /// Read the weights instead and print, per block, how far `b` is from `a`
        #[arg(long)]
        weights: bool,
    },
    /// Apply a recipe to metadata only and print the block-level report
    Plan {
        recipe: PathBuf,
        /// Print the report as JSON instead of text
        #[arg(long)]
        json: bool,
        /// Also read the replaced weights and report how far they moved from base
        #[arg(long)]
        weights_diff: bool,
    },
    /// Words the model's word table places next to `word` (reads the table, runs nothing)
    Words {
        model: PathBuf,
        word: String,
        #[arg(long, default_value_t = 12)]
        k: usize,
    },
    /// Words the strongest neurons of a feed-forward "down" matrix push toward (reads weights only)
    Neurons {
        model: PathBuf,
        /// Full tensor name, e.g. model.layers.25.mlp.down_proj.weight
        tensor: String,
        #[arg(long, default_value_t = 16)]
        n: usize,
        #[arg(long, default_value_t = 8)]
        k: usize,
    },
    /// Build the recipe's model, verify it and write a lock file
    Build {
        recipe: PathBuf,
        /// A `*.safetensors` file (weights only) or a new directory (weights + config.json + tokenizer)
        #[arg(short, long)]
        out: PathBuf,
        /// Allow the experimental adapter: no runtime support, no numerical or quality
        /// validation; the result is not a working model
        #[arg(long)]
        allow_experimental_adapter: bool,
    },
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> ff_core::Result<()> {
    let format = Safetensors;
    match cli.cmd {
        Cmd::Inspect { path } => print!("{}", inspect::report(format.open(&path)?.model())),
        Cmd::Compare { a, b, json, weights: true } => {
            let (progress, stop) = (AtomicU64::new(0), AtomicBool::new(false));
            let diff = compare::weights::run(&a, &b, &format, &progress, &stop)?.expect("never stopped");
            print!("{}", if json { diff.json() + "\n" } else { diff.text() });
        }
        Cmd::Words { model, word, k } => {
            let lens = lens::Lens::open(&model, &format)?;
            let (progress, stop) = (AtomicU64::new(0), AtomicBool::new(false));
            let watch = lens::Watch { progress: &progress, stop: &stop };
            let width = lens.source.model().tensor(&lens.table)?.shape[1];
            let hits = lens::nearest(
                &*lens.source,
                &lens.table,
                &lens.vocab,
                lens.word(&word)?,
                0..width,
                k,
                watch,
            )?
            .expect("never stopped");
            println!("next to {word:?} in {}:", lens.table);
            for h in hits {
                println!("  {:<24} {:.2}", format!("{:?}", h.text), h.score);
            }
        }
        Cmd::Neurons { model, tensor, n, k } => {
            let lens = lens::Lens::open(&model, &format)?;
            let (progress, stop) = (AtomicU64::new(0), AtomicBool::new(false));
            let watch = lens::Watch { progress: &progress, stop: &stop };
            let found = lens::neurons(&*lens.source, &tensor, &lens.table, &lens.vocab, n, k, watch)?
                .expect("never stopped");
            println!("{tensor}: words each neuron pushes toward, through {} (weights only):", lens.table);
            for nw in found {
                let words: Vec<String> = nw.words.iter().map(|h| format!("{:?}", h.text)).collect();
                println!("  neuron {:>6}  {}", nw.neuron, words.join(" "));
            }
        }
        Cmd::Compare { a, b, json, weights: false } => {
            let report = compare::run(&a, &b, &format)?;
            print!("{}", if json { report.json() + "\n" } else { report.text() });
        }
        Cmd::Plan { recipe, json, weights_diff } => {
            let r = Recipe::load(&recipe)?;
            let report = build::plan_report(&r, Recipe::base_dir(&recipe), &format, weights_diff)?;
            print!("{}", if json { report.json() + "\n" } else { report.text() });
            if report.experimental.is_some() {
                warn(&["adapter is experimental: build needs --allow-experimental-adapter".into()]);
            }
        }
        Cmd::Build { recipe, out, allow_experimental_adapter } => {
            let report = build::build(&recipe, &out, &format, allow_experimental_adapter)?;
            print!("{}", report.report.text());
            let what = match &report.experimental {
                None => "verified".to_owned(),
                Some(e) => {
                    eprint!("warning: EXPERIMENTAL build, not a working model:\n{}", e.text());
                    "file integrity verified; model NOT validated (experimental)".to_owned()
                }
            };
            println!(
                "wrote {} ({} tensors, {} parameters), {what}; lock {}; report {}",
                out.display(),
                report.tensors,
                report.params,
                report.lock_path.display(),
                report.report_paths[1].display()
            );
        }
    }
    Ok(())
}

fn warn(warnings: &[String]) {
    for w in warnings {
        eprintln!("warning: {w}");
    }
}
