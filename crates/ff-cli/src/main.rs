//! `frankenforge` CLI: wires the safetensors adapter into the surgery use cases.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, AtomicU64};

use clap::{Parser, Subcommand};
use ff_core::ModelFormat;
use ff_io::Safetensors;
use ff_surgery::{Recipe, build, compare, inspect};

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
