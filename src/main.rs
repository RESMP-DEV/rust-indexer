use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use rust_indexer::Indexer;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

/// Fast local-first semantic + lexical code indexing and search.
///
/// Works with zero configuration in lexical-only (BM25) mode. Set
/// EMBEDDING_URL to any OpenAI-compatible embeddings endpoint to enable
/// hybrid semantic search.
#[derive(Parser)]
#[command(name = "rust-indexer", version, about)]
struct Cli {
    /// Emit machine-readable JSON instead of human-readable text
    #[arg(long, global = true)]
    json: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Build the index for a codebase (walk, split, embed, store)
    Index {
        /// Codebase root directory
        #[arg(default_value = ".")]
        path: PathBuf,
        /// Rebuild from scratch, ignoring the existing manifest
        #[arg(long)]
        force: bool,
    },
    /// Incrementally update an existing index (changed/deleted files only)
    Update {
        /// Codebase root directory
        #[arg(default_value = ".")]
        path: PathBuf,
    },
    /// Search the index with hybrid semantic + lexical retrieval
    Search {
        /// Search query
        query: String,
        /// Codebase root directory
        #[arg(short, long, default_value = ".")]
        path: PathBuf,
        /// Maximum number of results
        #[arg(short = 'k', long, default_value_t = 10)]
        limit: usize,
        /// Restrict results to file extensions (repeatable), e.g. -e rs -e py
        #[arg(short = 'e', long = "ext")]
        extensions: Vec<String>,
    },
    /// Show index status for a codebase
    Status {
        /// Codebase root directory
        #[arg(default_value = ".")]
        path: PathBuf,
    },
    /// Remove the index (vector + lexical) for a codebase
    Clear {
        /// Codebase root directory
        #[arg(default_value = ".")]
        path: PathBuf,
    },
    /// List indexed collections and their row counts
    Collections,
}

fn absolute(path: PathBuf) -> Result<PathBuf> {
    path.canonicalize()
        .with_context(|| format!("cannot resolve path: {}", path.display()))
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    tracing_subscriber::registry()
        .with(fmt::layer().with_writer(std::io::stderr))
        .with(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("rust_indexer=warn,rust-indexer=warn,warn")),
        )
        .init();

    let indexer = Indexer::from_env();

    match cli.command {
        Command::Index { path, force } => {
            let path = absolute(path)?;
            let result = indexer.index(&path, force).await?;
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                println!(
                    "Indexed {} files into {} chunks in {}ms{}",
                    result.files_indexed,
                    result.chunks_created,
                    result.duration_ms,
                    if result.lexical_only {
                        " (lexical-only)"
                    } else {
                        ""
                    }
                );
                for warning in &result.warnings {
                    eprintln!("warning: {warning}");
                }
            }
        }
        Command::Update { path } => {
            let path = absolute(path)?;
            let result = indexer.update(&path).await?;
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                println!(
                    "Updated {} files ({} chunks) in {}ms{}",
                    result.files_indexed,
                    result.chunks_created,
                    result.duration_ms,
                    if result.lexical_only {
                        " (lexical-only)"
                    } else {
                        ""
                    }
                );
                for warning in &result.warnings {
                    eprintln!("warning: {warning}");
                }
            }
        }
        Command::Search {
            query,
            path,
            limit,
            extensions,
        } => {
            let path = absolute(path)?;
            let hits = indexer.search(&path, &query, limit, &extensions).await?;
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&hits)?);
            } else if hits.is_empty() {
                println!("No results.");
            } else {
                for hit in &hits {
                    println!(
                        "{}:{}-{}  [{:.4}] {}",
                        hit.relative_path, hit.start_line, hit.end_line, hit.score, hit.language
                    );
                    for line in hit.content.lines() {
                        println!("  {line}");
                    }
                    println!();
                }
            }
        }
        Command::Status { path } => {
            let path = absolute(path)?;
            let status = indexer.status(&path);
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&status)?);
            } else {
                println!(
                    "state: {:?}\nfiles: {}/{}\nchunks: {}\nembeddings: {}\nvectors: {}",
                    status.status,
                    status.processed_files,
                    status.total_files,
                    status.total_chunks,
                    status.embeddings_generated,
                    status.vectors_inserted
                );
            }
        }
        Command::Clear { path } => {
            let path = absolute(path)?;
            indexer.clear(&path).await?;
            if cli.json {
                println!("{{\"cleared\": true}}");
            } else {
                println!("Cleared index for {}", path.display());
            }
        }
        Command::Collections => {
            let collections = indexer.list_collections().await?;
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&collections)?);
            } else if collections.is_empty() {
                println!("No collections.");
            } else {
                for c in &collections {
                    println!("{}  ({} rows)", c.name, c.row_count);
                }
            }
        }
    }

    Ok(())
}
