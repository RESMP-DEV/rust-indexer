use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use tokio::task;
use tokio::time::{sleep, Duration};
use tracing::{debug, error, info, instrument, warn};

use crate::config::Config;
use crate::embedding::{Embedder, EmbeddingClient, EmbeddingConfig, RateLimiter};
use crate::engine::hybrid::{fuse_hybrid_hits, HybridFusionOptions, HybridHit};
use crate::engine::indexer::{self, IndexerState};
use crate::engine::state::{ContextState, SharedState};
use crate::lexical::LexicalIndex;
use crate::splitter::{CodeSplitter, Config as SplitterConfig};
use crate::types::{IndexState, IndexStatus};
use crate::vectordb::{collection_name_from_path, VectorStore};
use crate::walker::CodeWalker;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexResult {
    pub files_indexed: usize,
    pub chunks_created: usize,
    pub lexical_only: bool,
    pub duration_ms: u64,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub file_path: PathBuf,
    pub relative_path: String,
    pub content: String,
    pub start_line: u32,
    pub end_line: u32,
    pub language: String,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionInfo {
    pub name: String,
    pub row_count: u64,
}

pub struct Indexer {
    state: SharedState,
}

impl Indexer {
    pub fn new(config: Config) -> Self {
        let embedder_mode = if config.has_embedding_url() {
            "http"
        } else {
            "disabled"
        };
        info!(
            embedder = embedder_mode,
            concurrency = config.concurrency,
            chunk_size = config.chunk_size,
            embedding_dimension = config.embedding_dimension,
            "Indexer initialized"
        );
        Self {
            state: Arc::new(ContextState::new(config)),
        }
    }

    pub fn from_env() -> Self {
        Self::new(Config::from_env())
    }

    pub fn with_components(config: Config, embedder: Embedder, vector_store: VectorStore) -> Self {
        info!(
            concurrency = config.concurrency,
            chunk_size = config.chunk_size,
            "Indexer initialized with explicit components"
        );
        Self {
            state: Arc::new(ContextState::with_components(
                config,
                embedder,
                vector_store,
            )),
        }
    }

    /// Full or manifest-guided index build. `force` rebuilds from scratch.
    #[instrument(skip(self), fields(path = %path.display()))]
    pub async fn index(&self, path: &Path, force: bool) -> Result<IndexResult> {
        self.run_index(path, force, false).await
    }

    /// Incremental update touching only changed/deleted files. Fails if no
    /// compatible index exists rather than silently rebuilding.
    #[instrument(skip(self), fields(path = %path.display()))]
    pub async fn update(&self, path: &Path) -> Result<IndexResult> {
        self.run_index(path, false, true).await
    }

    async fn run_index(
        &self,
        path: &Path,
        force: bool,
        incremental_only: bool,
    ) -> Result<IndexResult> {
        validate_directory(path)?;

        if self.state.is_indexing(path) && !force {
            warn!(path = %path.display(), "Index request rejected: already indexing");
            bail!(
                "Indexing is already running for {}. Use force to rebuild.",
                path.display()
            );
        }

        info!(force, incremental_only, "Starting index operation");
        let start = Instant::now();

        let indexer_state = create_indexer_state(&self.state, path);
        // In-memory only: the engine owns the persisted status. Writing a
        // zeroed status to disk here would destroy the previous run's record
        // (e.g. the vector count guarding lexical-only runs) before the
        // engine can read it.
        self.state.indexing_status.insert(
            path.to_path_buf(),
            IndexStatus {
                status: IndexState::Indexing,
                ..Default::default()
            },
        );

        let state_for_mirror = self.state.clone();
        let is_clone = indexer_state.clone();
        let path_clone = path.to_path_buf();
        let status_mirror = tokio::spawn(async move {
            loop {
                let status = is_clone.get_status().await;
                let done = matches!(status.status, IndexState::Completed | IndexState::Failed);
                // Skip the engine's initial Idle so we never clobber the
                // Indexing marker set above, and stay in-memory: persisting
                // here would overwrite the engine's own status writes (a
                // refused run deliberately restores the pre-run file).
                if status.status != IndexState::Idle {
                    state_for_mirror
                        .indexing_status
                        .insert(path_clone.clone(), status);
                }
                if done {
                    break;
                }
                sleep(Duration::from_millis(250)).await;
            }
        });

        let result = if incremental_only {
            indexer::update_codebase_index(&indexer_state, path).await
        } else {
            indexer::index_codebase(&indexer_state, path, force).await
        };
        // The mirror task only exits on a terminal status; if an error path
        // ever returns while the status still says Indexing, force it to
        // Failed so the CLI reports the error instead of hanging here.
        if indexer_state.get_status().await.status == IndexState::Indexing {
            let mut status = indexer_state.indexing_status.write().await;
            status.status = if result.is_ok() {
                IndexState::Completed
            } else {
                IndexState::Failed
            };
        }
        let _ = status_mirror.await;

        match &result {
            Ok(r) => info!(
                files = r.files_processed,
                chunks = r.chunks_created,
                embeddings = r.embeddings_generated,
                vectors = r.vectors_inserted,
                lexical_only = r.lexical_only,
                duration_ms = r.duration_ms,
                warnings = r.warnings.len(),
                "Index operation completed"
            ),
            Err(e) => {
                error!(error = %e, elapsed_ms = start.elapsed().as_millis() as u64, "Index operation failed")
            }
        }

        let result = result.context("indexing failed")?;

        Ok(IndexResult {
            files_indexed: result.files_processed,
            chunks_created: result.chunks_created,
            lexical_only: result.lexical_only,
            duration_ms: result.duration_ms,
            warnings: result.warnings,
        })
    }

    #[instrument(skip(self, query), fields(path = %path.display(), query_len = query.len()))]
    pub async fn search(
        &self,
        path: &Path,
        query: &str,
        limit: usize,
        extensions: &[String],
    ) -> Result<Vec<SearchHit>> {
        validate_directory(path)?;
        if limit == 0 {
            bail!("limit must be greater than 0");
        }

        let start = Instant::now();
        debug!(limit, extensions = ?extensions, "Search request");

        let collection = collection_name_from_path(path);

        // Semantic hits are only trustworthy if the configured backend is
        // the one the recorded index was built against; after a re-home, a
        // surviving same-named collection in the old backend would fuse
        // stale vectors into current results. Lexical retrieval stays valid
        // either way, so skip semantic instead of failing the search.
        let provenance_ok = match self.state.manifest_store.load_backend(path) {
            Ok(Some(recorded)) => {
                let current = self.state.vector_store.provenance();
                if recorded == current {
                    true
                } else {
                    warn!(
                        recorded_backend = %recorded,
                        current_backend = %current,
                        "Vector backend differs from the recorded index; skipping semantic hits"
                    );
                    false
                }
            }
            _ => true,
        };

        let semantic_start = Instant::now();
        let vector_hits = if provenance_ok && self.state.embedder.is_enabled() {
            let hits = self.state.search(&collection, query, limit).await?;
            debug!(
                count = hits.len(),
                elapsed_ms = semantic_start.elapsed().as_millis() as u64,
                "Semantic search completed"
            );
            hits.into_iter()
                .map(|r| HybridHit {
                    chunk: r.chunk,
                    score: r.score,
                })
                .collect()
        } else {
            debug!("Semantic search skipped (embeddings disabled)");
            Vec::new()
        };

        let lexical_path = path.to_path_buf();
        let lexical_query = query.to_string();
        let lexical_start = Instant::now();
        let lexical_hits = task::spawn_blocking(move || -> Result<Vec<HybridHit>> {
            if !LexicalIndex::exists(&lexical_path)? {
                debug!("No lexical index found at {}", lexical_path.display());
                return Ok(Vec::new());
            }
            let idx = LexicalIndex::open(&lexical_path)?;
            let mut hits = idx.search(&lexical_query, limit)?;
            for hit in &mut hits {
                if hit.chunk.file_path.as_os_str().is_empty() {
                    hit.chunk.file_path = lexical_path.join(&hit.chunk.relative_path);
                }
            }
            Ok(hits)
        })
        .await
        .context("lexical search task panicked")?
        .context("lexical search failed")?;

        debug!(
            lexical_hits = lexical_hits.len(),
            lexical_ms = lexical_start.elapsed().as_millis() as u64,
            "Lexical search completed"
        );

        let options = HybridFusionOptions {
            limit,
            extension_filter: extensions.to_vec(),
        };
        let fused = fuse_hybrid_hits(query, vector_hits, lexical_hits, &options);

        info!(
            results = fused.len(),
            elapsed_ms = start.elapsed().as_millis() as u64,
            "Search completed"
        );

        Ok(fused
            .into_iter()
            .map(|hit| {
                // Semantic hits carry the absolute path recorded by whichever
                // host indexed them; rebuild from the local checkout so shared
                // (identity-scoped) collections resolve to real local files.
                let file_path = if hit.chunk.relative_path.is_empty() {
                    hit.chunk.file_path
                } else {
                    path.join(&hit.chunk.relative_path)
                };
                SearchHit {
                    file_path,
                    relative_path: hit.chunk.relative_path,
                    content: hit.chunk.content,
                    start_line: hit.chunk.start_line,
                    end_line: hit.chunk.end_line,
                    language: hit.chunk.language,
                    score: hit.score,
                }
            })
            .collect())
    }

    pub fn status(&self, path: &Path) -> IndexStatus {
        let status = self.state.get_status(path);
        debug!(path = %path.display(), status = ?status.status, "Status query");
        status
    }

    #[instrument(skip(self), fields(path = %path.display()))]
    pub async fn clear(&self, path: &Path) -> Result<()> {
        info!("Clearing index");
        let start = Instant::now();

        let collection_name = collection_name_from_path(path);
        let had_vector = self
            .state
            .vector_store
            .has_collection(&collection_name)
            .await?;
        // Clearing must not destroy the durable evidence of vectors held in
        // a backend this environment cannot see (e.g. Milvus without
        // MILVUS_URL): the surviving remote collection plus a later rebuilt
        // compatible manifest would present stale semantic results as
        // current.
        let recorded_vectors = self
            .state
            .manifest_store
            .load_status(path)
            .ok()
            .flatten()
            .map(|status| status.vectors_inserted)
            .unwrap_or(0);
        // A same-named collection in the local store is not proof that the
        // recorded vectors are visible: they may live in Milvus while a
        // stale local copy shadows them. For the local backend, require the
        // visible collection to hold at least the recorded count. Milvus
        // row counts lag inserts, so the remote backend keeps the
        // existence-only check (dropping there reaches the real vectors).
        let evidence_visible = match &self.state.vector_store {
            VectorStore::Milvus(_) => had_vector,
            VectorStore::Local(_) => {
                had_vector
                    && self
                        .state
                        .vector_store
                        .collection_stats(&collection_name)
                        .await
                        .map(|stats| stats.row_count as usize >= recorded_vectors)
                        .unwrap_or(false)
            }
        };
        if recorded_vectors > 0 && !evidence_visible {
            bail!(
                "The recorded index for {} has vectors in a backend this environment cannot \
                 see (e.g. Milvus without MILVUS_URL set); configure that backend so clear can \
                 drop the collection, or remove {}/.sindexer/ manually if the vectors are \
                 already gone",
                path.display(),
                path.display()
            );
        }
        if had_vector {
            self.state
                .vector_store
                .drop_collection(&collection_name)
                .await?;
            debug!(collection = collection_name, "Dropped vector collection");
        }
        self.state
            .set_status(path.to_path_buf(), IndexStatus::default());
        let _ = self.state.manifest_store.clear_status(path);
        // The manifest must go with the lexical index: a surviving manifest
        // makes the next index/update report "already up to date" against an
        // empty index.
        self.state
            .manifest_store
            .clear_manifest(path)
            .context("failed to remove index manifest")?;
        let _ = self.state.manifest_store.clear_backend(path);
        self.state.indexing_status.remove(&path.to_path_buf());

        let lexical_path = path.to_path_buf();
        task::spawn_blocking(move || -> Result<()> {
            let idx = LexicalIndex::create(&lexical_path)?;
            idx.clear()?;
            Ok(())
        })
        .await
        .context("lexical clear task panicked")?
        .context("failed to clear lexical index")?;

        info!(
            had_vector_collection = had_vector,
            elapsed_ms = start.elapsed().as_millis() as u64,
            "Index cleared"
        );
        Ok(())
    }

    #[instrument(skip(self))]
    pub async fn list_collections(&self) -> Result<Vec<CollectionInfo>> {
        let names = self.state.vector_store.list_collections().await?;
        debug!(count = names.len(), "Listed collections");
        let mut out = Vec::with_capacity(names.len());
        for name in &names {
            let row_count = self
                .state
                .vector_store
                .collection_stats(name)
                .await
                .map(|s| s.row_count)
                .unwrap_or(0);
            out.push(CollectionInfo {
                name: name.clone(),
                row_count,
            });
        }
        Ok(out)
    }
}

fn validate_directory(path: &Path) -> Result<()> {
    if !path.is_absolute() {
        bail!("path must be absolute: {}", path.display());
    }
    if !path.exists() {
        bail!("path does not exist: {}", path.display());
    }
    if !path.is_dir() {
        bail!("path is not a directory: {}", path.display());
    }
    Ok(())
}

fn create_indexer_state(state: &SharedState, root_path: &Path) -> Arc<IndexerState> {
    let config = &state.config;
    let splitter = CodeSplitter::new(SplitterConfig {
        root_path: root_path.to_path_buf(),
        max_chunk_bytes: config.chunk_size,
        overlap_lines: config.chunk_overlap / 80,
        ..SplitterConfig::default()
    });

    let embedder = if state.embedder.is_enabled() {
        let rate_limiter = RateLimiter::new(config.embedding_rpm, config.embedding_tpm);
        Embedder::Http(EmbeddingClient::with_rate_limiter(
            EmbeddingConfig::from_config(config),
            rate_limiter,
        ))
    } else {
        Embedder::Disabled
    };

    Arc::new(IndexerState::with_concurrency(
        CodeWalker::from_config(config),
        splitter,
        embedder,
        VectorStore::from_config(config),
        config.embedding_dimension,
        config.concurrency,
    ))
}
