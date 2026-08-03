//! High-performance parallel codebase indexer.
//!
//! Pipeline: walk → split (rayon) → embed (batched) → insert (streamed).
//! When embeddings are disabled, only the lexical index is populated.

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use futures::stream::{FuturesUnordered, StreamExt};
use rayon::prelude::*;
use tokio::sync::RwLock;
use tokio::task;
use tracing::{debug, info, instrument, warn};

use super::manifest::{diff_manifest_against_files, FileFingerprint, IndexInputs, ManifestStore};
use crate::embedding::Embedder;
use crate::lexical::LexicalIndex;
use crate::splitter::CodeSplitter;
use crate::types::{CodeChunk, EmbeddingVector, IndexState, IndexStatus};
use crate::vectordb::{collection_name_from_path, InsertRow, VectorStore};
use crate::walker::CodeWalker;

const EMBEDDING_BATCH_SIZE: usize = 32;
const INSERT_BATCH_SIZE: usize = 500;
const FILE_PARALLEL_CHUNK_SIZE: usize = 64;

#[derive(Debug, Clone)]
pub struct IndexResult {
    pub files_processed: usize,
    pub chunks_created: usize,
    pub embeddings_generated: usize,
    pub vectors_inserted: usize,
    pub duration_ms: u64,
    pub warnings: Vec<String>,
    pub lexical_only: bool,
}

pub struct IndexerState {
    pub indexing_status: Arc<RwLock<IndexStatus>>,
    pub walker: Arc<CodeWalker>,
    pub splitter: Arc<CodeSplitter>,
    pub embedder: Arc<Embedder>,
    pub vector_store: Arc<VectorStore>,
    pub manifest_store: ManifestStore,
    pub embedding_dimension: usize,
    pub concurrency: usize,
}

impl IndexerState {
    pub fn new(
        walker: CodeWalker,
        splitter: CodeSplitter,
        embedder: Embedder,
        vector_store: VectorStore,
        embedding_dimension: usize,
    ) -> Self {
        Self::with_concurrency(
            walker,
            splitter,
            embedder,
            vector_store,
            embedding_dimension,
            16,
        )
    }

    pub fn with_concurrency(
        walker: CodeWalker,
        splitter: CodeSplitter,
        embedder: Embedder,
        vector_store: VectorStore,
        embedding_dimension: usize,
        concurrency: usize,
    ) -> Self {
        Self {
            indexing_status: Arc::new(RwLock::new(IndexStatus::default())),
            walker: Arc::new(walker),
            splitter: Arc::new(splitter),
            embedder: Arc::new(embedder),
            vector_store: Arc::new(vector_store),
            manifest_store: ManifestStore,
            embedding_dimension,
            concurrency: concurrency.max(1),
        }
    }

    pub async fn get_status(&self) -> IndexStatus {
        self.indexing_status.read().await.clone()
    }
}

#[instrument(skip(state), fields(path = %path.display()))]
pub async fn index_codebase(state: &IndexerState, path: &Path, force: bool) -> Result<IndexResult> {
    run_index_codebase(state, path, force, false).await
}

#[instrument(skip(state), fields(path = %path.display()))]
pub async fn update_codebase_index(state: &IndexerState, path: &Path) -> Result<IndexResult> {
    run_index_codebase(state, path, false, true).await
}

async fn run_index_codebase(
    state: &IndexerState,
    path: &Path,
    force: bool,
    incremental_only: bool,
) -> Result<IndexResult> {
    let start = Instant::now();
    let mut warnings = Vec::new();
    let embeddings_enabled = state.embedder.is_enabled();
    let previous_status_before_index = state.manifest_store.load_status(path).ok().flatten();

    {
        let status = state.indexing_status.read().await;
        if status.status == IndexState::Indexing && (!force || incremental_only) {
            anyhow::bail!("Indexing already in progress");
        }
    }

    {
        let mut status = state.indexing_status.write().await;
        *status = IndexStatus {
            total_files: 0,
            processed_files: 0,
            total_chunks: 0,
            embeddings_generated: 0,
            vectors_inserted: 0,
            status: IndexState::Indexing,
        };
        let _ = state.manifest_store.write_status(path, &status);
    }

    info!("Starting codebase indexing at {}", path.display());

    if incremental_only
        && state.manifest_store.load(path).ok().flatten().is_some()
        && !LexicalIndex::exists(path).unwrap_or(false)
    {
        refuse_before_mutation(state, path, previous_status_before_index.as_ref()).await;
        anyhow::bail!(
            "Incremental update requires an existing lexical index for {}; run index --force to rebuild",
            path.display()
        );
    }

    let collection_name = collection_name_from_path(path);

    // A lexical-only run over a path with a live semantic collection would
    // refresh the shared manifest without touching the vectors, making a
    // later embeddings-enabled run (here or in rust_sindexer) see an empty
    // diff and keep stale semantic results forever. Refuse instead.
    if !embeddings_enabled {
        let has_semantic = match state.vector_store.has_collection(&collection_name).await {
            Ok(has) => has,
            Err(e) => {
                refuse_before_mutation(state, path, previous_status_before_index.as_ref()).await;
                return Err(e).context("Failed to check for an existing vector collection");
            }
        };
        // The configured backend may not be the one holding the semantic
        // index (e.g. MILVUS_URL unset outside the wrapper environment), so
        // also consult the shared status file both tools persist: a recorded
        // vector count is evidence of semantic state we cannot see.
        let status_reports_vectors = previous_status_before_index
            .as_ref()
            .map(|status| status.vectors_inserted > 0)
            .unwrap_or(false);
        // The authenticated provenance record is semantic evidence in its
        // own right: a collection built from an empty repository holds zero
        // vectors, yet advancing the shared manifest past it lexically would
        // let the sibling skip populating it forever.
        let recorded_backend = match state.manifest_store.load_backend(path) {
            Ok(record) => record.is_some(),
            Err(e) => {
                refuse_before_mutation(state, path, previous_status_before_index.as_ref()).await;
                return Err(e).context("Failed to read backend provenance record");
            }
        };
        if has_semantic || status_reports_vectors || recorded_backend {
            refuse_before_mutation(state, path, previous_status_before_index.as_ref()).await;
            anyhow::bail!(
                "A semantic index exists for {} (collection {}, a recorded vector count, or a \
                 backend provenance record) but embeddings are disabled; set EMBEDDING_URL (and \
                 MILVUS_URL if the vectors live in Milvus) to keep it current, or clear the \
                 index before lexical-only indexing",
                path.display(),
                collection_name
            );
        }
    }

    // A backend switch is never resolved by rebuilding here, forced or not:
    // rebuilding advances the shared manifest while the collection in the
    // recorded backend survives, and rust_sindexer reconnecting to that
    // backend would then serve its stale vectors as current. Re-homing goes
    // through `clear` with the recorded backend configured, which actually
    // drops the old collection along with the record.
    if embeddings_enabled {
        let current_backend = state.vector_store.provenance();
        match state.manifest_store.load_backend(path) {
            Ok(Some(recorded)) if recorded != current_backend => {
                refuse_before_mutation(state, path, previous_status_before_index.as_ref()).await;
                anyhow::bail!(
                    "The recorded index was built against vector backend '{}' but the current \
                     backend is '{}'; configure the recorded backend and run clear to retire its \
                     collection before re-indexing here",
                    recorded,
                    current_backend
                );
            }
            Ok(_) => {}
            Err(e) => {
                refuse_before_mutation(state, path, previous_status_before_index.as_ref()).await;
                return Err(e).context("Failed to read backend provenance record");
            }
        }
    }

    let index_inputs = IndexInputs::from_splitter_and_walker(
        state.splitter.config(),
        &state.walker.extensions,
        &state.walker.ignore_patterns,
        state.walker.max_file_size,
        state.walker.follow_symlinks,
        state.embedder.passage_prefix(),
    );

    // Phase 1: Walk files
    let files = match state.walker.walk(path).await {
        Ok(files) => files,
        Err(e) => {
            refuse_before_mutation(state, path, previous_status_before_index.as_ref()).await;
            return Err(e).context("Failed to walk codebase");
        }
    };

    let total_files = files.len();
    info!("Discovered {} files in codebase", total_files);

    let previous_manifest = match state.manifest_store.load(path) {
        Ok(previous) => previous,
        Err(e) => {
            refuse_before_mutation(state, path, previous_status_before_index.as_ref()).await;
            return Err(e).context("Failed to load index manifest");
        }
    };

    let mut full_reindex = force;
    let mut files_to_index = files.clone();
    let mut stale_relative_paths = Vec::new();
    let mut cached_fingerprints: Option<Vec<FileFingerprint>> = None;

    if !force {
        match previous_manifest.as_ref() {
            Some(previous) if previous.matches_index_inputs(&collection_name, &index_inputs) => {
                let (diff, fingerprints) = match diff_manifest_against_files(
                    previous,
                    path,
                    &collection_name,
                    &index_inputs,
                    &files,
                ) {
                    Ok(result) => result,
                    Err(e) => {
                        refuse_before_mutation(state, path, previous_status_before_index.as_ref())
                            .await;
                        return Err(e).context("Failed to diff index manifest");
                    }
                };

                cached_fingerprints = Some(fingerprints);

                let has_collection = if embeddings_enabled {
                    match state.vector_store.has_collection(&collection_name).await {
                        Ok(has) => has,
                        Err(e) => {
                            refuse_before_mutation(
                                state,
                                path,
                                previous_status_before_index.as_ref(),
                            )
                            .await;
                            return Err(e).context("Failed to check vector collection existence");
                        }
                    }
                } else {
                    true
                };
                if embeddings_enabled && !has_collection {
                    if incremental_only {
                        refuse_before_mutation(state, path, previous_status_before_index.as_ref())
                            .await;
                        anyhow::bail!(
                            "Incremental update requires existing vector collection {}; run index_codebase only when a full rebuild is intended",
                            collection_name
                        );
                    }
                    full_reindex = true;
                }

                if diff.is_empty() && !full_reindex {
                    let previous_status = previous_status_before_index.clone().unwrap_or_default();
                    let vectors_inserted = if embeddings_enabled {
                        if let Some(count) = completed_vector_count(&previous_status) {
                            count
                        } else {
                            vector_row_count(state, &collection_name)
                                .await
                                .unwrap_or(previous_status.vectors_inserted)
                        }
                    } else {
                        0
                    };
                    {
                        let mut status = state.indexing_status.write().await;
                        status.total_files = total_files;
                    }
                    update_status_completed_counts(
                        state,
                        path,
                        previous_status.total_chunks,
                        previous_status.embeddings_generated,
                        vectors_inserted,
                    )
                    .await;
                    if embeddings_enabled {
                        if let Err(e) = state
                            .manifest_store
                            .write_backend(path, &state.vector_store.provenance())
                        {
                            update_status_failed(
                                state,
                                path,
                                previous_status_before_index.as_ref(),
                            )
                            .await;
                            return Err(e).context("Failed to record vector backend provenance");
                        }
                    }
                    return Ok(IndexResult {
                        files_processed: 0,
                        chunks_created: 0,
                        embeddings_generated: 0,
                        vectors_inserted,
                        duration_ms: start.elapsed().as_millis() as u64,
                        warnings: vec!["already up to date".to_string()],
                        lexical_only: !embeddings_enabled,
                    });
                }

                if !full_reindex {
                    let changed_relative_paths = diff
                        .added
                        .iter()
                        .chain(diff.modified.iter())
                        .cloned()
                        .collect::<std::collections::BTreeSet<_>>();
                    files_to_index = files
                        .iter()
                        .filter(|file_path| {
                            changed_relative_paths.contains(&relative_path(path, file_path))
                        })
                        .cloned()
                        .collect();

                    stale_relative_paths = diff
                        .deleted
                        .iter()
                        .chain(diff.modified.iter())
                        .cloned()
                        .collect();
                }
            }
            Some(_) => {
                if incremental_only {
                    refuse_before_mutation(state, path, previous_status_before_index.as_ref())
                        .await;
                    anyhow::bail!(
                        "Incremental update requires a compatible index manifest; run index_codebase only when a full rebuild is intended"
                    );
                }
                full_reindex = true;
            }
            None => {
                if incremental_only {
                    refuse_before_mutation(state, path, previous_status_before_index.as_ref())
                        .await;
                    anyhow::bail!(
                        "Incremental update requires an existing index manifest; run index_codebase first only when a full build is intended"
                    );
                }
                full_reindex = true;
            }
        }
    }

    if embeddings_enabled {
        if let Err(e) = prepare_vector_index(state, &collection_name, full_reindex).await {
            update_status_failed(
                state,
                path,
                if full_reindex {
                    None
                } else {
                    previous_status_before_index.as_ref()
                },
            )
            .await;
            return Err(e).context("Failed to prepare vector collection");
        }
    }

    if let Err(e) = prepare_lexical_index(path, &stale_relative_paths, full_reindex).await {
        update_status_failed(
            state,
            path,
            if full_reindex {
                None
            } else {
                previous_status_before_index.as_ref()
            },
        )
        .await;
        return Err(e).context("Failed to prepare lexical index");
    }

    if !stale_relative_paths.is_empty() && embeddings_enabled {
        if let Err(e) = state
            .vector_store
            .delete_by_relative_paths(&collection_name, &stale_relative_paths)
            .await
        {
            update_status_failed(
                state,
                path,
                if full_reindex {
                    None
                } else {
                    previous_status_before_index.as_ref()
                },
            )
            .await;
            return Err(e).context("Failed to delete stale vectors");
        }
    }

    {
        let mut status = state.indexing_status.write().await;
        status.total_files = files_to_index.len();
    }

    if files_to_index.is_empty() {
        if let Err(e) = write_manifest(
            state,
            path,
            &collection_name,
            &index_inputs,
            &files,
            cached_fingerprints.take(),
        ) {
            update_status_failed(
                state,
                path,
                if full_reindex {
                    None
                } else {
                    previous_status_before_index.as_ref()
                },
            )
            .await;
            return Err(e).context("Failed to write index manifest");
        }

        let vectors_inserted = if embeddings_enabled {
            vector_row_count(state, &collection_name).await.unwrap_or(0)
        } else {
            0
        };
        if embeddings_enabled {
            if let Err(e) = state
                .manifest_store
                .write_backend(path, &state.vector_store.provenance())
            {
                update_status_failed(
                    state,
                    path,
                    failure_evidence(full_reindex, previous_status_before_index.as_ref()),
                )
                .await;
                return Err(e).context("Failed to record vector backend provenance");
            }
        }
        {
            let mut status = state.indexing_status.write().await;
            status.total_files = total_files;
        }
        update_status_completed_counts(
            state,
            path,
            if embeddings_enabled {
                vectors_inserted
            } else {
                0
            },
            if embeddings_enabled {
                vectors_inserted
            } else {
                0
            },
            vectors_inserted,
        )
        .await;
        return Ok(IndexResult {
            files_processed: 0,
            chunks_created: 0,
            embeddings_generated: 0,
            vectors_inserted: 0,
            duration_ms: start.elapsed().as_millis() as u64,
            warnings,
            lexical_only: !embeddings_enabled,
        });
    }

    // Phase 2: Split files into chunks (rayon)
    let processed_files = Arc::new(AtomicUsize::new(0));
    let splitter = state.splitter.clone();
    let status_ref = state.indexing_status.clone();
    let processed_ref = processed_files.clone();

    let chunk_results: Vec<Result<Vec<CodeChunk>, String>> = files_to_index
        .par_chunks(FILE_PARALLEL_CHUNK_SIZE)
        .flat_map(|file_batch| {
            file_batch
                .par_iter()
                .map(|file_path| match splitter.split_file(file_path) {
                    Ok(chunks) => {
                        let count = processed_ref.fetch_add(1, Ordering::Relaxed) + 1;
                        if count.is_multiple_of(10) {
                            if let Ok(mut status) = status_ref.try_write() {
                                status.processed_files = count;
                                let snapshot = status.clone();
                                drop(status);
                                let _ = state.manifest_store.write_status(path, &snapshot);
                            }
                        }
                        Ok(chunks)
                    }
                    Err(e) => {
                        debug!("Failed to split {}: {}", file_path.display(), e);
                        Err(format!("Failed to split {}: {}", file_path.display(), e))
                    }
                })
        })
        .collect();

    let mut all_chunks = Vec::new();
    for result in chunk_results {
        match result {
            Ok(chunks) => all_chunks.extend(chunks),
            Err(warning) => warnings.push(warning),
        }
    }

    let total_chunks = all_chunks.len();
    info!(
        "Split {} files into {} chunks ({} warnings)",
        files_to_index.len(),
        total_chunks,
        warnings.len()
    );

    {
        let mut status = state.indexing_status.write().await;
        status.processed_files = files_to_index.len();
        status.total_chunks = total_chunks;
        let _ = state.manifest_store.write_status(path, &status);
    }

    // Phase 3: Lexical index + embedding pipeline run concurrently
    let lexical_chunks = all_chunks.clone();
    let lexical_path = path.to_path_buf();
    let lexical_handle = task::spawn_blocking(move || -> Result<()> {
        let lexical_index = LexicalIndex::create(&lexical_path)?;
        lexical_index.insert_chunks(&lexical_chunks)?;
        Ok(())
    });

    let split_warnings = warnings.clone();
    let embedding_handle = async {
        if total_chunks == 0 || !embeddings_enabled {
            return Ok((0usize, 0usize, Vec::new()));
        }

        let embedder = state.embedder.clone();
        let vector_store = state.vector_store.clone();

        let chunk_batches: Vec<_> = all_chunks.chunks(EMBEDDING_BATCH_SIZE).collect();
        let num_embedding_batches = chunk_batches.len();
        info!(
            "Streaming embeddings + inserts in {} batches of {}",
            num_embedding_batches, EMBEDDING_BATCH_SIZE
        );

        let semaphore = Arc::new(tokio::sync::Semaphore::new(state.concurrency));
        let mut batch_handles = FuturesUnordered::new();
        let mut batch_failures = Vec::new();
        let mut embedding_warnings = Vec::new();
        let mut embeddings_generated = 0usize;
        let mut vectors_inserted = 0usize;

        for batch in chunk_batches {
            let batch_chunks: Vec<CodeChunk> = batch.to_vec();
            let emb = embedder.clone();
            let vs = vector_store.clone();
            let collection = collection_name.clone();
            let permit = semaphore.clone().acquire_owned().await?;

            batch_handles.push(tokio::spawn(async move {
                let (embedded_chunks, warnings) =
                    embed_chunks_lossy(emb.as_ref(), batch_chunks).await;
                let embedded = embedded_chunks.len();
                let insert_rows: Vec<InsertRow> = embedded_chunks
                    .into_iter()
                    .map(|(chunk, embedding)| InsertRow {
                        id: chunk.id.clone(),
                        content: chunk.content.clone(),
                        vector: embedding.vector,
                        metadata: crate::vectordb::ChunkMeta {
                            file_path: chunk.file_path,
                            relative_path: chunk.relative_path,
                            start_line: chunk.start_line,
                            end_line: chunk.end_line,
                            language: chunk.language,
                        },
                    })
                    .collect();

                let mut inserted = 0usize;
                for rows in insert_rows.chunks(INSERT_BATCH_SIZE) {
                    inserted += vs.insert_batch(&collection, rows).await?;
                }

                drop(permit);
                Ok::<(usize, usize, Vec<String>), anyhow::Error>((embedded, inserted, warnings))
            }));

            if batch_handles.len() >= state.concurrency {
                if let Some(handle) = batch_handles.next().await {
                    record_embedding_batch_result(
                        handle,
                        state,
                        path,
                        &mut embeddings_generated,
                        &mut vectors_inserted,
                        &mut embedding_warnings,
                        &mut batch_failures,
                    )
                    .await;
                }
            }
        }

        while let Some(handle) = batch_handles.next().await {
            record_embedding_batch_result(
                handle,
                state,
                path,
                &mut embeddings_generated,
                &mut vectors_inserted,
                &mut embedding_warnings,
                &mut batch_failures,
            )
            .await;
        }

        if !batch_failures.is_empty() {
            anyhow::bail!(
                "Streaming index failed after inserting {} vectors: {}",
                vectors_inserted,
                batch_failures.join("; ")
            );
        }

        if embeddings_generated == 0 {
            anyhow::bail!("Failed to generate any embeddings");
        }

        if vectors_inserted == 0 {
            let mut all_warnings = split_warnings.clone();
            all_warnings.extend(embedding_warnings.clone());
            anyhow::bail!(
                "Generated {} chunks and {} embeddings, but inserted 0 vectors{}",
                total_chunks,
                embeddings_generated,
                if all_warnings.is_empty() {
                    String::new()
                } else {
                    format!(" ({})", all_warnings.join("; "))
                }
            );
        }

        Ok((embeddings_generated, vectors_inserted, embedding_warnings))
    };

    let (lexical_result, embedding_result) = tokio::join!(lexical_handle, embedding_handle);

    if let Err(e) = lexical_result
        .context("Lexical index task panicked")
        .and_then(|r| r)
    {
        update_status_failed(
            state,
            path,
            if full_reindex {
                None
            } else {
                previous_status_before_index.as_ref()
            },
        )
        .await;
        return Err(e).context("Failed to update lexical index");
    }

    let (embeddings_generated, vectors_inserted, mut embedding_warnings) = match embedding_result {
        Ok(counts) => counts,
        Err(e) => {
            update_status_failed(
                state,
                path,
                if full_reindex {
                    None
                } else {
                    previous_status_before_index.as_ref()
                },
            )
            .await;
            return Err(e);
        }
    };
    warnings.append(&mut embedding_warnings);
    if embeddings_enabled {
        if let Some(row_count) = vector_row_count(state, &collection_name).await {
            if full_reindex && row_count != vectors_inserted {
                warnings.push(format!(
                    "Vector store stats report {} rows after {} accepted inserts; preserving the insert response count",
                    row_count, vectors_inserted
                ));
            }
        }
    }
    if embeddings_enabled && vectors_inserted < embeddings_generated {
        warnings.push(format!(
            "Vector store accepted {} rows for {} generated embeddings",
            vectors_inserted, embeddings_generated
        ));
    }

    let completed_chunks = if embeddings_enabled && !full_reindex {
        vectors_inserted
    } else {
        total_chunks
    };
    let completed_embeddings = if embeddings_enabled && !full_reindex {
        vectors_inserted
    } else {
        embeddings_generated
    };

    if let Err(e) = write_manifest(
        state,
        path,
        &collection_name,
        &index_inputs,
        &files,
        cached_fingerprints.take(),
    ) {
        update_status_failed(
            state,
            path,
            if full_reindex {
                None
            } else {
                previous_status_before_index.as_ref()
            },
        )
        .await;
        return Err(e).context("Failed to write index manifest");
    }

    if embeddings_enabled {
        if let Err(e) = state
            .manifest_store
            .write_backend(path, &state.vector_store.provenance())
        {
            update_status_failed(
                state,
                path,
                failure_evidence(full_reindex, previous_status_before_index.as_ref()),
            )
            .await;
            return Err(e).context("Failed to record vector backend provenance");
        }
    }
    {
        let mut status = state.indexing_status.write().await;
        status.total_files = total_files;
    }
    update_status_completed_counts(
        state,
        path,
        completed_chunks,
        completed_embeddings,
        vectors_inserted,
    )
    .await;

    let duration_ms = start.elapsed().as_millis() as u64;
    info!(
        "Indexing completed in {}ms: {} files, {} chunks, {} vectors (lexical_only={})",
        duration_ms, total_files, total_chunks, vectors_inserted, !embeddings_enabled
    );

    Ok(IndexResult {
        files_processed: files_to_index.len(),
        chunks_created: total_chunks,
        embeddings_generated,
        vectors_inserted,
        duration_ms,
        warnings,
        lexical_only: !embeddings_enabled,
    })
}

pub fn spawn_index_codebase(
    state: Arc<IndexerState>,
    path: std::path::PathBuf,
    force: bool,
) -> tokio::task::JoinHandle<Result<IndexResult>> {
    tokio::spawn(async move { index_codebase(&state, &path, force).await })
}

async fn record_embedding_batch_result(
    handle: std::result::Result<
        anyhow::Result<(usize, usize, Vec<String>)>,
        tokio::task::JoinError,
    >,
    state: &IndexerState,
    path: &Path,
    embeddings_generated: &mut usize,
    vectors_inserted: &mut usize,
    embedding_warnings: &mut Vec<String>,
    batch_failures: &mut Vec<String>,
) {
    match handle {
        Ok(Ok((embedded, inserted, mut warnings))) => {
            *embeddings_generated += embedded;
            *vectors_inserted += inserted;
            embedding_warnings.append(&mut warnings);
            info!(
                "Streaming progress: embeddings_generated={} vectors_inserted={}",
                *embeddings_generated, *vectors_inserted
            );
            let mut status = state.indexing_status.write().await;
            status.embeddings_generated = *embeddings_generated;
            status.vectors_inserted = *vectors_inserted;
            let _ = state.manifest_store.write_status(path, &status);
        }
        Ok(Err(e)) => {
            warn!("Streaming batch failed: {}", e);
            batch_failures.push(format!("Streaming batch failed: {}", e));
        }
        Err(e) => {
            warn!("Streaming batch task panicked: {}", e);
            batch_failures.push(format!("Streaming batch task panicked: {}", e));
        }
    }
}

async fn embed_chunks_lossy(
    embedder: &Embedder,
    chunks: Vec<CodeChunk>,
) -> (Vec<(CodeChunk, EmbeddingVector)>, Vec<String>) {
    let mut pending = vec![chunks];
    let mut embedded_chunks = Vec::new();
    let mut warnings = Vec::new();

    while let Some(batch) = pending.pop() {
        if batch.is_empty() {
            continue;
        }

        let batch_size = batch.len();
        let texts: Vec<String> = batch.iter().map(|chunk| chunk.content.clone()).collect();
        match embedder.embed_batch(&texts).await {
            Ok(embeddings) if embeddings.len() == batch_size => {
                embedded_chunks.extend(batch.into_iter().zip(embeddings));
            }
            Ok(embeddings) => {
                let err = format!(
                    "embedding API returned {} results for {} inputs",
                    embeddings.len(),
                    batch_size
                );
                split_or_skip_embedding_batch(batch, err, &mut pending, &mut warnings);
            }
            Err(err) => {
                split_or_skip_embedding_batch(batch, err.to_string(), &mut pending, &mut warnings);
            }
        }
    }

    (embedded_chunks, warnings)
}

fn split_or_skip_embedding_batch(
    mut batch: Vec<CodeChunk>,
    err: String,
    pending: &mut Vec<Vec<CodeChunk>>,
    warnings: &mut Vec<String>,
) {
    if batch.len() == 1 {
        let chunk = batch.pop().expect("single chunk exists");
        warnings.push(format!(
            "Skipped embedding for {}:{}-{}: {}",
            chunk.relative_path, chunk.start_line, chunk.end_line, err
        ));
        return;
    }

    warn!(
        error = %err,
        batch_size = batch.len(),
        "Embedding batch failed; splitting batch to isolate unembeddable chunks"
    );
    let right = batch.split_off(batch.len() / 2);
    pending.push(right);
    pending.push(batch);
}

/// Evidence to preserve on failure: the pre-run status, unless this run
/// dropped the collection (full reindex), in which case partial counts are
/// the truth.
fn failure_evidence(full_reindex: bool, previous: Option<&IndexStatus>) -> Option<&IndexStatus> {
    if full_reindex {
        None
    } else {
        previous
    }
}

/// Persist a Failed status for a run that failed after mutations began.
/// `prior_evidence` (the pre-run status, None when this run dropped the
/// collection) keeps the durable vector counts from being zeroed by an
/// early failure: remote vectors still exist, and the lexical-only guard
/// depends on the recorded count to know that.
async fn update_status_failed(
    state: &IndexerState,
    path: &Path,
    prior_evidence: Option<&IndexStatus>,
) {
    let mut status = state.indexing_status.write().await;
    status.status = IndexState::Failed;
    if let Some(previous) = prior_evidence {
        status.vectors_inserted = status.vectors_inserted.max(previous.vectors_inserted);
        status.embeddings_generated = status
            .embeddings_generated
            .max(previous.embeddings_generated);
    }
    let _ = state.manifest_store.write_status(path, &status);
}

/// Fail a run that was refused before mutating any index state. The failure
/// is recorded in memory, but the persisted status is restored to its
/// pre-run value: a zeroed Failed status would destroy durable evidence
/// (such as a recorded vector count) that later runs depend on.
async fn refuse_before_mutation(
    state: &IndexerState,
    path: &Path,
    previous_status: Option<&IndexStatus>,
) {
    {
        let mut status = state.indexing_status.write().await;
        status.status = IndexState::Failed;
    }
    match previous_status {
        Some(previous) => {
            let _ = state.manifest_store.write_status(path, previous);
        }
        None => {
            let _ = state.manifest_store.clear_status(path);
        }
    }
}

async fn update_status_completed_counts(
    state: &IndexerState,
    path: &Path,
    chunks: usize,
    embeddings_generated: usize,
    vectors_inserted: usize,
) {
    let mut status = state.indexing_status.write().await;
    status.total_chunks = chunks;
    status.processed_files = status.total_files;
    status.embeddings_generated = embeddings_generated;
    status.vectors_inserted = vectors_inserted;
    status.status = IndexState::Completed;
    let _ = state.manifest_store.write_status(path, &status);
}

async fn vector_row_count(state: &IndexerState, collection_name: &str) -> Option<usize> {
    match state.vector_store.collection_stats(collection_name).await {
        Ok(stats) => Some(checked_vector_row_count(stats.row_count)),
        Err(err) => {
            debug!(
                collection = collection_name,
                error = %err,
                "Unable to read vector row count"
            );
            None
        }
    }
}

fn completed_vector_count(status: &IndexStatus) -> Option<usize> {
    (status.status == IndexState::Completed && status.vectors_inserted > 0)
        .then_some(status.vectors_inserted)
}

fn checked_vector_row_count(row_count: u64) -> usize {
    match usize::try_from(row_count) {
        Ok(count) => count,
        Err(_) => {
            warn!(
                row_count,
                "Vector row count exceeds this platform's usize; capping status counters"
            );
            usize::MAX
        }
    }
}

async fn prepare_vector_index(
    state: &IndexerState,
    collection_name: &str,
    full_reindex: bool,
) -> Result<()> {
    if full_reindex && state.vector_store.has_collection(collection_name).await? {
        state.vector_store.drop_collection(collection_name).await?;
    }

    if !state.vector_store.has_collection(collection_name).await? {
        state
            .vector_store
            .create_collection(collection_name, state.embedding_dimension)
            .await?;
    }

    Ok(())
}

async fn prepare_lexical_index(
    path: &Path,
    stale_relative_paths: &[String],
    full_reindex: bool,
) -> Result<()> {
    let lexical_index = LexicalIndex::create(path)?;
    if full_reindex {
        lexical_index.clear()?;
    } else {
        lexical_index.delete_by_paths(stale_relative_paths)?;
    }
    Ok(())
}

fn write_manifest(
    state: &IndexerState,
    path: &Path,
    collection_name: &str,
    index_inputs: &IndexInputs,
    files: &[std::path::PathBuf],
    cached_fingerprints: Option<Vec<FileFingerprint>>,
) -> Result<()> {
    match cached_fingerprints {
        Some(fp) => {
            state
                .manifest_store
                .write_with_fingerprints(path, collection_name, index_inputs, fp)
        }
        None => state
            .manifest_store
            .write_for_files(path, collection_name, index_inputs, files),
    }
}

fn relative_path(root: &Path, file_path: &Path) -> String {
    file_path
        .strip_prefix(root)
        .unwrap_or(file_path)
        .to_string_lossy()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::fs;
    use std::io;

    use serde_json::json;
    use tempfile::TempDir;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    use crate::config::Config;
    use crate::embedding::{EmbeddingClient, EmbeddingConfig};
    use crate::lexical::test_support::set_test_cache_dir_async;
    use crate::splitter::{CodeSplitter, Config as SplitterConfig};
    use crate::walker::CodeWalker;

    struct MockHttpServer {
        base_url: String,
        handle: tokio::task::JoinHandle<()>,
    }

    #[test]
    fn test_completed_vector_count_preserves_nonzero_accepted_count() {
        let completed = IndexStatus {
            vectors_inserted: 33_345,
            status: IndexState::Completed,
            ..IndexStatus::default()
        };
        assert_eq!(completed_vector_count(&completed), Some(33_345));

        let indexing = IndexStatus {
            vectors_inserted: 33_345,
            status: IndexState::Indexing,
            ..IndexStatus::default()
        };
        assert_eq!(completed_vector_count(&indexing), None);
        assert_eq!(
            completed_vector_count(&IndexStatus {
                status: IndexState::Completed,
                ..IndexStatus::default()
            }),
            None
        );
    }

    impl MockHttpServer {
        async fn wait(self) {
            self.handle.await.unwrap();
        }
    }

    async fn spawn_mock_embedding_server(response_body: serde_json::Value) -> MockHttpServer {
        spawn_mock_json_server(HashMap::from([("/v1/embeddings", response_body)])).await
    }

    async fn spawn_dynamic_mock_embedding_server() -> MockHttpServer {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let handle = tokio::spawn(async move {
            loop {
                let accept =
                    tokio::time::timeout(std::time::Duration::from_millis(500), listener.accept())
                        .await;
                let Ok(Ok((mut stream, _))) = accept else {
                    break;
                };

                let request = read_http_request(&mut stream).await.unwrap();
                let request_line = request.lines().next().unwrap_or_default();

                let (status, response_body) = if request_line.contains("/v1/embeddings") {
                    let body = request.split("\r\n\r\n").nth(1).unwrap_or_default();
                    let input_count = serde_json::from_str::<serde_json::Value>(body)
                        .ok()
                        .and_then(|payload| {
                            payload
                                .get("input")
                                .and_then(|v| v.as_array())
                                .map(|v| v.len())
                        })
                        .unwrap_or(0);
                    let response = json!({
                        "data": (0..input_count)
                            .map(|idx| json!({ "embedding": [idx as f32 + 0.1, 0.2, 0.3, 0.4] }))
                            .collect::<Vec<_>>()
                    });
                    ("200 OK", response.to_string())
                } else {
                    (
                        "404 Not Found",
                        json!({ "code": 404, "message": "not found" }).to_string(),
                    )
                };

                let response = format!(
                    "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    response_body.len(),
                    response_body
                );
                stream.write_all(response.as_bytes()).await.unwrap();
            }
        });

        MockHttpServer {
            base_url: format!("http://{addr}"),
            handle,
        }
    }

    async fn spawn_selective_embedding_server() -> MockHttpServer {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let handle = tokio::spawn(async move {
            loop {
                let accept =
                    tokio::time::timeout(std::time::Duration::from_millis(500), listener.accept())
                        .await;
                let Ok(Ok((mut stream, _))) = accept else {
                    break;
                };

                let request = read_http_request(&mut stream).await.unwrap();
                let request_line = request.lines().next().unwrap_or_default();

                let (status, response_body) = if request_line.contains("/v1/embeddings") {
                    let body = request.split("\r\n\r\n").nth(1).unwrap_or_default();
                    let inputs = serde_json::from_str::<serde_json::Value>(body)
                        .ok()
                        .and_then(|payload| {
                            payload.get("input").and_then(|v| v.as_array()).cloned()
                        })
                        .unwrap_or_default();
                    let has_bad_input = inputs.iter().any(|input| {
                        input
                            .as_str()
                            .is_some_and(|text| text.contains("bad_embedding_payload"))
                    });

                    if has_bad_input {
                        (
                            "400 Bad Request",
                            json!({
                                "detail": {
                                    "message": "Failed to encode text: input too large"
                                }
                            })
                            .to_string(),
                        )
                    } else {
                        let response = json!({
                            "data": (0..inputs.len())
                                .map(|idx| json!({ "embedding": [idx as f32 + 0.1, 0.2, 0.3, 0.4] }))
                                .collect::<Vec<_>>()
                        });
                        ("200 OK", response.to_string())
                    }
                } else {
                    (
                        "404 Not Found",
                        json!({ "code": 404, "message": "not found" }).to_string(),
                    )
                };

                let response = format!(
                    "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    response_body.len(),
                    response_body
                );
                stream.write_all(response.as_bytes()).await.unwrap();
            }
        });

        MockHttpServer {
            base_url: format!("http://{addr}"),
            handle,
        }
    }

    async fn spawn_mock_json_server(
        responses: HashMap<&'static str, serde_json::Value>,
    ) -> MockHttpServer {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let response_bodies: HashMap<&'static str, String> = responses
            .into_iter()
            .map(|(path, body)| (path, body.to_string()))
            .collect();

        let handle = tokio::spawn(async move {
            loop {
                let accept =
                    tokio::time::timeout(std::time::Duration::from_millis(500), listener.accept())
                        .await;
                let Ok(Ok((mut stream, _))) = accept else {
                    break;
                };

                let request = read_http_request(&mut stream).await.unwrap();
                let request_line = request.lines().next().unwrap_or_default();
                let matched = response_bodies.iter().find_map(|(path, body)| {
                    request_line
                        .contains(path)
                        .then_some((*path, body.as_str()))
                });

                let (status, response_body) = if let Some((_, body)) = matched {
                    ("200 OK", body.to_string())
                } else {
                    (
                        "404 Not Found",
                        r#"{"code":404,"message":"not found"}"#.to_string(),
                    )
                };

                let response = format!(
                    "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    response_body.len(),
                    response_body
                );
                stream.write_all(response.as_bytes()).await.unwrap();
            }
        });

        MockHttpServer {
            base_url: format!("http://{addr}"),
            handle,
        }
    }

    async fn read_http_request(stream: &mut tokio::net::TcpStream) -> io::Result<String> {
        let mut buffer = Vec::new();
        let mut temp = [0_u8; 1024];
        let mut content_length = None;

        loop {
            let read = stream.read(&mut temp).await?;
            if read == 0 {
                break;
            }
            buffer.extend_from_slice(&temp[..read]);

            if let Some(header_end) = find_header_end(&buffer) {
                if content_length.is_none() {
                    let headers = String::from_utf8_lossy(&buffer[..header_end]);
                    content_length = headers.lines().find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        if name.eq_ignore_ascii_case("content-length") {
                            value.trim().parse::<usize>().ok()
                        } else {
                            None
                        }
                    });
                }

                let body_len = buffer.len() - header_end - 4;
                if body_len >= content_length.unwrap_or(0) {
                    break;
                }
            }
        }

        Ok(String::from_utf8_lossy(&buffer).into_owned())
    }

    fn find_header_end(buffer: &[u8]) -> Option<usize> {
        buffer.windows(4).position(|window| window == b"\r\n\r\n")
    }

    #[tokio::test]
    async fn test_index_result_default() {
        let result = IndexResult {
            files_processed: 0,
            chunks_created: 0,
            embeddings_generated: 0,
            vectors_inserted: 0,
            duration_ms: 0,
            warnings: vec![],
            lexical_only: false,
        };
        assert_eq!(result.files_processed, 0);
    }

    fn make_test_indexer_state(root: &Path, embedding_url: &str, dimension: usize) -> IndexerState {
        IndexerState::new(
            CodeWalker::new(),
            CodeSplitter::new(SplitterConfig {
                root_path: root.to_path_buf(),
                max_chunk_bytes: Config::default().chunk_size,
                overlap_lines: Config::default().chunk_overlap / 80,
                ..SplitterConfig::default()
            }),
            Embedder::Http(
                EmbeddingClient::new(EmbeddingConfig {
                    url: format!("{}/v1/embeddings", embedding_url),
                    model: "test".to_string(),
                    batch_size: 100,
                    api_key: None,
                    query_prefix: String::new(),
                    passage_prefix: String::new(),
                })
                .unwrap(),
            ),
            VectorStore::Local(crate::vectordb::LocalStore::new()),
            dimension,
        )
    }

    fn make_lexical_indexer_state(root: &Path) -> IndexerState {
        IndexerState::new(
            CodeWalker::new(),
            CodeSplitter::new(SplitterConfig {
                root_path: root.to_path_buf(),
                max_chunk_bytes: Config::default().chunk_size,
                overlap_lines: Config::default().chunk_overlap / 80,
                ..SplitterConfig::default()
            }),
            Embedder::Disabled,
            VectorStore::Local(crate::vectordb::LocalStore::new()),
            384,
        )
    }

    #[tokio::test]
    async fn test_incremental_update_requires_existing_manifest() {
        let temp_dir = TempDir::new().unwrap();
        let root = temp_dir.path();
        let cache_dir = TempDir::new().unwrap();
        let _cache_lock = set_test_cache_dir_async(cache_dir.path()).await;
        fs::write(root.join("main.py"), "def add(a, b):\n    return a + b\n").unwrap();

        let state = make_lexical_indexer_state(root);
        let err = update_codebase_index(&state, root).await.unwrap_err();

        assert!(err
            .to_string()
            .contains("Incremental update requires an existing index manifest"));
        assert_eq!(state.get_status().await.status, IndexState::Failed);
    }

    #[tokio::test]
    async fn test_incremental_update_requires_existing_lexical_index() {
        let temp_dir = TempDir::new().unwrap();
        let root = temp_dir.path();
        let cache_dir = TempDir::new().unwrap();
        let _cache_lock = set_test_cache_dir_async(cache_dir.path()).await;
        fs::write(root.join("main.py"), "def add(a, b):\n    return a + b\n").unwrap();

        let state = make_lexical_indexer_state(root);
        index_codebase(&state, root, false).await.unwrap();

        // Simulate a lost lexical cache (manifest still present in the repo).
        let other_cache = TempDir::new().unwrap();
        std::env::set_var("XDG_CACHE_HOME", other_cache.path());

        let err = update_codebase_index(&state, root).await.unwrap_err();
        assert!(err
            .to_string()
            .contains("Incremental update requires an existing lexical index"));
        assert_eq!(state.get_status().await.status, IndexState::Failed);
    }

    #[tokio::test]
    async fn test_incremental_update_processes_only_changed_files() {
        let temp_dir = TempDir::new().unwrap();
        let root = temp_dir.path();
        let cache_dir = TempDir::new().unwrap();
        let _cache_lock = set_test_cache_dir_async(cache_dir.path()).await;
        let src_dir = root.join("src");
        fs::create_dir_all(&src_dir).unwrap();
        fs::write(
            src_dir.join("lib.rs"),
            "pub fn alpha() -> i32 {\n    1\n}\n",
        )
        .unwrap();
        fs::write(
            src_dir.join("main.rs"),
            "fn main() {\n    println!(\"hi\");\n}\n",
        )
        .unwrap();

        let state = make_lexical_indexer_state(root);
        let first_result = index_codebase(&state, root, false).await.unwrap();
        assert_eq!(first_result.files_processed, 2);

        fs::write(
            src_dir.join("main.rs"),
            "fn main() {\n    println!(\"incremental update\");\n}\n",
        )
        .unwrap();

        let update_result = update_codebase_index(&state, root).await.unwrap();
        assert_eq!(update_result.files_processed, 1);
        assert!(update_result.chunks_created > 0);
        let status = state.get_status().await;
        assert_eq!(status.total_files, 2);
        assert_eq!(status.processed_files, 2);

        let manifest = state.manifest_store.load(root).unwrap().unwrap();
        assert_eq!(manifest.files.len(), 2);

        let noop_result = update_codebase_index(&state, root).await.unwrap();
        assert_eq!(noop_result.files_processed, 0);
        assert_eq!(noop_result.chunks_created, 0);
        assert!(noop_result
            .warnings
            .iter()
            .any(|warning| warning.contains("already up to date")));
    }

    #[tokio::test]
    async fn test_reindex_skips_unchanged_files() {
        let temp_dir = TempDir::new().unwrap();
        let root = temp_dir.path();
        let cache_dir = TempDir::new().unwrap();
        let _cache_lock = set_test_cache_dir_async(cache_dir.path()).await;
        let src_dir = root.join("src");
        fs::create_dir_all(&src_dir).unwrap();
        fs::write(
            src_dir.join("lib.rs"),
            "pub fn alpha() -> i32 {\n    1\n}\n",
        )
        .unwrap();
        fs::write(
            src_dir.join("main.rs"),
            "fn main() {\n    println!(\"hi\");\n}\n",
        )
        .unwrap();

        let embedding = spawn_mock_embedding_server(serde_json::json!({
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]}
            ]
        }))
        .await;

        let state = make_test_indexer_state(root, &embedding.base_url, 3);

        let first_result = index_codebase(&state, root, false).await.unwrap();
        assert!(first_result.files_processed >= 2);

        let manifest_path = root.join(".sindexer").join("index-manifest.json");
        assert!(manifest_path.exists());

        let second_result = index_codebase(&state, root, false).await.unwrap();
        assert_eq!(second_result.chunks_created, 0);
        assert!(
            second_result.warnings.is_empty()
                || second_result
                    .warnings
                    .iter()
                    .any(|warning| warning.contains("already up to date"))
        );
        let persisted_status = state.manifest_store.load_status(root).unwrap().unwrap();
        assert_eq!(persisted_status.total_files, 2);
        assert_eq!(persisted_status.processed_files, 2);
        assert_eq!(persisted_status.total_chunks, first_result.chunks_created);
        assert_eq!(
            persisted_status.embeddings_generated,
            first_result.embeddings_generated
        );
        assert!(manifest_path.exists());

        embedding.wait().await;
    }

    #[tokio::test]
    async fn test_force_reindex_ignores_manifest() {
        let temp_dir = TempDir::new().unwrap();
        let root = temp_dir.path();
        let cache_dir = TempDir::new().unwrap();
        let _cache_lock = set_test_cache_dir_async(cache_dir.path()).await;
        let src_dir = root.join("src");
        fs::create_dir_all(&src_dir).unwrap();
        fs::write(
            src_dir.join("lib.rs"),
            "pub fn alpha() -> i32 {\n    let value = 1;\n    value\n}\n",
        )
        .unwrap();
        fs::write(
            src_dir.join("main.rs"),
            "fn main() {\n    println!(\"force reindex\");\n}\n",
        )
        .unwrap();

        let embedding = spawn_dynamic_mock_embedding_server().await;

        let state = make_test_indexer_state(root, &embedding.base_url, 4);

        let first_result = index_codebase(&state, root, false).await.unwrap();
        let forced_result = index_codebase(&state, root, true).await.unwrap();

        assert_eq!(first_result.files_processed, 2);
        assert!(first_result.chunks_created > 0);
        assert_eq!(forced_result.files_processed, 2);
        assert!(forced_result.chunks_created > 0);

        embedding.wait().await;
    }

    #[tokio::test]
    async fn test_index_skips_unembeddable_chunks_after_batch_failure() {
        let temp_dir = TempDir::new().unwrap();
        let root = temp_dir.path();
        let cache_dir = TempDir::new().unwrap();
        let _cache_lock = set_test_cache_dir_async(cache_dir.path()).await;
        fs::write(root.join("good.py"), "def good():\n    return 1\n").unwrap();
        fs::write(
            root.join("bad.py"),
            "def bad():\n    return 'bad_embedding_payload'\n",
        )
        .unwrap();

        let embedding = spawn_selective_embedding_server().await;
        let state = make_test_indexer_state(root, &embedding.base_url, 4);

        let result = index_codebase(&state, root, true).await.unwrap();

        assert!(result.chunks_created >= 2);
        assert!(result.embeddings_generated > 0);
        assert!(result.embeddings_generated < result.chunks_created);
        assert_eq!(result.vectors_inserted, result.embeddings_generated);
        assert!(result
            .warnings
            .iter()
            .any(|warning| warning.contains("Skipped embedding for bad.py")));
        assert_eq!(state.get_status().await.status, IndexState::Completed);

        embedding.wait().await;
    }

    #[tokio::test]
    async fn test_index_surfaces_vector_backend_errors_instead_of_hanging() {
        let temp_dir = TempDir::new().unwrap();
        let root = temp_dir.path();
        let cache_dir = TempDir::new().unwrap();
        let _cache_lock = set_test_cache_dir_async(cache_dir.path()).await;
        fs::write(root.join("main.py"), "def add(a, b):\n    return a + b\n").unwrap();

        // First run against a healthy mock Milvus writes a compatible manifest.
        let healthy = spawn_mock_json_server(HashMap::from([
            (
                "/v2/vectordb/collections/has",
                serde_json::json!({"code": 0, "data": {"has": false}}),
            ),
            (
                "/v2/vectordb/collections/create",
                serde_json::json!({"code": 0}),
            ),
            (
                "/v2/vectordb/entities/upsert",
                serde_json::json!({"code": 0}),
            ),
            (
                "/v2/vectordb/entities/delete",
                serde_json::json!({"code": 0}),
            ),
        ]))
        .await;
        let embedding = spawn_dynamic_mock_embedding_server().await;
        let state = IndexerState::new(
            CodeWalker::new(),
            CodeSplitter::new(SplitterConfig {
                root_path: root.to_path_buf(),
                max_chunk_bytes: Config::default().chunk_size,
                overlap_lines: Config::default().chunk_overlap / 80,
                ..SplitterConfig::default()
            }),
            Embedder::Http(
                EmbeddingClient::new(EmbeddingConfig {
                    url: format!("{}/v1/embeddings", embedding.base_url),
                    model: "test".to_string(),
                    batch_size: 100,
                    api_key: None,
                    query_prefix: String::new(),
                    passage_prefix: String::new(),
                })
                .unwrap(),
            ),
            VectorStore::Milvus(crate::vectordb::MilvusClient::new(&healthy.base_url, None)),
            4,
        );
        index_codebase(&state, root, false).await.unwrap();
        fs::write(root.join("other.py"), "def sub(a, b):\n    return a - b\n").unwrap();

        // Second run: the existence check fails (mock knows no routes).
        // Drop the provenance record first: the broken mock lives on a
        // different port, which would otherwise trip the backend-mismatch
        // refusal before the existence check this test targets.
        ManifestStore.clear_backend(root).unwrap();
        let broken = spawn_mock_json_server(HashMap::new()).await;
        let state = IndexerState::new(
            CodeWalker::new(),
            CodeSplitter::new(SplitterConfig {
                root_path: root.to_path_buf(),
                max_chunk_bytes: Config::default().chunk_size,
                overlap_lines: Config::default().chunk_overlap / 80,
                ..SplitterConfig::default()
            }),
            Embedder::Http(
                EmbeddingClient::new(EmbeddingConfig {
                    url: format!("{}/v1/embeddings", embedding.base_url),
                    model: "test".to_string(),
                    batch_size: 100,
                    api_key: None,
                    query_prefix: String::new(),
                    passage_prefix: String::new(),
                })
                .unwrap(),
            ),
            VectorStore::Milvus(crate::vectordb::MilvusClient::new(&broken.base_url, None)),
            4,
        );

        let err = index_codebase(&state, root, false).await.unwrap_err();
        assert!(err
            .to_string()
            .contains("Failed to check vector collection existence"));
        assert_eq!(state.get_status().await.status, IndexState::Failed);

        embedding.wait().await;
        healthy.wait().await;
        broken.wait().await;
    }

    #[tokio::test]
    async fn test_lexical_only_indexing_refuses_live_semantic_collection() {
        let temp_dir = TempDir::new().unwrap();
        let root = temp_dir.path();
        let cache_dir = TempDir::new().unwrap();
        let _cache_lock = set_test_cache_dir_async(cache_dir.path()).await;
        fs::write(root.join("main.py"), "def add(a, b):\n    return a + b\n").unwrap();

        // Simulate a semantic collection previously built for this path.
        let collection = collection_name_from_path(root);
        let local = crate::vectordb::LocalStore::new();
        local.create_collection(&collection, 4).unwrap();

        let state = make_lexical_indexer_state(root);
        let err = index_codebase(&state, root, false).await.unwrap_err();
        assert!(err.to_string().contains("embeddings are disabled"));
        assert_eq!(state.get_status().await.status, IndexState::Failed);
    }

    #[tokio::test]
    async fn test_lexical_only_indexing_refuses_recorded_remote_vectors() {
        let temp_dir = TempDir::new().unwrap();
        let root = temp_dir.path();
        let cache_dir = TempDir::new().unwrap();
        let _cache_lock = set_test_cache_dir_async(cache_dir.path()).await;
        fs::write(root.join("main.py"), "def add(a, b):\n    return a + b\n").unwrap();

        // Simulate a semantic index built elsewhere (e.g. Milvus via the MCP
        // server): no local collection, but the shared status records vectors.
        ManifestStore
            .write_status(
                root,
                &IndexStatus {
                    total_files: 1,
                    processed_files: 1,
                    total_chunks: 3,
                    embeddings_generated: 3,
                    vectors_inserted: 3,
                    status: IndexState::Completed,
                },
            )
            .unwrap();

        let state = make_lexical_indexer_state(root);
        let err = index_codebase(&state, root, false).await.unwrap_err();
        assert!(err.to_string().contains("embeddings are disabled"));
        assert_eq!(state.get_status().await.status, IndexState::Failed);
    }

    #[tokio::test]
    async fn test_lexical_only_refusal_preserves_remote_vector_evidence() {
        let temp_dir = TempDir::new().unwrap();
        let root = temp_dir.path();
        let cache_dir = TempDir::new().unwrap();
        let _cache_lock = set_test_cache_dir_async(cache_dir.path()).await;
        fs::write(root.join("main.py"), "def add(a, b):\n    return a + b\n").unwrap();

        let recorded = IndexStatus {
            total_files: 1,
            processed_files: 1,
            total_chunks: 3,
            embeddings_generated: 3,
            vectors_inserted: 3,
            status: IndexState::Completed,
        };
        ManifestStore.write_status(root, &recorded).unwrap();

        // Refuse twice: the first refusal must not zero the persisted vector
        // count, or the second run would sail through and go stale.
        for _ in 0..2 {
            let state = make_lexical_indexer_state(root);
            let err = index_codebase(&state, root, false).await.unwrap_err();
            assert!(err.to_string().contains("embeddings are disabled"));
            assert_eq!(state.get_status().await.status, IndexState::Failed);
        }
        let persisted = ManifestStore.load_status(root).unwrap().unwrap();
        assert_eq!(persisted.vectors_inserted, 3);
        assert_eq!(persisted.status, IndexState::Completed);
    }

    #[tokio::test]
    async fn test_api_run_preserves_remote_vector_evidence() {
        let temp_dir = TempDir::new().unwrap();
        let root = temp_dir.path();
        let cache_dir = TempDir::new().unwrap();
        let _cache_lock = set_test_cache_dir_async(cache_dir.path()).await;
        fs::write(root.join("main.py"), "def add(a, b):\n    return a + b\n").unwrap();

        let recorded = IndexStatus {
            total_files: 1,
            processed_files: 1,
            total_chunks: 3,
            embeddings_generated: 3,
            vectors_inserted: 3,
            status: IndexState::Completed,
        };
        ManifestStore.write_status(root, &recorded).unwrap();

        // Full API path (not just the engine): Indexer::index must refuse the
        // lexical-only run and must not destroy the persisted vector count.
        let api = crate::api::Indexer::with_components(
            Config::default(),
            Embedder::Disabled,
            VectorStore::Local(crate::vectordb::LocalStore::new()),
        );
        for _ in 0..2 {
            let err = api.index(root, false).await.unwrap_err();
            assert!(format!("{err:#}").contains("embeddings are disabled"));
        }
        let persisted = ManifestStore.load_status(root).unwrap().unwrap();
        assert_eq!(persisted.vectors_inserted, 3);
        assert_eq!(persisted.status, IndexState::Completed);
    }

    #[tokio::test]
    async fn test_milvus_search_missing_collection_returns_empty() {
        let milvus = spawn_mock_json_server(HashMap::from([(
            "/v2/vectordb/entities/search",
            serde_json::json!({"code": 100, "message": "can't find collection"}),
        )]))
        .await;
        let store = VectorStore::Milvus(crate::vectordb::MilvusClient::new(&milvus.base_url, None));

        let hits = store.search("missing", &[0.1, 0.2], 5).await.unwrap();
        assert!(hits.is_empty());
        milvus.wait().await;
    }

    #[tokio::test]
    async fn test_clear_refuses_when_recorded_vectors_are_invisible() {
        let temp_dir = TempDir::new().unwrap();
        let root = temp_dir.path();
        let cache_dir = TempDir::new().unwrap();
        let _cache_lock = set_test_cache_dir_async(cache_dir.path()).await;

        ManifestStore
            .write_status(
                root,
                &IndexStatus {
                    total_files: 1,
                    processed_files: 1,
                    total_chunks: 3,
                    embeddings_generated: 3,
                    vectors_inserted: 3,
                    status: IndexState::Completed,
                },
            )
            .unwrap();

        // No collection visible to the configured (local) backend: refuse.
        let api = crate::api::Indexer::with_components(
            Config::default(),
            Embedder::Disabled,
            VectorStore::Local(crate::vectordb::LocalStore::new()),
        );
        let err = api.clear(root).await.unwrap_err();
        assert!(format!("{err:#}").contains("cannot"));
        assert_eq!(
            ManifestStore
                .load_status(root)
                .unwrap()
                .unwrap()
                .vectors_inserted,
            3
        );

        // A same-named but underfilled local collection is not sufficient
        // evidence: the recorded vectors may live in a backend we cannot see.
        let collection = collection_name_from_path(root);
        let local = crate::vectordb::LocalStore::new();
        local.create_collection(&collection, 4).unwrap();
        assert!(api.clear(root).await.is_err());

        // Once the visible collection holds the recorded count, clear works.
        local
            .insert_rows(
                &collection,
                &["a".into(), "b".into(), "c".into()],
                &["x".into(), "y".into(), "z".into()],
                &[vec![0.1; 4], vec![0.2; 4], vec![0.3; 4]],
                &[
                    crate::vectordb::ChunkMeta::default(),
                    crate::vectordb::ChunkMeta::default(),
                    crate::vectordb::ChunkMeta::default(),
                ],
            )
            .unwrap();
        api.clear(root).await.unwrap();
        assert!(ManifestStore.load_status(root).unwrap().is_none());
        // Fresh store instance: the old one still caches the collection in
        // memory; the on-disk file is what clear must have removed.
        assert!(!crate::vectordb::LocalStore::new()
            .has_collection(&collection)
            .unwrap());
    }

    #[tokio::test]
    async fn test_failed_incremental_update_preserves_vector_evidence() {
        let temp_dir = TempDir::new().unwrap();
        let root = temp_dir.path();
        let cache_dir = TempDir::new().unwrap();
        let _cache_lock = set_test_cache_dir_async(cache_dir.path()).await;
        fs::write(root.join("main.py"), "def add(a, b):\n    return a + b\n").unwrap();

        // First run against healthy mocks records real vector counts.
        let milvus = spawn_mock_json_server(HashMap::from([
            (
                "/v2/vectordb/collections/has",
                serde_json::json!({"code": 0, "data": {"has": true}}),
            ),
            (
                "/v2/vectordb/collections/create",
                serde_json::json!({"code": 0}),
            ),
            (
                "/v2/vectordb/entities/upsert",
                serde_json::json!({"code": 0}),
            ),
            (
                "/v2/vectordb/entities/delete",
                serde_json::json!({"code": 0}),
            ),
            (
                "/v2/vectordb/collections/drop",
                serde_json::json!({"code": 0}),
            ),
        ]))
        .await;
        let embedding = spawn_dynamic_mock_embedding_server().await;
        let make_state = |embedding_url: &str| {
            IndexerState::new(
                CodeWalker::new(),
                CodeSplitter::new(SplitterConfig {
                    root_path: root.to_path_buf(),
                    max_chunk_bytes: Config::default().chunk_size,
                    overlap_lines: Config::default().chunk_overlap / 80,
                    ..SplitterConfig::default()
                }),
                Embedder::Http(
                    EmbeddingClient::new(EmbeddingConfig {
                        url: format!("{}/v1/embeddings", embedding_url),
                        model: "test".to_string(),
                        batch_size: 100,
                        api_key: None,
                        query_prefix: String::new(),
                        passage_prefix: String::new(),
                    })
                    .unwrap(),
                ),
                VectorStore::Milvus(crate::vectordb::MilvusClient::new(&milvus.base_url, None)),
                4,
            )
        };
        let first = index_codebase(&make_state(&embedding.base_url), root, false)
            .await
            .unwrap();
        assert!(first.vectors_inserted > 0);

        // Change a file, then fail the incremental run at the embedding stage.
        fs::write(root.join("main.py"), "def add(a, b):\n    return b + a\n").unwrap();
        let dead_embedding = spawn_mock_json_server(HashMap::new()).await;
        let state = make_state(&dead_embedding.base_url);
        index_codebase(&state, root, false).await.unwrap_err();

        // The failure must not zero the durable vector evidence.
        let persisted = ManifestStore.load_status(root).unwrap().unwrap();
        assert_eq!(persisted.status, IndexState::Failed);
        assert!(persisted.vectors_inserted >= first.vectors_inserted);

        embedding.wait().await;
        dead_embedding.wait().await;
        milvus.wait().await;
    }

    #[tokio::test]
    async fn test_backend_switch_requires_clear() {
        let temp_dir = TempDir::new().unwrap();
        let root = temp_dir.path();
        let cache_dir = TempDir::new().unwrap();
        let _cache_lock = set_test_cache_dir_async(cache_dir.path()).await;
        fs::write(root.join("main.py"), "def add(a, b):\n    return a + b\n").unwrap();

        let embedding = spawn_dynamic_mock_embedding_server().await;
        let make_local_state = || {
            IndexerState::new(
                CodeWalker::new(),
                CodeSplitter::new(SplitterConfig {
                    root_path: root.to_path_buf(),
                    max_chunk_bytes: Config::default().chunk_size,
                    overlap_lines: Config::default().chunk_overlap / 80,
                    ..SplitterConfig::default()
                }),
                Embedder::Http(
                    EmbeddingClient::new(EmbeddingConfig {
                        url: format!("{}/v1/embeddings", embedding.base_url),
                        model: "test".to_string(),
                        batch_size: 100,
                        api_key: None,
                        query_prefix: String::new(),
                        passage_prefix: String::new(),
                    })
                    .unwrap(),
                ),
                VectorStore::Local(crate::vectordb::LocalStore::new()),
                4,
            )
        };

        // Build against the local backend; provenance is recorded.
        let first = index_codebase(&make_local_state(), root, false)
            .await
            .unwrap();
        assert!(first.vectors_inserted > 0);
        assert_eq!(
            ManifestStore.load_backend(root).unwrap().as_deref(),
            Some("local")
        );

        // Simulate the index having been produced by a different backend.
        ManifestStore
            .write_backend(root, "milvus http://elsewhere:19530")
            .unwrap();

        // Incremental update refuses the mismatch outright.
        let err = update_codebase_index(&make_local_state(), root)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("vector backend"));

        // A plain index run refuses too: rebuilding into the wrong backend
        // would advance the shared manifest while the recorded backend's
        // vectors go stale.
        let err = index_codebase(&make_local_state(), root, false)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("vector backend"));

        // Even --force refuses: re-homing without retiring the recorded
        // backend's collection would let it authenticate the new manifest.
        let err = index_codebase(&make_local_state(), root, true)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("vector backend"));

        // After clear retires the record (the sanctioned re-homing path),
        // indexing into the new backend proceeds and rewrites provenance.
        ManifestStore.clear_backend(root).unwrap();
        let rebuilt = index_codebase(&make_local_state(), root, true)
            .await
            .unwrap();
        assert!(rebuilt.chunks_created > 0);
        assert_eq!(
            ManifestStore.load_backend(root).unwrap().as_deref(),
            Some("local")
        );

        embedding.wait().await;
    }

    #[tokio::test]
    async fn test_sibling_manifest_rewrite_invalidates_backend_record() {
        let temp_dir = TempDir::new().unwrap();
        let root = temp_dir.path();
        let cache_dir = TempDir::new().unwrap();
        let _cache_lock = set_test_cache_dir_async(cache_dir.path()).await;
        fs::write(root.join("main.py"), "def add(a, b):\n    return a + b\n").unwrap();

        ManifestStore
            .write_for_files(
                root,
                "collection",
                &IndexInputs {
                    chunk_size: 512,
                    overlap_lines: 3,
                    min_chunk_lines: 5,
                    target_chunk_lines: 50,
                    extensions: vec!["py".into()],
                    ignore_patterns: vec![],
                    max_file_size: 1024 * 1024,
                    follow_symlinks: false,
                    embedding_passage_prefix_sha256: String::new(),
                },
                &[root.join("main.py")],
            )
            .unwrap();
        ManifestStore.write_backend(root, "local").unwrap();
        assert_eq!(
            ManifestStore.load_backend(root).unwrap().as_deref(),
            Some("local")
        );

        // Simulate rust_sindexer rewriting the shared manifest (it does not
        // know about the sidecar): the record must stop authenticating.
        let manifest_path = root.join(".sindexer").join("index-manifest.json");
        let mut manifest_json: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).unwrap()).unwrap();
        manifest_json["files"][0]["sha256"] = serde_json::json!("rewritten-by-sibling");
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_json).unwrap(),
        )
        .unwrap();

        assert_eq!(ManifestStore.load_backend(root).unwrap(), None);
    }

    #[tokio::test]
    async fn test_search_skips_semantic_on_backend_mismatch() {
        let temp_dir = TempDir::new().unwrap();
        let root = temp_dir.path();
        let cache_dir = TempDir::new().unwrap();
        let _cache_lock = set_test_cache_dir_async(cache_dir.path()).await;
        fs::write(root.join("main.py"), "def add(a, b):\n    return a + b\n").unwrap();

        ManifestStore
            .write_for_files(
                root,
                "collection",
                &IndexInputs {
                    chunk_size: 512,
                    overlap_lines: 3,
                    min_chunk_lines: 5,
                    target_chunk_lines: 50,
                    extensions: vec!["py".into()],
                    ignore_patterns: vec![],
                    max_file_size: 1024 * 1024,
                    follow_symlinks: false,
                    embedding_passage_prefix_sha256: String::new(),
                },
                &[root.join("main.py")],
            )
            .unwrap();
        ManifestStore
            .write_backend(root, "milvus http://elsewhere:19530")
            .unwrap();

        // Embedder points at a dead endpoint: if the semantic path ran, the
        // search would fail. The provenance mismatch must skip it instead.
        let api = crate::api::Indexer::with_components(
            Config::default(),
            Embedder::Http(
                EmbeddingClient::new(EmbeddingConfig {
                    url: "http://127.0.0.1:9/v1/embeddings".to_string(),
                    model: "test".to_string(),
                    batch_size: 100,
                    api_key: None,
                    query_prefix: String::new(),
                    passage_prefix: String::new(),
                })
                .unwrap(),
            ),
            VectorStore::Local(crate::vectordb::LocalStore::new()),
        );
        let hits = api.search(root, "add", 5, &[]).await.unwrap();
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn test_clear_refuses_backend_mismatch_even_with_zero_vectors() {
        let temp_dir = TempDir::new().unwrap();
        let root = temp_dir.path();
        let cache_dir = TempDir::new().unwrap();
        let _cache_lock = set_test_cache_dir_async(cache_dir.path()).await;
        fs::write(root.join("main.py"), "def add(a, b):\n    return a + b\n").unwrap();

        // A semantic index in another backend that happens to hold zero
        // vectors (e.g. built from an empty repository): the provenance
        // record is the only evidence, and clear must respect it.
        ManifestStore
            .write_for_files(
                root,
                "collection",
                &IndexInputs {
                    chunk_size: 512,
                    overlap_lines: 3,
                    min_chunk_lines: 5,
                    target_chunk_lines: 50,
                    extensions: vec!["py".into()],
                    ignore_patterns: vec![],
                    max_file_size: 1024 * 1024,
                    follow_symlinks: false,
                    embedding_passage_prefix_sha256: String::new(),
                },
                &[root.join("main.py")],
            )
            .unwrap();
        ManifestStore
            .write_backend(root, "milvus http://elsewhere:19530")
            .unwrap();

        let api = crate::api::Indexer::with_components(
            Config::default(),
            Embedder::Disabled,
            VectorStore::Local(crate::vectordb::LocalStore::new()),
        );
        let err = api.clear(root).await.unwrap_err();
        assert!(format!("{err:#}").contains("vector backend"));
        assert!(ManifestStore.load(root).unwrap().is_some());

        // With the record gone (backend reconfigured or manually removed),
        // clear proceeds.
        ManifestStore.clear_backend(root).unwrap();
        api.clear(root).await.unwrap();
        assert!(ManifestStore.load(root).unwrap().is_none());
    }

    #[tokio::test]
    async fn test_lexical_only_indexing_refuses_provenance_record() {
        let temp_dir = TempDir::new().unwrap();
        let root = temp_dir.path();
        let cache_dir = TempDir::new().unwrap();
        let _cache_lock = set_test_cache_dir_async(cache_dir.path()).await;
        fs::write(root.join("main.py"), "def add(a, b):\n    return a + b\n").unwrap();

        // Zero recorded vectors and no visible collection, but an
        // authenticated provenance record: still semantic state.
        ManifestStore
            .write_for_files(
                root,
                "collection",
                &IndexInputs {
                    chunk_size: 512,
                    overlap_lines: 3,
                    min_chunk_lines: 5,
                    target_chunk_lines: 50,
                    extensions: vec!["py".into()],
                    ignore_patterns: vec![],
                    max_file_size: 1024 * 1024,
                    follow_symlinks: false,
                    embedding_passage_prefix_sha256: String::new(),
                },
                &[root.join("main.py")],
            )
            .unwrap();
        ManifestStore
            .write_backend(root, "milvus http://elsewhere:19530")
            .unwrap();

        let state = make_lexical_indexer_state(root);
        let err = index_codebase(&state, root, false).await.unwrap_err();
        assert!(err.to_string().contains("embeddings are disabled"));
        assert_eq!(state.get_status().await.status, IndexState::Failed);
    }

    #[tokio::test]
    async fn test_index_fails_when_provenance_cannot_be_recorded() {
        let temp_dir = TempDir::new().unwrap();
        let root = temp_dir.path();
        let cache_dir = TempDir::new().unwrap();
        let _cache_lock = set_test_cache_dir_async(cache_dir.path()).await;
        fs::write(root.join("main.py"), "def add(a, b):\n    return a + b\n").unwrap();

        // A directory squatting on the record path makes the write fail.
        fs::create_dir_all(root.join(".sindexer").join("vector-backend.json")).unwrap();

        let embedding = spawn_dynamic_mock_embedding_server().await;
        let state = IndexerState::new(
            CodeWalker::new(),
            CodeSplitter::new(SplitterConfig {
                root_path: root.to_path_buf(),
                max_chunk_bytes: Config::default().chunk_size,
                overlap_lines: Config::default().chunk_overlap / 80,
                ..SplitterConfig::default()
            }),
            Embedder::Http(
                EmbeddingClient::new(EmbeddingConfig {
                    url: format!("{}/v1/embeddings", embedding.base_url),
                    model: "test".to_string(),
                    batch_size: 100,
                    api_key: None,
                    query_prefix: String::new(),
                    passage_prefix: String::new(),
                })
                .unwrap(),
            ),
            VectorStore::Local(crate::vectordb::LocalStore::new()),
            4,
        );

        let err = index_codebase(&state, root, false).await.unwrap_err();
        assert!(format!("{err:#}").contains("provenance"));
        assert_eq!(state.get_status().await.status, IndexState::Failed);
        embedding.wait().await;
    }

    #[tokio::test]
    async fn test_lexical_only_indexing() {
        let temp_dir = TempDir::new().unwrap();
        let root = temp_dir.path();
        let cache_dir = TempDir::new().unwrap();
        let _cache_lock = set_test_cache_dir_async(cache_dir.path()).await;
        fs::write(root.join("main.py"), "def add(a, b):\n    return a + b\n").unwrap();

        let state = IndexerState::new(
            CodeWalker::new(),
            CodeSplitter::new(SplitterConfig {
                root_path: root.to_path_buf(),
                max_chunk_bytes: Config::default().chunk_size,
                overlap_lines: Config::default().chunk_overlap / 80,
                ..SplitterConfig::default()
            }),
            Embedder::Disabled,
            VectorStore::Local(crate::vectordb::LocalStore::new()),
            384,
        );

        let result = index_codebase(&state, root, true).await.unwrap();
        assert!(result.lexical_only);
        assert_eq!(result.embeddings_generated, 0);
        assert_eq!(result.vectors_inserted, 0);
        assert!(result.chunks_created > 0);
    }
}
