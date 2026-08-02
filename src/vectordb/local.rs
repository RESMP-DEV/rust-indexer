use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

/// Chunk metadata stored alongside each embedding vector.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ChunkMeta {
    pub file_path: PathBuf,
    pub relative_path: String,
    pub start_line: u32,
    pub end_line: u32,
    pub language: String,
}

/// A stored document: content plus its embedding vector and chunk metadata.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LocalDoc {
    pub id: String,
    pub content: String,
    pub vector: Vec<f32>,
    pub metadata: ChunkMeta,
}

/// A row to insert into the vector store.
#[derive(Clone, Debug)]
pub struct InsertRow {
    pub id: String,
    pub content: String,
    pub vector: Vec<f32>,
    pub metadata: ChunkMeta,
}

/// A search result from the vector store.
#[derive(Clone, Debug)]
pub struct SearchHit {
    pub id: String,
    pub score: f32,
    pub content: String,
    pub metadata: ChunkMeta,
}

#[derive(Clone, Debug, Default)]
pub struct CollectionStats {
    pub row_count: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Collection {
    dimension: usize,
    docs: Vec<LocalDoc>,
}

/// Brute-force in-memory cosine-similarity store with bincode disk persistence.
pub struct LocalStore {
    collections: parking_lot::Mutex<HashMap<String, Collection>>,
}

impl Default for LocalStore {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalStore {
    pub fn new() -> Self {
        Self {
            collections: parking_lot::Mutex::new(HashMap::new()),
        }
    }

    fn storage_dir() -> PathBuf {
        dirs_path().join("vector-indexes")
    }

    fn collection_path(name: &str) -> PathBuf {
        Self::storage_dir().join(format!("{name}.bin"))
    }

    fn load_collection(&self, name: &str) -> Option<Collection> {
        let path = Self::collection_path(name);
        let data = std::fs::read(&path).ok()?;
        bincode::deserialize(&data).ok()
    }

    fn persist_collection(&self, name: &str, collection: &Collection) -> Result<()> {
        let dir = Self::storage_dir();
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("create vector index dir: {}", dir.display()))?;
        let path = Self::collection_path(name);
        let data = bincode::serialize(collection)?;
        std::fs::write(&path, data)?;
        Ok(())
    }

    pub fn create_collection(&self, name: &str, dimension: usize) -> Result<()> {
        debug!(collection = name, dimension, "Creating local collection");
        let mut collections = self.collections.lock();
        let collection = Collection {
            dimension,
            docs: Vec::new(),
        };
        collections.insert(name.to_string(), collection.clone());
        self.persist_collection(name, &collection)?;
        info!(collection = name, dimension, "Local collection created");
        Ok(())
    }

    pub fn has_collection(&self, name: &str) -> Result<bool> {
        let collections = self.collections.lock();
        if collections.contains_key(name) {
            return Ok(true);
        }
        Ok(Self::collection_path(name).exists())
    }

    pub fn drop_collection(&self, name: &str) -> Result<()> {
        info!(collection = name, "Dropping local collection");
        self.collections.lock().remove(name);
        let path = Self::collection_path(name);
        if path.exists() {
            std::fs::remove_file(&path)?;
        }
        Ok(())
    }

    pub fn insert_docs(&self, collection_name: &str, docs: Vec<LocalDoc>) -> Result<()> {
        if docs.is_empty() {
            return Ok(());
        }
        let count = docs.len();
        debug!(
            collection = collection_name,
            docs = count,
            "Inserting docs into local store"
        );
        let mut collections = self.collections.lock();
        let collection = collections
            .entry(collection_name.to_string())
            .or_insert_with(|| self.load_collection(collection_name).unwrap_or_default());
        collection.docs.extend(docs);
        self.persist_collection(collection_name, collection)?;
        debug!(
            collection = collection_name,
            total_docs = collection.docs.len(),
            "Insert completed"
        );
        Ok(())
    }

    pub fn insert_rows(
        &self,
        collection_name: &str,
        ids: &[String],
        contents: &[String],
        vectors: &[Vec<f32>],
        metadatas: &[ChunkMeta],
    ) -> Result<()> {
        let docs: Vec<LocalDoc> = ids
            .iter()
            .zip(contents.iter())
            .zip(vectors.iter())
            .zip(metadatas.iter())
            .map(|(((id, content), vector), metadata)| LocalDoc {
                id: id.clone(),
                content: content.clone(),
                vector: vector.clone(),
                metadata: metadata.clone(),
            })
            .collect();
        self.insert_docs(collection_name, docs)
    }

    pub fn search(
        &self,
        collection_name: &str,
        query_vector: &[f32],
        top_k: usize,
    ) -> Result<Vec<SearchHit>> {
        let start = std::time::Instant::now();
        let mut collections = self.collections.lock();
        let collection = match collections.get(collection_name) {
            Some(c) => c,
            None => {
                if let Some(loaded) = self.load_collection(collection_name) {
                    debug!(
                        collection = collection_name,
                        docs = loaded.docs.len(),
                        "Loaded collection from disk"
                    );
                    collections.insert(collection_name.to_string(), loaded);
                    collections.get(collection_name).unwrap()
                } else {
                    warn!(
                        collection = collection_name,
                        "Collection not found for search"
                    );
                    return Ok(Vec::new());
                }
            }
        };

        debug!(
            collection = collection_name,
            candidates = collection.docs.len(),
            top_k,
            "Brute-force cosine search"
        );
        let mut scored: Vec<(f32, &LocalDoc)> = collection
            .docs
            .iter()
            .map(|doc| (cosine_similarity(query_vector, &doc.vector), doc))
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        let results: Vec<SearchHit> = scored
            .into_iter()
            .map(|(score, doc)| SearchHit {
                id: doc.id.clone(),
                score,
                content: doc.content.clone(),
                metadata: doc.metadata.clone(),
            })
            .collect();
        debug!(
            collection = collection_name,
            results = results.len(),
            elapsed_us = start.elapsed().as_micros() as u64,
            "Local search completed"
        );
        Ok(results)
    }

    pub fn list_collections(&self) -> Vec<String> {
        let mut names: Vec<String> = self.collections.lock().keys().cloned().collect();
        if let Ok(entries) = std::fs::read_dir(Self::storage_dir()) {
            for entry in entries.flatten() {
                if let Some(name) = entry.path().file_stem().and_then(|s| s.to_str()) {
                    let name = name.to_string();
                    if !names.contains(&name) {
                        names.push(name);
                    }
                }
            }
        }
        names
    }

    pub fn collection_size(&self, name: &str) -> usize {
        let collections = self.collections.lock();
        if let Some(c) = collections.get(name) {
            return c.docs.len();
        }
        self.load_collection(name)
            .map(|c| c.docs.len())
            .unwrap_or(0)
    }

    pub fn delete_by_filter(&self, collection_name: &str, relative_paths: &[String]) -> Result<()> {
        debug!(
            collection = collection_name,
            paths = relative_paths.len(),
            "Deleting docs by relative path filter"
        );
        let mut collections = self.collections.lock();
        let collection = match collections.get_mut(collection_name) {
            Some(c) => c,
            None => match self.load_collection(collection_name) {
                Some(loaded) => collections
                    .entry(collection_name.to_string())
                    .or_insert(loaded),
                None => return Ok(()),
            },
        };

        let before = collection.docs.len();
        collection.docs.retain(|doc| {
            !relative_paths
                .iter()
                .any(|p| p == &doc.metadata.relative_path)
        });

        let deleted = before - collection.docs.len();
        debug!(
            collection = collection_name,
            deleted,
            remaining = collection.docs.len(),
            "Delete filter applied"
        );
        self.persist_collection(collection_name, collection)?;
        Ok(())
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

fn dirs_path() -> PathBuf {
    if let Ok(dir) = std::env::var("XDG_CACHE_HOME") {
        PathBuf::from(dir).join("rust-indexer")
    } else if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home).join(".cache").join("rust-indexer")
    } else {
        PathBuf::from("/tmp/rust-indexer")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_local_store_crud() {
        let cache_dir = tempfile::TempDir::new().unwrap();
        let _cache_lock = crate::lexical::test_support::set_test_cache_dir(cache_dir.path());
        let store = LocalStore::new();
        let name = format!("test_{}", uuid::Uuid::new_v4());

        assert!(!store.has_collection(&name).unwrap());
        store.create_collection(&name, 3).unwrap();
        assert!(store.has_collection(&name).unwrap());

        store
            .insert_rows(
                &name,
                &["id1".to_string()],
                &["hello world".to_string()],
                &[vec![1.0, 0.0, 0.0]],
                &[ChunkMeta {
                    relative_path: "src/lib.rs".to_string(),
                    ..ChunkMeta::default()
                }],
            )
            .unwrap();

        let results = store.search(&name, &[1.0, 0.0, 0.0], 5).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "hello world");

        store.drop_collection(&name).unwrap();
        assert!(!store.has_collection(&name).unwrap());
    }

    #[test]
    fn test_persistence_round_trips_binary_format() {
        let cache_dir = tempfile::TempDir::new().unwrap();
        let _cache_lock = crate::lexical::test_support::set_test_cache_dir(cache_dir.path());
        let store = LocalStore::new();
        let name = format!("test_{}", uuid::Uuid::new_v4());
        store.create_collection(&name, 2).unwrap();
        store
            .insert_rows(
                &name,
                &["a".to_string()],
                &["fn alpha() {}".to_string()],
                &[vec![0.5, 0.5]],
                &[ChunkMeta {
                    relative_path: "src/a.rs".to_string(),
                    ..ChunkMeta::default()
                }],
            )
            .unwrap();

        // A fresh store instance must read the persisted collection from disk.
        let fresh = LocalStore::new();
        assert_eq!(fresh.collection_size(&name), 1);
        let results = fresh.search(&name, &[0.5, 0.5], 5).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "fn alpha() {}");

        fresh
            .delete_by_filter(&name, &["src/a.rs".to_string()])
            .unwrap();
        assert_eq!(LocalStore::new().collection_size(&name), 0);

        fresh.drop_collection(&name).unwrap();
    }
}
