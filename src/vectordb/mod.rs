pub mod local;

pub use local::{ChunkMeta, CollectionStats, InsertRow, LocalStore, SearchHit};

use anyhow::Result;
use sha2::{Digest, Sha256};
use std::path::Path;
use tracing::{debug, info};

/// The local brute-force vector store behind an async-friendly facade.
pub struct VectorStore {
    store: LocalStore,
}

impl Default for VectorStore {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorStore {
    pub fn new() -> Self {
        Self {
            store: LocalStore::new(),
        }
    }

    pub async fn create_collection(&self, name: &str, dimension: usize) -> Result<()> {
        info!(collection = name, dimension, "creating collection");
        self.store.create_collection(name, dimension)
    }

    pub async fn has_collection(&self, name: &str) -> Result<bool> {
        self.store.has_collection(name)
    }

    pub async fn drop_collection(&self, name: &str) -> Result<()> {
        info!(collection = name, "dropping collection");
        self.store.drop_collection(name)
    }

    pub async fn insert_batch(&self, collection: &str, data: &[InsertRow]) -> Result<usize> {
        debug!(collection, batch_size = data.len(), "inserting batch");
        let docs: Vec<_> = data
            .iter()
            .map(|row| local::LocalDoc {
                id: row.id.clone(),
                content: row.content.clone(),
                vector: row.vector.clone(),
                metadata: row.metadata.clone(),
            })
            .collect();
        let inserted = docs.len();
        self.store.insert_docs(collection, docs)?;
        Ok(inserted)
    }

    pub async fn search(
        &self,
        collection: &str,
        vector: &[f32],
        top_k: usize,
    ) -> Result<Vec<SearchHit>> {
        debug!(collection, top_k, "searching vector store");
        self.store.search(collection, vector, top_k)
    }

    pub async fn list_collections(&self) -> Result<Vec<String>> {
        Ok(self.store.list_collections())
    }

    pub async fn collection_stats(&self, name: &str) -> Result<CollectionStats> {
        Ok(CollectionStats {
            row_count: self.store.collection_size(name) as u64,
        })
    }

    pub async fn delete_by_relative_paths(
        &self,
        collection: &str,
        relative_paths: &[String],
    ) -> Result<()> {
        if relative_paths.is_empty() {
            return Ok(());
        }
        debug!(
            collection,
            path_count = relative_paths.len(),
            "deleting by relative paths"
        );
        self.store.delete_by_filter(collection, relative_paths)
    }
}

/// Generate a sanitized, hashed collection name from a codebase root path.
///
/// The last path component becomes a human-readable prefix; a truncated
/// SHA-256 of the full path guarantees uniqueness across checkouts.
pub fn collection_name_from_path(path: &Path) -> String {
    let mut hasher = Sha256::new();
    hasher.update(path.to_string_lossy().as_bytes());
    let hash = hex::encode(hasher.finalize());
    let hash_prefix = &hash[..16];

    let prefix = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("collection");

    let sanitized: String = prefix
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();

    let sanitized = if sanitized
        .chars()
        .next()
        .map(|c| c.is_ascii_digit())
        .unwrap_or(true)
    {
        format!("_{}", sanitized)
    } else {
        sanitized
    };

    format!("{}_{}", sanitized, hash_prefix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_collection_name_basic() {
        let path = PathBuf::from("/home/user/my-project");
        let name = collection_name_from_path(&path);
        assert!(name.starts_with("my_project_"));
        assert!(name.len() <= 255);
    }

    #[test]
    fn test_collection_name_numeric_start() {
        let path = PathBuf::from("/home/user/123project");
        let name = collection_name_from_path(&path);
        assert!(name.starts_with('_'));
    }

    #[test]
    fn test_collection_name_special_chars() {
        let path = PathBuf::from("/home/user/my.project-name@v2");
        let name = collection_name_from_path(&path);
        assert!(!name.contains('.'));
        assert!(!name.contains('-'));
        assert!(!name.contains('@'));
    }

    #[test]
    fn test_collection_name_uniqueness() {
        let path1 = PathBuf::from("/home/user/project");
        let path2 = PathBuf::from("/home/other/project");
        assert_ne!(
            collection_name_from_path(&path1),
            collection_name_from_path(&path2)
        );
    }
}
