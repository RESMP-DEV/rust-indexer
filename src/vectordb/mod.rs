pub mod client;
pub mod local;

pub use client::MilvusClient;
pub use local::{ChunkMeta, CollectionStats, InsertRow, LocalStore, SearchHit};

pub(crate) use client::milvus_id_for_chunk_id;

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::env;
use std::path::Path;
use tracing::{debug, info, warn};

use crate::config::Config;

const COLLECTION_IDENTITY_ENV: &str = "SINDEXER_COLLECTION_IDENTITY";
const COLLECTION_ROOT_ENV: &str = "SINDEXER_COLLECTION_ROOT";

/// Collection scoping resolved once per process. Reading the environment on
/// every call would let a runtime mutation silently retarget collections
/// mid-process; freezing at first use keeps naming deterministic for the
/// process lifetime, which is the semantics a short-lived CLI wants.
static COLLECTION_SCOPE: once_cell::sync::Lazy<(Option<String>, Option<String>)> =
    once_cell::sync::Lazy::new(|| {
        (
            env::var(COLLECTION_IDENTITY_ENV).ok(),
            env::var(COLLECTION_ROOT_ENV).ok(),
        )
    });

/// Selects between the local brute-force vector store and a remote
/// Milvus/Zilliz instance. Collection naming and stored metadata match
/// rust_sindexer, so both tools can operate on the same index.
pub enum VectorStore {
    Local(LocalStore),
    Milvus(MilvusClient),
}

impl VectorStore {
    pub fn from_config(config: &Config) -> Self {
        if config.has_milvus_url() {
            info!(milvus_url = %config.milvus_url, "using Milvus vector store");
            Self::Milvus(MilvusClient::new(
                &config.milvus_url,
                config.milvus_token.clone(),
            ))
        } else {
            info!("using local vector store");
            Self::Local(LocalStore::new())
        }
    }

    /// Stable identity of the backend holding the vectors, recorded next to
    /// the shared manifest so a backend switch forces revalidation.
    pub fn provenance(&self) -> String {
        match self {
            Self::Local(_) => "local".to_string(),
            // base_url is normalized at client construction.
            Self::Milvus(client) => format!("milvus {}", client.base_url()),
        }
    }

    pub async fn create_collection(&self, name: &str, dimension: usize) -> Result<()> {
        info!(collection = name, dimension, "creating collection");
        match self {
            Self::Local(store) => store.create_collection(name, dimension),
            Self::Milvus(client) => client.create_collection(name, dimension).await,
        }
    }

    pub async fn has_collection(&self, name: &str) -> Result<bool> {
        match self {
            Self::Local(store) => store.has_collection(name),
            Self::Milvus(client) => client.has_collection(name).await,
        }
    }

    pub async fn drop_collection(&self, name: &str) -> Result<()> {
        info!(collection = name, "dropping collection");
        match self {
            Self::Local(store) => store.drop_collection(name),
            Self::Milvus(client) => client.drop_collection(name).await,
        }
    }

    pub async fn insert_batch(&self, collection: &str, data: &[InsertRow]) -> Result<usize> {
        debug!(collection, batch_size = data.len(), "inserting batch");
        match self {
            Self::Local(store) => {
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
                store.insert_docs(collection, docs)?;
                Ok(inserted)
            }
            Self::Milvus(client) => {
                let rows: Vec<client::InsertRow> = data
                    .iter()
                    .map(|row| {
                        Ok(client::InsertRow {
                            id: milvus_id_for_chunk_id(&row.id),
                            content: row.content.clone(),
                            vector: row.vector.clone(),
                            metadata: serde_json::to_value(&row.metadata).with_context(|| {
                                format!("chunk metadata for {} must serialize", row.id)
                            })?,
                        })
                    })
                    .collect::<Result<_>>()?;
                client.insert_batch(collection, &rows).await
            }
        }
    }

    pub async fn search(
        &self,
        collection: &str,
        vector: &[f32],
        top_k: usize,
    ) -> Result<Vec<SearchHit>> {
        debug!(collection, top_k, "searching vector store");
        match self {
            Self::Local(store) => store.search(collection, vector, top_k),
            Self::Milvus(client) => {
                // The client maps a missing collection to an empty result
                // set (a repo may only have a lexical index), so no
                // pre-flight existence check is needed here.
                let hits = client.search(collection, vector, top_k).await?;
                Ok(hits
                    .into_iter()
                    .map(|hit| {
                        let metadata = serde_json::from_value(hit.metadata).unwrap_or_else(|e| {
                            warn!(
                                collection,
                                hit_id = %hit.id,
                                error = %e,
                                "Milvus metadata does not match the shared ChunkMeta layout; \
                                 result paths will be empty"
                            );
                            ChunkMeta::default()
                        });
                        SearchHit {
                            id: hit.id,
                            score: hit.score,
                            content: hit.content,
                            metadata,
                        }
                    })
                    .collect())
            }
        }
    }

    pub async fn list_collections(&self) -> Result<Vec<String>> {
        match self {
            Self::Local(store) => Ok(store.list_collections()),
            Self::Milvus(client) => client.list_collections().await,
        }
    }

    pub async fn collection_stats(&self, name: &str) -> Result<CollectionStats> {
        match self {
            Self::Local(store) => Ok(CollectionStats {
                row_count: store.collection_size(name) as u64,
            }),
            Self::Milvus(client) => {
                let stats = client.collection_stats(name).await?;
                Ok(CollectionStats {
                    row_count: stats.row_count,
                })
            }
        }
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
        match self {
            Self::Local(store) => store.delete_by_filter(collection, relative_paths),
            Self::Milvus(client) => {
                let filter = build_relative_path_milvus_filter(relative_paths);
                client.delete(collection, &filter).await
            }
        }
    }
}

fn build_relative_path_milvus_filter(relative_paths: &[String]) -> String {
    let serialized = relative_paths
        .iter()
        .map(|path| serde_json::to_string(path).expect("relative path must serialize"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("metadata[\"relative_path\"] in [{serialized}]")
}

/// Generate a sanitized, hashed collection name from a filesystem path or
/// an operator-provided stable collection identity.
///
/// Identical to rust_sindexer's naming so both tools address the same
/// collections. Set `SINDEXER_COLLECTION_IDENTITY` together with
/// `SINDEXER_COLLECTION_ROOT` to make multiple hosts with different checkout
/// paths share stable, path-scoped collections. Descendant codebases append
/// their path relative to the configured root so separate projects cannot
/// alias the root collection or one another.
pub fn collection_name_from_path(path: &Path) -> String {
    let (identity, root) = &*COLLECTION_SCOPE;
    let identity = scoped_collection_identity(path, identity.as_deref(), root.as_deref());
    collection_name_from_path_with_identity(path, identity.as_deref())
}

fn scoped_collection_identity(
    path: &Path,
    identity: Option<&str>,
    root: Option<&str>,
) -> Option<String> {
    let identity_value = identity.and_then(non_empty_trimmed);
    let root_value = root.and_then(non_empty_trimmed);
    if identity_value.is_some() != root_value.is_some() {
        warn!(
            identity_set = identity_value.is_some(),
            root_set = root_value.is_some(),
            "SINDEXER_COLLECTION_IDENTITY and SINDEXER_COLLECTION_ROOT must both be set for \
             stable cross-host collection naming; ignoring the partial configuration"
        );
    }
    let identity = identity_value?;
    let root = root_value?;
    let path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let root = Path::new(root)
        .canonicalize()
        .unwrap_or_else(|_| Path::new(root).to_path_buf());
    let relative = match path.strip_prefix(&root) {
        Ok(relative) => relative,
        Err(_) => {
            warn!(
                path = %path.display(),
                root = %root.display(),
                "path is outside SINDEXER_COLLECTION_ROOT; collection identity does not apply \
                 and the collection name falls back to hashing the absolute path"
            );
            return None;
        }
    };

    if relative.as_os_str().is_empty() {
        Some(identity.to_string())
    } else {
        // Normalize to forward slashes so the identity is identical across
        // hosts regardless of platform path separator.
        let relative_normalized = relative
            .components()
            .map(|component| component.as_os_str().to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join("/");
        Some(format!("{identity}/{relative_normalized}"))
    }
}

fn collection_name_from_path_with_identity(path: &Path, identity: Option<&str>) -> String {
    let identity = identity.and_then(non_empty_trimmed);
    let collection_identity = identity
        .map(Cow::Borrowed)
        .unwrap_or_else(|| Cow::Owned(path.to_string_lossy().into_owned()));

    // Hash the collection identity (provided identity or full path) for uniqueness.
    let mut hasher = Sha256::new();
    hasher.update(collection_identity.as_bytes());
    let hash = hex::encode(hasher.finalize());
    let hash_prefix = &hash[..16];

    // Extract and sanitize the last component for readability
    let prefix = identity
        .and_then(identity_prefix)
        .or_else(|| path.file_name().and_then(|s| s.to_str()))
        .unwrap_or("collection");

    let sanitized: String = prefix
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();

    // Ensure it starts with a letter or underscore
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

    // Truncate prefix if needed to fit within 255 chars with hash.
    // Char-boundary-safe: identical to byte slicing for today's ASCII-only
    // sanitization (so names stay byte-compatible with rust_sindexer), but
    // cannot panic if the sanitization map ever admits multi-byte chars.
    let max_prefix_len = 255 - 1 - hash_prefix.len(); // 1 for underscore separator
    let prefix_part = if sanitized.len() > max_prefix_len {
        let end = sanitized
            .char_indices()
            .nth(max_prefix_len)
            .map(|(i, _)| i)
            .unwrap_or(sanitized.len());
        &sanitized[..end]
    } else {
        &sanitized
    };

    format!("{}_{}", prefix_part, hash_prefix)
}

fn non_empty_trimmed(value: &str) -> Option<&str> {
    let value = value.trim();
    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}

fn identity_prefix(identity: &str) -> Option<&str> {
    Path::new(identity)
        .file_name()
        .and_then(|s| s.to_str())
        .and_then(non_empty_trimmed)
        .or_else(|| non_empty_trimmed(identity))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_collection_name_basic() {
        let path = PathBuf::from("/home/user/my-project");
        let name = collection_name_from_path_with_identity(&path, None);
        assert!(name.starts_with("my_project_"));
        assert!(name.len() <= 255);
    }

    #[test]
    fn test_collection_name_numeric_start() {
        let path = PathBuf::from("/home/user/123project");
        let name = collection_name_from_path_with_identity(&path, None);
        assert!(name.starts_with('_'));
    }

    #[test]
    fn test_collection_name_special_chars() {
        let path = PathBuf::from("/home/user/my.project-name@v2");
        let name = collection_name_from_path_with_identity(&path, None);
        assert!(!name.contains('.'));
        assert!(!name.contains('-'));
        assert!(!name.contains('@'));
    }

    #[test]
    fn test_collection_name_uniqueness() {
        let path1 = PathBuf::from("/home/user/project");
        let path2 = PathBuf::from("/home/other/project");
        let name1 = collection_name_from_path_with_identity(&path1, None);
        let name2 = collection_name_from_path_with_identity(&path2, None);
        assert_ne!(name1, name2);
    }

    #[test]
    fn test_collection_name_identity_overrides_absolute_path() {
        let mac_path = PathBuf::from("/Users/kearm/AlphaHENG");
        let linux_path = PathBuf::from("/home/kearm/AlphaHENG");

        let mac_name = collection_name_from_path_with_identity(&mac_path, Some("AlphaHENG"));
        let linux_name = collection_name_from_path_with_identity(&linux_path, Some("AlphaHENG"));

        assert_eq!(mac_name, linux_name);
        assert!(mac_name.starts_with("AlphaHENG_"));
    }

    #[test]
    fn test_scoped_identity_keeps_root_stable_across_hosts() {
        let identity = Some("AlphaHENG@nomic768");
        let mac_path = PathBuf::from("/Users/kearm/AlphaHENG");
        let linux_path = PathBuf::from("/home/kearm/AlphaHENG");

        let mac_identity =
            scoped_collection_identity(&mac_path, identity, Some("/Users/kearm/AlphaHENG"));
        let linux_identity =
            scoped_collection_identity(&linux_path, identity, Some("/home/kearm/AlphaHENG"));

        assert_eq!(mac_identity, linux_identity);
        assert_eq!(mac_identity.as_deref(), identity);
    }

    #[test]
    fn test_scoped_identity_namespaces_nested_codebases() {
        let identity = Some("AlphaHENG@nomic768");
        let mac_path = PathBuf::from("/Users/kearm/AlphaHENG/contrib/gigatoken");
        let linux_path = PathBuf::from("/home/kearm/AlphaHENG/contrib/gigatoken");

        let mac_identity =
            scoped_collection_identity(&mac_path, identity, Some("/Users/kearm/AlphaHENG"));
        let linux_identity =
            scoped_collection_identity(&linux_path, identity, Some("/home/kearm/AlphaHENG"));

        assert_eq!(mac_identity, linux_identity);
        assert_eq!(
            mac_identity.as_deref(),
            Some("AlphaHENG@nomic768/contrib/gigatoken")
        );
        let collection =
            collection_name_from_path_with_identity(&mac_path, mac_identity.as_deref());
        assert!(collection.starts_with("gigatoken_"));
    }

    #[test]
    fn test_scoped_identity_does_not_escape_configured_root() {
        let path = PathBuf::from("/Users/kearm/PrismML-Bonsai-MLX-Metal");
        let identity = scoped_collection_identity(
            &path,
            Some("AlphaHENG@nomic768"),
            Some("/Users/kearm/AlphaHENG"),
        );

        assert!(identity.is_none());
    }

    #[test]
    fn test_scoped_identity_requires_a_root() {
        let path = PathBuf::from("/Users/kearm/AlphaHENG");
        let identity = scoped_collection_identity(&path, Some("AlphaHENG@nomic768"), None);

        assert!(identity.is_none());
    }

    #[test]
    fn test_collection_name_blank_identity_falls_back_to_path() {
        let path1 = PathBuf::from("/home/user/project");
        let path2 = PathBuf::from("/home/other/project");
        let name1 = collection_name_from_path_with_identity(&path1, Some("  "));
        let name2 = collection_name_from_path_with_identity(&path2, Some(""));

        assert_ne!(name1, name2);
    }
}
