# Changelog

## Unreleased

- Initial fork from rust_sindexer (MCP server variant) as a standalone CLI.
- Removed the MCP protocol surface (rmcp) and the Milvus/Zilliz vector
  backend; the local store is now the only vector backend.
- Replaced JSON vector persistence with bincode and a typed `ChunkMeta`
  (serde_json::Value cannot round-trip through bincode), making cold-start
  collection loads ~1ms at 500-chunk scale.
- Added clap subcommands: `index`, `update`, `search`, `status`, `clear`,
  `collections`, each with `--json` output.
- Renamed on-disk locations: repo manifest in `.rust-indexer/`, caches under
  `$XDG_CACHE_HOME/rust-indexer/`.
- Simplified collection identity to path-hash naming; removed the
  cross-host `SINDEXER_COLLECTION_IDENTITY`/`SINDEXER_COLLECTION_ROOT`
  machinery and `MILVUS_*` configuration.
