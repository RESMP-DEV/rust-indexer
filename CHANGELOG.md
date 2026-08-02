# Changelog

## Unreleased

- Share one index with rust_sindexer: restored the Milvus/Zilliz backend and
  identity-scoped collection naming (`SINDEXER_COLLECTION_IDENTITY` /
  `SINDEXER_COLLECTION_ROOT`), moved the manifest back to `<repo>/.sindexer/`
  and the lexical cache back to `$XDG_CACHE_HOME/sindexer/`, and restored the
  Milvus i64 id canonicalization in hybrid fusion. An index built by the MCP
  server is searchable and updatable from this CLI and vice versa.
- Auto-load `~/.context/.env` (existing environment wins) so the CLI sees the
  same embedding and Milvus configuration as the sindexer wrapper script.
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
