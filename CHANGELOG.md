# Changelog

## Unreleased

- Initial fork from rust_sindexer (MCP server variant) as a standalone CLI
  with clap subcommands `index`, `update`, `search`, `status`, `clear`, and
  `collections`, each supporting `--json` output. The MCP protocol surface
  (rmcp) is removed.
- Index compatibility with rust_sindexer is preserved: identical collection
  naming (including `SINDEXER_COLLECTION_IDENTITY` / `SINDEXER_COLLECTION_ROOT`
  scoping), the shared `<repo>/.sindexer/` manifest, the shared tantivy
  lexical cache, the same Milvus metadata layout, and Milvus i64 id
  canonicalization in hybrid fusion. When both tools use the Milvus/Zilliz
  backend, an index built by the MCP server is searchable and incrementally
  updatable from this CLI and vice versa; lexical indexes are shared in all
  modes.
- The local (no `MILVUS_URL`) vector store is the one non-shared surface: it
  persists bincode with a typed `ChunkMeta` because `serde_json::Value`
  cannot round-trip through bincode, while rust_sindexer's local store uses
  JSON. Cold-start collection loads are ~1ms at 500-chunk scale.
- `~/.context/.env` is loaded automatically (existing environment wins) so
  the CLI sees the same embedding and Milvus configuration as the sindexer
  wrapper script.
- Review hardening (PR #1): the env-file loader strips both quote styles and
  rejects NUL bytes and unbalanced quotes; Milvus auth-header construction
  no longer panics on invalid tokens and marks the header sensitive; chunk
  metadata serialization errors propagate instead of panicking; metadata
  layout mismatches on search log a warning instead of silently returning
  empty paths; a discarded collection identity (path outside
  `SINDEXER_COLLECTION_ROOT`) logs a warning; incremental `update` fails
  fast when the lexical cache is missing instead of creating a fresh one;
  search results rebuild `file_path` from the local checkout root so
  identity-scoped collections shared across hosts resolve to local files;
  the unused Milvus `Document`/`insert` API was removed; and `clear` now
  deletes the index manifest along with the lexical index so the next
  `index`/`update` rebuilds instead of reporting "already up to date"
  against an empty index.
