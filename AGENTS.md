# rust-indexer — Agent Guide

Standalone CLI for hybrid (BM25 + optional semantic) code search. Forked from
`contrib/rust_sindexer`; that repo remains the MCP server and must not gain
dependencies on this one, nor vice versa.

## Contract

- This is a CLI, not a server. There is no MCP surface and no daemon; do not
  reintroduce them.
- Index compatibility with rust_sindexer is a hard requirement: collection
  naming (identity scoping included), `<repo>/.sindexer/` manifest format,
  `IndexInputs` equality semantics (including the ignore-pattern list), the
  tantivy schema and cache path, and the Milvus metadata JSON layout must
  all stay byte-compatible with the sibling repo. Breaking any of these
  silently forces full reindexes for one tool or the other.
- Every subcommand must work with zero configuration (lexical-only) and keep
  `--json` output stable for scripted callers.
- Local vector persistence is bincode with the typed `ChunkMeta` struct.
  serde_json::Value does not survive bincode round-trips; keep persisted
  types self-describing-format-free. The local store is the only
  non-shared surface (sindexer's local store is JSON).

## Layout

- `src/main.rs` — clap CLI.
- `src/api.rs` — `Indexer`: index/update/search/status/clear/collections.
- `src/engine/` — pipeline (`indexer.rs`), shared state, manifest diffing,
  RRF fusion (`hybrid.rs`).
- `src/walker/`, `src/splitter/`, `src/lexical/`, `src/embedding/`,
  `src/vectordb/` — discovery, AST chunking, tantivy BM25, HTTP embedder,
  local vector store.

## Validation

```bash
cargo fmt && cargo clippy --all-targets && cargo test
```

Tests that touch on-disk indexes must hold the cache-env lock
(`lexical::test_support::set_test_cache_dir`) so parallel tests do not race
on `XDG_CACHE_HOME`.

## Style

Less code is better: single-file modules where possible, no stubs, no
backwards-compatibility shims, delete rather than deprecate. Update
`CHANGELOG.md` for any behavior or interface change.
