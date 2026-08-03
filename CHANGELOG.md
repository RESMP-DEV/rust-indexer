# Changelog

## Unreleased

- Fix: parse `upsertCount` from Milvus upsert responses (the endpoint switch
  to upsert left the client reading only `insertCount`, so every accepted
  batch counted as zero vectors and indexing aborted).
- Fix: I/O errors while reading the manifest or provenance record propagate
  instead of degrading to "no provenance", which would have bypassed the
  backend-mismatch safeguards; search treats such errors like a mismatch
  (semantic skipped, lexical served).
- The Milvus client maps the collection-not-found response (code 100) to an
  empty semantic result set directly, removing the per-search existence
  round trip, and warns when only one of `SINDEXER_COLLECTION_IDENTITY` /
  `SINDEXER_COLLECTION_ROOT` is set. Provenance-record writes must succeed
  for an embeddings-enabled run to report success, and Milvus URLs are
  normalized (trailing slashes stripped) at client construction, keeping
  request URLs free of double slashes and backend identities
  slash-insensitive. Search result paths rebuilt from the local checkout
  accept both separator styles, so collections indexed on Windows resolve
  on Unix hosts and vice versa.

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
  against an empty index; vector-backend errors during the collection
  existence check now mark the run failed (previously the CLI could hang
  waiting on the status mirror), with an additional terminal-status safety
  net in `Indexer::run_index`; and lexical-only runs now refuse a path whose
  semantic vector collection exists or the shared status file records a
  nonzero vector count (covering vectors held in a backend the current
  environment cannot see), because refreshing the shared manifest without
  updating vectors would permanently hide semantic staleness from both
  tools. Runs refused before any index mutation restore the pre-run
  persisted status instead of zeroing it, so the recorded vector evidence
  survives repeated refusals; the API layer and status mirror no longer
  persist status at all (the engine is the sole owner of the on-disk
  status file); and hybrid search with a configured Milvus backend now
  returns empty semantic results for a missing collection instead of
  failing, so lexical-only indexes stay searchable; and `clear` refuses to delete
  the shared manifest, status, and lexical index while the recorded status
  shows vectors in a backend the current environment cannot see, since the
  surviving remote collection plus a rebuilt manifest would present stale
  semantic results as current. Mid-run failures preserve the prior
  recorded vector counts unless the run dropped the collection, and the
  local backend must hold at least the recorded row count before clear
  treats the evidence as visible. Embeddings-enabled runs record backend
  provenance in `.sindexer/vector-backend.json` (a rust-indexer-owned
  sidecar that does not change the shared manifest schema); a backend
  switch refuses both `index` and `update` (rebuilding into the wrong
  backend would advance the shared manifest while the recorded backend's
  vectors go stale); `index --force` is the explicit override for
  re-homing an index, so a
  same-named collection in a previously used backend cannot satisfy an
  empty manifest diff with stale vectors. The record stores a SHA-256 of
  the manifest it was written alongside, so a manifest later rewritten by
  rust_sindexer invalidates the record rather than vouching for vectors it
  never described; an absent or unauthenticated record adopts the current
  backend, which is the required behavior for sindexer-built indexes.
  Fully closing the backend flip-flop window would need rust_sindexer to
  write the same provenance record (lockstep follow-up). `search` also
  consults the record: on an authenticated mismatch it skips semantic
  retrieval with a warning instead of fusing stale vectors, while lexical
  results continue to serve, and `clear` refuses an authenticated backend
  mismatch regardless of the recorded vector count, so even an empty
  remote collection must be dropped by the backend that owns it. The
  lexical-only guard likewise treats an authenticated provenance record as
  semantic evidence, covering collections that hold zero vectors.
