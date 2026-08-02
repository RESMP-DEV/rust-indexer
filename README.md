# rust-indexer

Fast, local-first CLI for hybrid code search: BM25 lexical retrieval fused
with optional semantic embeddings, in a single native binary.

Point it at a codebase, index it once, then search from any shell or agent.
No server process, no vector database, no configuration required.

```bash
rust-indexer index ~/code/my-project
rust-indexer search "where do we retry failed uploads" -p ~/code/my-project
```

## Install

```bash
cargo build --release
install -m 755 target/release/rust-indexer ~/.local/bin/
```

## Usage

```
rust-indexer index [PATH] [--force]    Build (or manifest-guided rebuild of) the index
rust-indexer update [PATH]             Incremental update: changed/deleted files only
rust-indexer search QUERY [-p PATH] [-k N] [-e EXT]...
rust-indexer status [PATH]             Index state and counts
rust-indexer clear [PATH]              Remove vector + lexical index for PATH
rust-indexer collections               List indexed collections and row counts
```

`PATH` defaults to the current directory. Add `--json` to any command for
machine-readable output (useful when calling from agents or scripts).

## Modes

- **Lexical only (default)** — zero configuration. Tantivy BM25 over
  tree-sitter AST chunks. Good for symbols and exact terms.
- **Hybrid semantic + lexical** — set `EMBEDDING_URL` to any OpenAI-compatible
  embeddings endpoint. Results are fused with reciprocal rank fusion.
- **Milvus/Zilliz backend** — set `MILVUS_URL` to store vectors remotely
  instead of the local store.

## Shared index with rust_sindexer

rust-indexer is index-compatible with the
[rust_sindexer](https://github.com/RESMP-DEV/rust_sindexer) MCP server:
identical collection naming (including `SINDEXER_COLLECTION_IDENTITY` /
`SINDEXER_COLLECTION_ROOT` scoping), the same `<repo>/.sindexer/` manifest,
the same tantivy lexical cache, and the same Milvus metadata layout. A
codebase indexed by the MCP server can be searched and incrementally updated
by this CLI, and vice versa, without re-embedding anything. Variables from
`~/.context/.env` are loaded automatically (existing environment wins), so
both tools see one configuration.

## Storage

- `<repo>/.sindexer/` — index manifest (per-file SHA-256) and status,
  shared with rust_sindexer.
- `$XDG_CACHE_HOME/sindexer/lexical-indexes/` — tantivy lexical indexes,
  shared with rust_sindexer.
- `$XDG_CACHE_HOME/rust-indexer/vector-indexes/` — binary (bincode) local
  vector collections. Only the local vector store is private to this tool;
  rust_sindexer's local store uses JSON and the formats are not
  interchangeable. With `MILVUS_URL` set this directory is unused.

The binary vector format loads a 500-chunk collection in about a millisecond,
so per-invocation cold start is negligible; search latency is dominated by
the embedding endpoint when semantic mode is enabled.

## Environment variables

All optional.

- `EMBEDDING_URL` — OpenAI-compatible embeddings base URL; enables semantic
  search (`OPENAI_BASE_URL` also accepted).
- `EMBEDDING_API_KEY` — bearer token if the endpoint needs one
  (`OPENAI_API_KEY` also accepted).
- `EMBEDDING_MODEL` — model name (default `all-minilm`).
- `EMBEDDING_DIMENSION` — vector dimension, must match the model (default `384`).
- `EMBEDDING_QUERY_PREFIX` / `EMBEDDING_PASSAGE_PREFIX` — task-instruction
  prefixes for models that use them.
- `EMBEDDING_RPM` / `EMBEDDING_TPM` — client-side rate limits (defaults 400 / 1.6M).
- `MILVUS_URL` — Milvus/Zilliz endpoint; enables the remote vector backend
  (`MILVUS_ADDRESS` also accepted).
- `MILVUS_TOKEN` — bearer token for authenticated Milvus endpoints.
- `SINDEXER_COLLECTION_IDENTITY` / `SINDEXER_COLLECTION_ROOT` — stable
  cross-host collection naming, shared with rust_sindexer.
- `CHUNK_SIZE`, `CHUNK_OVERLAP`, `BATCH_SIZE`, `INDEXING_CONCURRENCY`,
  `MAX_FILE_SIZE`, `FOLLOW_SYMLINKS` — pipeline tuning.
- `RUST_LOG` — tracing filter; logs go to stderr.

## Architecture

```
Walker (ignore-aware) → Splitter (tree-sitter AST) → Embedder (HTTP, optional)
                                   │                        │
                              Lexical (tantivy BM25)   Vector store (bincode)
                                   └────────┬───────────────┘
                                     Hybrid fusion (RRF)
```

Incremental updates diff a per-file SHA-256 manifest and re-process only
added, modified, and deleted files. `update` refuses to fall back to a full
rebuild; use `index --force` when you actually want one.

Supported AST languages: Python, JavaScript, TypeScript, TSX, Rust, Go, Java,
C++, C, Ruby, PHP, Swift, Scala, C#. Other supported file types fall back to
markdown-heading or line-based splitting.

## Provenance

Forked from [rust_sindexer](https://github.com/RESMP-DEV/rust_sindexer), the
MCP server variant. This project drops the MCP protocol surface in favor of a
plain CLI while staying index-compatible with its sibling. If you need an MCP
server, use rust_sindexer; both can share the same index.

## Development

```bash
cargo test
cargo clippy --all-targets
cargo fmt
```
