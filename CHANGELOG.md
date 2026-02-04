# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1](https://github.com/doublewordai/qlm/compare/v0.1.0...v0.1.1) - 2026-02-04

### Added

- add lazy vector index creation and create_vector_index SQL function
- convert vector_search to table-valued function
- add --create-index CLI option for non-REPL usage
- add LanceDB vector search with batch embeddings

### Fixed

- support string IDs and improve error handling for vector search
- llm_unfold fan-out now returns all split items

### Other

- Merge pull request #1 from doublewordai/renovate/configure
- warn about slow lazy index creation for large tables
- fix remaining 'large datasets' framing
- clarify batching benefits - consistency over token savings
- add semantic search / vector_search documentation
- restructure README around use cases
- release v0.1.0

## [0.1.0](https://github.com/doublewordai/qlm/releases/tag/v0.1.0) - 2026-02-04

### Added

- add release-plz for automated crates.io publishing

### Fixed

- correct rust-toolchain action name in release-plz workflow

### Other

- Initial commit
