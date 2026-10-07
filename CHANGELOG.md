# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-07

First public release.

### Added
- `crush-notebook-core`: the `.crush-nb` document model (cells, kinds, states,
  outputs, execution stats, `@wip`/`@temporary`/`@decision` annotations) and
  its JSON Schema (`schemas/notebook.schema.json`).
- `crush-notebook-kernel`: an MCP server over stdio with 12 tools to open,
  evaluate, edit, save and reload notebooks, plus agent coordination
  (`@wip` claims, checklists, releases, delegation cells).
- Crush, Sona-tagged and Nepali-tagged cells compile with `crush-frontend` and
  run on CVM1. Variables persist across cells.
- Optional Cranelift JIT tier for standalone `fn main` cells (`--features jit`).
- Shared notebooks: atomic saves under a cross-process `.lock` sidecar, and
  reloading when the file changes on disk.
- `crush-notebook-render`: a self-contained HTML page for a notebook, as a
  library function and a CLI.

### Known limitations
- Python and JavaScript cells are not executed; they finish with an error that
  says so.
- AI query and AI-generated cells are stored and rendered, not executed.
- The rendered page's toolbar has no backend yet: there is no browser gateway.
- The default build fetches ONNX Runtime binaries at build time (through
  `crush-vm`'s `native-plugins` feature).

[0.1.0]: https://github.com/nixpt/crush-notebook/releases/tag/v0.1.0
