# crush-notebook

A notebook for the [Crush](https://github.com/nixpt/crush-ast) language, built
for humans and AI agents working on the same document.

- A notebook is a plain JSON file (`.crush-nb`). The file is the state: there
  is no long-lived server you have to keep alive, and notebooks diff and merge
  in git.
- The kernel is an [MCP](https://modelcontextprotocol.io) server over stdio.
  Any MCP client (an editor, an agent harness) can open a notebook, run cells,
  read variables and edit cells.
- Agents coordinate through cell metadata: `@wip` ownership claims with
  checklists, delegation cells, and atomic saves that several kernel processes
  can share.
- The renderer turns a notebook into one self-contained HTML page (no CDN, no
  external assets).

Status: early (0.1). Crush cells work end to end. Several cell kinds are
stored and rendered but not executed yet; see [Cell kinds](#cell-kinds).

## Crates

| Crate | What it is |
|-------|------------|
| `crush-notebook-core` | The document model: `NotebookDocument`, `Cell`, `CellKind`, `CellState`, outputs, annotations. Matches `schemas/notebook.schema.json`. |
| `crush-notebook-kernel` | The `crush-notebook-kernel` binary: an MCP server that evaluates cells with `crush-frontend`, `crush-lang-sdk` and `crush-vm`. |
| `crush-notebook-render` | `render_html()` and the `crush-notebook-render` binary: notebook to a self-contained HTML page. |

## Quick start

Install the two binaries:

```sh
cargo install crush-notebook-kernel crush-notebook-render
```

Or, from a clone of this repository, use `cargo run -p <crate> --`.

### Render a notebook

```sh
crush-notebook-render examples/hello.crush-nb            # writes examples/hello.html
crush-notebook-render examples/hello.crush-nb --out /tmp/hello.html
```

The page shows cells, outputs, states, `@wip` claims and a tier summary. Its
toolbar buttons post to a `/mcp` endpoint that doesn't exist yet: there is no
browser gateway, so treat the page as read-only for now.

### Run the kernel as an MCP server

The kernel speaks JSON-RPC 2.0 on stdin/stdout (logs go to stderr). Register it
with your MCP client, for example:

```json
{
  "mcpServers": {
    "crush-notebook": { "command": "crush-notebook-kernel" }
  }
}
```

Or drive it by hand:

```sh
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"notebook_open","arguments":{"path":"examples/hello.crush-nb"}}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"notebook_eval_all","arguments":{}}}' \
  '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"notebook_list_vars","arguments":{}}}' \
  | crush-notebook-kernel
```

The last response lists the session variables. Running the cells also saves
their results back into the file, so the example notebook in this repository
will be modified.

```text
6 vars:
  x: int = 42
  y: int = 10
  total: int = 52
  name: str = "Crush Notebook"
  multiplier: int = 3
  scaled: int = 156
```

Tools: `notebook_open`, `notebook_eval_cell`, `notebook_eval_all`,
`notebook_list_vars`, `notebook_insert_cell`, `notebook_delete_cell`,
`notebook_get_state`, `notebook_save`, `notebook_reload`,
`notebook_claim_cell`, `notebook_update_wip`, `notebook_release_cell`. Every
mutation is saved to the file (atomic rename under a `.lock` sidecar), and
before each evaluation the kernel reloads the file if another process changed
it. See [docs/AGENT-NATIVE.md](docs/AGENT-NATIVE.md) for the agent workflow.

## Cell kinds

| Kind | Status |
|------|--------|
| `crush` | **Runs.** Compiled by `crush-frontend` and lowered by `crush-lang-sdk`, the same path as `crush run`, then executed on the CVM1 interpreter (`crush-vm`). Variables, functions and structs persist across cells (see below). |
| `sona` | **Runs, as Crush.** crush-ast has no separate Sona parser yet: these cells compile through the Crush front end and are tagged `sona`. |
| `nepali` | **Runs, as Crush.** Same as `sona`: tagged `nepali`, compiled by the Crush front end. |
| `markdown` | Rendered; nothing to execute. |
| `python`, `javascript` | **Not executed.** The cell finishes with an error saying so. The kernel doesn't yet service `crush-vm`'s polyglot (`exec_lang`) requests, and doing so means running notebook code with full host access, so it will need an explicit opt-in. |
| `ai_query`, `ai_generated` | Stored and rendered, not executed. The kernel has no model backend; an agent answers the query and adds the proposal with `notebook_insert_cell`. |
| `ai_agent_delegate` | Sends the cell's task to another agent through an external `squad-msg`-style command (`CRUSH_NOTEBOOK_SQUAD_MSG`, default `squad-msg`); the cell stays `ai_pending`. Only useful where such a command exists. |

### What a cell may do

A cell runs with the capabilities `crush run FILE` grants when given no flags:

- `print` and the other VM built-ins (`str.len`, `str.concat`, `str.contains`,
  `str.split`, `str.join`, `str.replace`, `conv.chr`, `conv.ord`; `io.read`
  sees an empty input),
- `cson.parse`.

Nothing else: no filesystem, network, environment, process, clock, crypto or
polyglot (`@python { … }`) access. A cell that calls one of those ends in an
error naming the capability (for example `unknown capability: fs.read`).

### Sharing between cells

Each Crush-family cell is parsed as a script, the way `crush run` reads a
`.crush` file: top-level statements form `main`, and top-level `fn` and
`struct` declarations are allowed. The kernel puts the session in front of the
cell (structs, then variables, with the session's functions alongside), runs
it, and then keeps:

- every top-level `fn` and `struct` the cell declares (a later declaration with
  the same name replaces the earlier one),
- the cell's top-level `let` bindings and any session variables it reassigned.

Limits:

- Only values with a literal form cross a cell boundary: null, bool, int,
  float, string, and arrays/maps of those. Other values stay in their cell,
  and the cell output says so.
- A cell that contains `return` can read session variables, but its own
  bindings are not kept.
- A cell that defines its own `fn main` is a standalone program: it can use
  the session's functions and structs, but doesn't see or change its
  variables.
- A struct instance crosses a cell boundary as a map of its fields.
- Opening a notebook starts a fresh session.

### Execution tiers

Session cells run on CVM1. Building the kernel with `--features jit` adds the
Cranelift JIT (`crush-jit`) for standalone `fn main` Crush cells that call no
capabilities (no `print`), falling back to FastVM when the JIT can't compile a
program. Programs that call a capability run on CVM1. The feature is off by
default.

## Developing

```sh
cargo test --workspace
cargo test -p crush-notebook-kernel --features jit
```

The crush-ast crates come from crates.io. To build against a local crush-ast
checkout, add a patch to your own `.cargo/config.toml` (don't commit it):

```toml
[patch.crates-io]
crush-frontend = { path = "../crush-ast/crates/crush-frontend" }
crush-vm = { path = "../crush-ast/crates/crush-vm" }
casm = { path = "../crush-ast/crates/casm" }
crush-cast = { path = "../crush-ast/crates/crush-cast" }
crush-lang-sdk = { path = "../crush-ast/crates/crush-lang-sdk" }
```

The default build enables `crush-vm`'s `native-plugins` feature (the kernel
uses its FastVM). That pulls in `ort`, which downloads ONNX Runtime binaries at
build time, so a fully offline build doesn't work yet.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
