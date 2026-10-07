# Agent-Native Notebook Execution Model

## Core Idea

A crush notebook is a **shared workspace** between humans and AI agents. Cells are the unit of collaboration: an agent can write, execute, observe, and modify cells just like a human. The notebook file is the **persistent state** across sessions, and each cell's `@wip`/`@temporary`/`@decision` metadata gives agents context without parsing source.

## Agent Capabilities

All of these are tools of the `crush-notebook-kernel` MCP server.

| Action | MCP Tool | What it does |
|--------|----------|-------------|
| Load notebook | `notebook_open` | Load a .crush-nb file and start a fresh variable session |
| Execute cell | `notebook_eval_cell` | Run a cell, get output/error |
| Execute all | `notebook_eval_all` | Run every cell in order |
| Add cell | `notebook_insert_cell` | Write new code; pass `agent` to record ownership |
| Delete cell | `notebook_delete_cell` | Remove a cell |
| Claim cell | `notebook_claim_cell` | Atomically claim a cell through `@wip.started_by` |
| Update WIP | `notebook_update_wip` | Update `done`/`todo`/`unresolved` as owner |
| Release cell | `notebook_release_cell` | Release an owned claim while retaining WIP history |
| Delegate task | `notebook_insert_cell` (`ai_agent_delegate`) | Send a task to another agent through an external messaging command |
| Read state | `notebook_get_state` | Snapshot: cell states, source previews, output counts, variables |
| Save notebook | `notebook_save` | Atomically persist through a cross-process file lock |
| Reload notebook | `notebook_reload` | Pick up a complete externally saved snapshot |
| Inspect vars | `notebook_list_vars` | Session variables and their values |

Outside the kernel, agents typically pair it with a browser-automation tool (to look at the rendered page) and whatever channel they use to talk to each other.

## Agent Workflow

### Orientation (arriving at a notebook)
```
1. notebook_open("examples/hello.crush-nb")
2. notebook_get_state() → cells, states, variables
3. Read each cell's meta in the .crush-nb file:
   wip → what's in progress and who owns it
   decision → why choices were made
   temporary → what's temporary and when it expires
```

### Writing + executing (doing work)
```
1. notebook_insert_cell({kind: "crush", source: "let total = x + y", agent: "agent-a"})
2. If the cell already exists, notebook_claim_cell({cell_id: "cell-4", agent: "agent-a", intent: "Analyze data"})
3. notebook_update_wip({cell_id: "cell-4", agent: "agent-a", todo: ["run tests"]})
4. notebook_eval_cell("cell-4")
5. If error: read error output, fix source, re-eval
6. If done: update WIP with `done` and clear completed `todo` items
```

Top-level `let` bindings of a Crush cell stay available to later cells; see the README's "Variables across cells".

### Delegation (asking another agent)
```
1. notebook_insert_cell({kind: "ai_agent_delegate", delegate_to: "agent-b", task: "Review the output", source: "Review the output"})
2. notebook_eval_cell("cell-5")
3. Read the delegation acknowledgement; the cell remains `ai_pending` until the target agent writes back
```
Evaluating a delegation cell runs the command named by `CRUSH_NOTEBOOK_SQUAD_MSG` (default `squad-msg`) with two arguments: `@<agent>` and the message. Arguments are passed directly to the process, so task text is not interpreted by a shell. Set `AGENT_NAME` (or `CRUSH_NOTEBOOK_AGENT`) to identify the sender. Point the variable at any script that forwards the message to your agents.

### Visual verification
```
1. crush-notebook-render notebook.crush-nb --out render.html
2. Open file:///path/to/render.html in a browser-automation tool
3. Screenshot it and check the output looks right
```

### Collaboration (working with others)
```
1. notebook_claim_cell({cell_id: "cell-4", agent: "agent-a", intent: "Review proposal"})
2. Tell the other agents: "agent-a is reviewing cell-4"
3. notebook_update_wip({cell_id: "cell-4", agent: "agent-a", done: ["review"], todo: []})
4. notebook_release_cell({cell_id: "cell-4", agent: "agent-a"})
```
Claims are exclusive: another agent receives a conflict until the owner explicitly releases the claim. Releasing clears only `@wip.started_by`, retaining the checklist and intent for the next agent. Agent-created cells should pass `agent` to `notebook_insert_cell`, which creates the `@wip.started_by` metadata automatically.

All notebook mutations are saved through a per-file `<notebook>.lock` sidecar. The kernel writes a unique temporary file beside the notebook, flushes it, and renames it into place while holding the lock, so readers see either the previous complete JSON document or the next complete document, never a partial write. Lock contention returns an error after a bounded wait instead of hanging an MCP request. Before evaluating a cell, the kernel reloads the file if another process changed it.

## Architecture

```
+-------------------------------------------------------------+
|                       Agent session                         |
|                                                             |
|  +----------------+  +----------------+  +---------------+  |
|  | crush-notebook |  | browser        |  | agent-to-     |  |
|  | -kernel (MCP)  |  | automation     |  | agent channel |  |
|  | cell CRUD/eval |  | (visual check) |  | (coordination)|  |
|  +-------+--------+  +-------+--------+  +-------+-------+  |
|          |                   |                   |          |
|          v                   v                   v          |
|  +-------------------------------------------------------+  |
|  |               Crush notebook (.crush-nb)              |  |
|  |   cells + outputs + @wip/@decision/@temporary meta    |  |
|  |   + execution stats                                   |  |
|  +-------------------------------------------------------+  |
+-------------------------------------------------------------+
```

## Compared with Jupyter, for agents

| Jupyter | Crush-Notebook |
|---------|---------------|
| Agents parse comments for context | `@wip`, `@decision`, `@temporary` are structured cell metadata |
| Kernel process holds the state | The notebook is a file; `notebook_open` is all you need |
| No built-in multi-agent coordination | `@wip.started_by` claims say who is doing what |
| No agent delegation | `ai_agent_delegate` cells send a task and record the pending state |
| Output is inline HTML, hard to parse | `notebook_get_state()` returns structured JSON |
| Single user | Shared notebook: agents and humans edit cells together, with atomic saves |
