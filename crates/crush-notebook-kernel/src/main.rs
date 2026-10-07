//! Crush-Notebook Kernel — MCP server for cell evaluation.
//!
//! Provides 7 tools over JSON-RPC stdio:
//!   notebook_open, eval_cell, eval_all, list_vars, insert_cell, delete_cell, get_state
//!
//! Two-lock design: notebook (Arc<Mutex<>>) is separate from session variables
//! (Arc<Mutex<>>), so eval can hold the vars lock while reading/writing cells
//! without deadlocking.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use clap::Parser;
use crush_cast::{Expression, Statement};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::Mutex;

use crush_notebook_core::{
    Cell, CellMeta, CellOutput, CellState, ExecutionStats, ExecutionTier, NotebookDocument,
    OutputKind, WipAnnotation,
};

#[derive(Parser, Debug)]
#[command(name = "crush-notebook-kernel")]
struct Args {
    #[arg(short, long)]
    verbose: bool,
}

// ── MCP types ────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
struct RpcMessage {
    #[serde(default)]
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}

#[derive(Debug, Serialize)]
struct RpcResponse {
    jsonrpc: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<RpcError>,
}

#[derive(Debug, Serialize, PartialEq)]
struct RpcError {
    code: i32,
    message: String,
}

impl RpcResponse {
    fn ok(id: Option<Value>, result: Value) -> Self {
        RpcResponse {
            jsonrpc: "2.0",
            id,
            result: Some(result),
            error: None,
        }
    }
    fn err(id: Option<Value>, code: i32, message: impl Into<String>) -> Self {
        RpcResponse {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(RpcError {
                code,
                message: message.into(),
            }),
        }
    }
}

// ── State (two-lock design: notebook + vars are independent) ���────────────

struct NotebookState {
    doc: Option<NotebookDocument>,
    file_path: Option<String>,
    /// File identity at the last load/save, used to detect external changes.
    last_save_stamp: Option<FileStamp>,
    run_count: usize,
}

/// Identity of a notebook file on disk. mtime alone misses a write that lands
/// in the same timestamp tick as our own save; every save is an atomic rename,
/// so the inode changes too, and comparing the whole stamp catches it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FileStamp {
    mtime: Option<std::time::SystemTime>,
    len: u64,
    #[cfg(unix)]
    ino: u64,
    #[cfg(unix)]
    dev: u64,
}

impl FileStamp {
    fn of(meta: &std::fs::Metadata) -> Self {
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;
        FileStamp {
            mtime: meta.modified().ok(),
            len: meta.len(),
            #[cfg(unix)]
            ino: meta.ino(),
            #[cfg(unix)]
            dev: meta.dev(),
        }
    }
}

/// Session variables, in first-binding order. Crush-family cells read them
/// through an injected prelude and hand their top-level bindings back.
struct Vars {
    bindings: Vec<(String, SessionValue)>,
}

impl Vars {
    fn new() -> Self {
        Vars {
            bindings: Vec::new(),
        }
    }

    fn set(&mut self, name: String, value: SessionValue) {
        match self.bindings.iter_mut().find(|(n, _)| *n == name) {
            Some(slot) => slot.1 = value,
            None => self.bindings.push((name, value)),
        }
    }

    fn eval(
        &mut self,
        source: &str,
        kind: &crush_notebook_core::CellKind,
        run: usize,
    ) -> (CellState, Vec<CellOutput>, Option<ExecutionStats>) {
        match kind {
            crush_notebook_core::CellKind::Crush => self.eval_session_cell(source, run, "crush"),
            crush_notebook_core::CellKind::Nepali => self.eval_session_cell(source, run, "nepali"),
            crush_notebook_core::CellKind::Sona => self.eval_session_cell(source, run, "sona"),
            crush_notebook_core::CellKind::Python => eval_polyglot(source, run, "python"),
            crush_notebook_core::CellKind::JavaScript => eval_polyglot(source, run, "javascript"),
            crush_notebook_core::CellKind::Markdown => (CellState::Done, vec![], None),
            _ => eval_sim_source(source, run),
        }
    }

    /// Run a Crush-family cell as the body of `main`, with the session's
    /// variables in scope, then store the cell's top-level bindings.
    ///
    /// A cell that defines its own `fn main` is a standalone program: it runs
    /// without the session, as before.
    fn eval_session_cell(
        &mut self,
        source: &str,
        run: usize,
        lang: &str,
    ) -> (CellState, Vec<CellOutput>, Option<ExecutionStats>) {
        if source.contains("fn main") {
            #[cfg(feature = "jit")]
            if lang == "crush" {
                return eval_jit_source(source, run);
            }
            return eval_crush_source(source, run, lang);
        }
        let t0 = std::time::Instant::now();
        let mut program = match crush_frontend::parse_source(&format!("fn main() {{\n{source}\n}}"))
        {
            Ok(p) => p,
            Err(e) => return error_cell(run, format!("Compile: {e}")),
        };
        let entry = program.entry.clone();
        let Some(main) = program.functions.get_mut(&entry) else {
            return error_cell(run, format!("Compile: no entry function `{entry}`"));
        };
        // A cell with its own `return` can't also return the capture (the
        // front end infers one return type per function), so its bindings stay
        // cell-local; it still sees the session's variables.
        let captured = if contains_return(&main.body) {
            Vec::new()
        } else {
            session_capture_names(&self.bindings, &main.body)
        };
        let mut body: Vec<Statement> = self
            .bindings
            .iter()
            .map(|(name, value)| Statement::VarDecl {
                name: name.clone(),
                value: value.to_expr(),
                type_hint: Default::default(),
                meta: HashMap::new(),
            })
            .collect();
        body.append(&mut main.body);
        if !captured.is_empty() {
            let mut elements = vec![Expression::StringLiteral {
                value: SESSION_SENTINEL.to_string(),
                meta: HashMap::new(),
            }];
            elements.extend(captured.iter().map(|name| Expression::Var {
                name: name.clone(),
                meta: HashMap::new(),
            }));
            body.push(Statement::Return {
                value: Some(Expression::ArrayLiteral {
                    elements,
                    meta: HashMap::new(),
                }),
                meta: HashMap::new(),
            });
        }
        main.body = body;
        let mut casm_program = match crush_frontend::compile_cast_owned(program) {
            Ok(p) => p,
            Err(e) => return error_cell(run, format!("Compile: {e}")),
        };
        casm_program.lang = Some(lang.to_string());
        let result = match run_cvm1(&casm_program) {
            Ok(r) => r,
            Err(e) => return error_cell(run, e),
        };
        let mut out = Vec::new();
        if !result.output.is_empty() {
            out.push(CellOutput {
                id: format!("out-{run}"),
                kind: OutputKind::Text {
                    text: result.output.clone(),
                },
                data: json!({"output": result.output}),
                timestamp: None,
            });
        }
        // A cell that returns early never reaches the capture; its bindings
        // simply don't persist.
        if let Some(values) = result.stack.last().and_then(session_values) {
            let mut skipped = Vec::new();
            for (name, value) in captured.into_iter().zip(values) {
                match SessionValue::from_vm(&value) {
                    Some(v) => self.set(name, v),
                    None => skipped.push(name),
                }
            }
            if !skipped.is_empty() {
                let text = format!(
                    "not kept in the session (no literal form): {}",
                    skipped.join(", ")
                );
                out.push(CellOutput {
                    id: format!("warn-{run}"),
                    kind: OutputKind::Text { text },
                    data: json!({"skipped": skipped}),
                    timestamp: None,
                });
            }
        }
        let stats = ExecutionStats {
            steps: result.steps,
            duration_ms: t0.elapsed().as_millis() as u64,
            tier: ExecutionTier::Cvm1,
            frontend: lang.into(),
            jit_compiled: false,
        };
        (CellState::Done, out, Some(stats))
    }
}

// ── Session values ───────────────────────────────────────────────────

/// First element of the array a session cell returns, so a cell's own
/// `return [..]` is never mistaken for captured bindings.
const SESSION_SENTINEL: &str = "__crush_notebook_session__";

/// A variable value that can cross a cell boundary: anything with a CAST
/// literal form. Functions, handles, bytes and the like stay cell-local.
#[derive(Clone, Debug, PartialEq)]
enum SessionValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Array(Vec<SessionValue>),
    Map(Vec<(String, SessionValue)>),
}

impl SessionValue {
    fn from_vm(value: &crush_vm::vm::Value) -> Option<Self> {
        use crush_vm::vm::Value as V;
        Some(match value {
            V::Null => SessionValue::Null,
            V::Bool(b) => SessionValue::Bool(*b),
            V::Int(i) => SessionValue::Int(*i),
            V::Float(f) => SessionValue::Float(*f),
            V::Str(s) => SessionValue::Str(s.clone()),
            V::Array(items) => SessionValue::Array(
                items
                    .borrow()
                    .iter()
                    .map(SessionValue::from_vm)
                    .collect::<Option<_>>()?,
            ),
            V::Map(map) => {
                let mut entries = map
                    .borrow()
                    .iter()
                    .map(|(k, v)| SessionValue::from_vm(v).map(|v| (k.clone(), v)))
                    .collect::<Option<Vec<_>>>()?;
                entries.sort_by(|a, b| a.0.cmp(&b.0));
                SessionValue::Map(entries)
            }
            _ => return None,
        })
    }

    fn to_expr(&self) -> Expression {
        let meta = HashMap::new();
        match self {
            SessionValue::Null => Expression::NullLiteral { meta },
            SessionValue::Bool(value) => Expression::BoolLiteral {
                value: *value,
                meta,
            },
            SessionValue::Int(value) => Expression::IntLiteral {
                value: *value,
                meta,
            },
            SessionValue::Float(value) => Expression::FloatLiteral {
                value: *value,
                meta,
            },
            SessionValue::Str(value) => Expression::StringLiteral {
                value: value.clone(),
                meta,
            },
            SessionValue::Array(items) => Expression::ArrayLiteral {
                elements: items.iter().map(SessionValue::to_expr).collect(),
                meta,
            },
            SessionValue::Map(entries) => Expression::ObjectLiteral {
                properties: entries
                    .iter()
                    .map(|(k, v)| (k.clone(), v.to_expr()))
                    .collect(),
                meta,
            },
        }
    }

    fn type_name(&self) -> &'static str {
        match self {
            SessionValue::Null => "null",
            SessionValue::Bool(_) => "bool",
            SessionValue::Int(_) => "int",
            SessionValue::Float(_) => "float",
            SessionValue::Str(_) => "str",
            SessionValue::Array(_) => "array",
            SessionValue::Map(_) => "map",
        }
    }

    fn to_json(&self) -> Value {
        match self {
            SessionValue::Null => Value::Null,
            SessionValue::Bool(b) => json!(b),
            SessionValue::Int(i) => json!(i),
            SessionValue::Float(f) => json!(f),
            SessionValue::Str(s) => json!(s),
            SessionValue::Array(items) => {
                Value::Array(items.iter().map(SessionValue::to_json).collect())
            }
            SessionValue::Map(entries) => Value::Object(
                entries
                    .iter()
                    .map(|(k, v)| (k.clone(), v.to_json()))
                    .collect(),
            ),
        }
    }
}

/// Names a session cell hands back: every session variable (the cell may
/// reassign it) plus each name the cell declares at the top level of its body.
fn session_capture_names(bindings: &[(String, SessionValue)], body: &[Statement]) -> Vec<String> {
    let mut names: Vec<String> = bindings.iter().map(|(n, _)| n.clone()).collect();
    for stmt in body {
        if let Statement::VarDecl { name, .. } = stmt {
            if !names.contains(name) {
                names.push(name.clone());
            }
        }
    }
    names
}

/// Whether any statement in `body`, at any depth, is a `return`.
fn contains_return(body: &[Statement]) -> bool {
    fn walk(node: &Value) -> bool {
        match node {
            Value::Object(map) => {
                map.get("type").and_then(Value::as_str) == Some("Return") || map.values().any(walk)
            }
            Value::Array(items) => items.iter().any(walk),
            _ => false,
        }
    }
    // Statement is serde-tagged by "type"; walking its JSON form covers every
    // nested body without mirroring the CAST enum here.
    serde_json::to_value(body).map(|v| walk(&v)).unwrap_or(true)
}

/// The captured values, if `value` is a session capture array.
fn session_values(value: &crush_vm::vm::Value) -> Option<Vec<crush_vm::vm::Value>> {
    let crush_vm::vm::Value::Array(items) = value else {
        return None;
    };
    let items = items.borrow();
    match items.first() {
        Some(crush_vm::vm::Value::Str(tag)) if tag == SESSION_SENTINEL => Some(items[1..].to_vec()),
        _ => None,
    }
}

// ── Kernel ───────────────────────────────────────────────────────────────

struct Kernel {
    nb: Arc<Mutex<NotebookState>>,
    vars: Arc<Mutex<Vars>>,
}

impl Kernel {
    fn new() -> Self {
        Kernel {
            nb: Arc::new(Mutex::new(NotebookState {
                doc: None,
                file_path: None,
                last_save_stamp: None,
                run_count: 0,
            })),
            vars: Arc::new(Mutex::new(Vars::new())),
        }
    }

    /// Save the current notebook to its file. No-op if no file path.
    async fn save(&self) -> Result<(), String> {
        // Keep the in-process lock while waiting for the cross-process lock so
        // concurrent requests cannot save an older in-memory snapshot later.
        let mut nb = self.nb.lock().await;
        let doc = nb.doc.as_ref().ok_or("No notebook loaded".to_string())?;
        let path = nb
            .file_path
            .as_ref()
            .ok_or("No file path (notebook not opened from file)".to_string())?
            .clone();
        let mut doc = doc.clone();
        doc.meta.modified = Some(chrono_now());
        let json = serde_json::to_string_pretty(&doc).map_err(|e| format!("serialize: {e}"))?;
        let path_ref = Path::new(&path);
        let _lock = acquire_save_lock(path_ref, SAVE_LOCK_TIMEOUT).await?;
        atomic_write(path_ref, json.as_bytes()).await?;
        let saved = tokio::fs::metadata(path_ref)
            .await
            .map_err(|e| format!("stat after save: {e}"))?;
        nb.last_save_stamp = Some(FileStamp::of(&saved));
        Ok(())
    }

    /// Reload the notebook from disk if it changed since last save.
    /// Returns true if a reload happened.
    async fn maybe_reload(&self) -> Result<bool, String> {
        let nb = self.nb.lock().await;
        let path = match &nb.file_path {
            Some(p) => p.clone(),
            None => return Ok(false),
        };
        let last_stamp = nb.last_save_stamp;
        drop(nb);

        let meta = tokio::fs::metadata(&path)
            .await
            .map_err(|e| format!("stat: {e}"))?;
        let file_stamp = FileStamp::of(&meta);

        let needs_reload = last_stamp != Some(file_stamp);

        if !needs_reload {
            return Ok(false);
        }

        let content = tokio::fs::read_to_string(&path)
            .await
            .map_err(|e| format!("read: {e}"))?;
        let doc: NotebookDocument =
            serde_json::from_str(&content).map_err(|e| format!("json: {e}"))?;

        let mut nb = self.nb.lock().await;
        nb.doc = Some(doc);
        nb.last_save_stamp = Some(file_stamp);
        Ok(true)
    }

    async fn dispatch(&self, method: &str, id: Option<Value>, params: &Value) -> RpcResponse {
        match method {
            "initialize" => RpcResponse::ok(
                id,
                json!({
                "protocolVersion": "2024-11-05", "capabilities": { "tools": {} },
                "serverInfo": { "name": "crush-notebook-kernel", "version": "0.1.0" } }),
            ),
            "ping" => RpcResponse::ok(id, json!({})),
            "tools/list" => self.tools_list(id),
            "tools/call" => self.tool_call(id, params).await,
            "resources/list" => RpcResponse::ok(id, json!({"resources":[]})),
            "prompts/list" => RpcResponse::ok(id, json!({"prompts":[]})),
            "notifications/initialized" => RpcResponse::ok(id, json!({})),
            _ => RpcResponse::err(id, -32601, format!("Unknown: {method}")),
        }
    }

    fn tools_list(&self, id: Option<Value>) -> RpcResponse {
        RpcResponse::ok(
            id,
            json!({"tools": [
                {"name":"notebook_open","description":"Load a .crush-nb file","inputSchema":{
                    "type":"object","properties":{"path":{"type":"string"}},"required":["path"]}},
                {"name":"notebook_eval_cell","description":"Evaluate a cell by index or ID","inputSchema":{
                    "type":"object","properties":{"cell_id":{"type":"string"}},"required":["cell_id"]}},
                {"name":"notebook_eval_all","description":"Execute all cells in order","inputSchema":{
                    "type":"object","properties":{}}},
                {"name":"notebook_list_vars","description":"List session variables","inputSchema":{
                    "type":"object","properties":{}}},
                {"name":"notebook_insert_cell","description":"Insert a new cell","inputSchema":{
                    "type":"object","properties":{"index":{"type":"number"},"source":{"type":"string"},"agent":{"type":"string"},
                    "kind":{"type":"string","enum":["crush","nepali","sona","markdown","python","javascript","ai_query","ai_agent_delegate"]},
                    "delegate_to":{"type":"string"},"task":{"type":"string"}},
                    "required":["source","kind"]}},
                {"name":"notebook_delete_cell","description":"Remove a cell by ID","inputSchema":{
                    "type":"object","properties":{"cell_id":{"type":"string"}},"required":["cell_id"]}},
                {"name":"notebook_get_state","description":"Full notebook state","inputSchema":{
                    "type":"object","properties":{}}},
                {"name":"notebook_save","description":"Save notebook to file (auto-saved on mutations)","inputSchema":{
                    "type":"object","properties":{}}},
                {"name":"notebook_reload","description":"Reload notebook from disk if external changes detected","inputSchema":{
                    "type":"object","properties":{}}},
                {"name":"notebook_claim_cell","description":"Claim a cell for an agent with @wip.started_by ownership","inputSchema":{
                    "type":"object","properties":{"cell_id":{"type":"string"},"agent":{"type":"string"},"intent":{"type":"string"}},
                    "required":["cell_id","agent"]}},
                {"name":"notebook_update_wip","description":"Update a cell's @wip checklist (owner only)","inputSchema":{
                    "type":"object","properties":{"cell_id":{"type":"string"},"agent":{"type":"string"},"intent":{"type":"string"},
                    "done":{"type":"array","items":{"type":"string"}},"todo":{"type":"array","items":{"type":"string"}},
                    "unresolved":{"type":"array","items":{"type":"string"}}},"required":["cell_id","agent"]}},
                {"name":"notebook_release_cell","description":"Release a cell's @wip claim (owner only)","inputSchema":{
                    "type":"object","properties":{"cell_id":{"type":"string"},"agent":{"type":"string"}},"required":["cell_id","agent"]}}
            ]}),
        )
    }

    async fn tool_call(&self, id: Option<Value>, params: &Value) -> RpcResponse {
        let name = params.get("name").and_then(Value::as_str).unwrap_or("");
        let args = params.get("arguments").cloned().unwrap_or(json!({}));
        match name {
            "notebook_open" => self.open(id, &args).await,
            "notebook_eval_cell" => self.eval_cell(id, &args).await,
            "notebook_eval_all" => self.eval_all(id).await,
            "notebook_list_vars" => self.list_vars(id).await,
            "notebook_insert_cell" => self.insert_cell(id, &args).await,
            "notebook_delete_cell" => self.delete_cell(id, &args).await,
            "notebook_get_state" => self.get_state(id).await,
            "notebook_save" => self.save_tool(id).await,
            "notebook_reload" => self.reload_tool(id).await,
            "notebook_claim_cell" => self.claim_cell(id, &args).await,
            "notebook_update_wip" => self.update_wip(id, &args).await,
            "notebook_release_cell" => self.release_cell(id, &args).await,
            _ => RpcResponse::err(id, -32602, format!("Unknown tool: {name}")),
        }
    }

    // ── Tools ────────────────────────────────────────────────────────

    async fn open(&self, id: Option<Value>, args: &Value) -> RpcResponse {
        let path = args.get("path").and_then(Value::as_str).unwrap_or("");
        if path.is_empty() {
            return RpcResponse::err(id, -32602, "Missing: path");
        }
        let content = match tokio::fs::read_to_string(path).await {
            Ok(c) => c,
            Err(e) => return RpcResponse::err(id, -32000, format!("Read: {e}")),
        };
        let doc: NotebookDocument = match serde_json::from_str(&content) {
            Ok(d) => d,
            Err(e) => return RpcResponse::err(id, -32000, format!("JSON: {e}")),
        };
        let title = doc.meta.title.clone();
        let count = doc.cells.len();
        {
            let mut nb = self.nb.lock().await;
            nb.doc = Some(doc);
            nb.file_path = Some(path.to_string());
            // Kernel state is scoped to one notebook: a fresh open starts a fresh session.
            *self.vars.lock().await = Vars::new();
            nb.last_save_stamp = tokio::fs::metadata(path)
                .await
                .ok()
                .map(|m| FileStamp::of(&m));
        }
        RpcResponse::ok(
            id,
            json!({"content":[{"type":"text",
            "text":format!("Loaded \"{title}\" with {count} cells")}],"isError":false}),
        )
    }

    async fn eval_cell(&self, id: Option<Value>, args: &Value) -> RpcResponse {
        let cell_id = args
            .get("cell_id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        if cell_id.is_empty() {
            return RpcResponse::err(id, -32602, "Missing: cell_id");
        }

        // Auto-reload to pick up external changes from other agents
        if let Err(e) = self.maybe_reload().await {
            // File delete/rename is non-fatal — proceed with in-memory state
            tracing::warn!("reload: {e}");
        }

        // Phase 1: snapshot source + mark running (notebook lock only)
        let run = {
            let mut nb = self.nb.lock().await;
            nb.run_count += 1;
            nb.run_count
        };
        let (source, cell_kind, idx) = {
            let mut nb = self.nb.lock().await;
            let doc = match &mut nb.doc {
                Some(d) => d,
                None => return RpcResponse::err(id, -32000, "No notebook loaded"),
            };
            let idx = find_cell_index(doc, &cell_id);
            if idx >= doc.cells.len() {
                return RpcResponse::err(id, -32000, format!("Not found: {cell_id}"));
            }
            doc.cells[idx].state = CellState::Running;
            (
                doc.cells[idx].source.clone(),
                doc.cells[idx].kind.clone(),
                idx,
            )
        };

        // Phase 2: evaluate (vars lock only — no notebook lock). Delegation
        // cells hand work to another agent instead of executing locally.
        let (new_state, outputs, stats) = match &cell_kind {
            crush_notebook_core::CellKind::AiAgentDelegate { agent, task } => {
                self.delegate_cell(&cell_id, agent, task, run).await
            }
            _ => self.vars.lock().await.eval(&source, &cell_kind, run),
        };

        // Phase 3: write results back (notebook lock only)
        let is_err = matches!(new_state, CellState::Error { .. });
        let outcome = if matches!(&new_state, CellState::AiPending) {
            "queued"
        } else {
            "done"
        };
        let out_count = outputs.len();
        {
            let mut nb = self.nb.lock().await;
            let doc = nb.doc.as_mut().unwrap();
            let c = &mut doc.cells[idx];
            c.state = new_state;
            c.outputs = outputs;
            c.execution = stats;
        }

        // Persist so other kernel instances sharing this file see the change
        if let Err(e) = self.save().await {
            return RpcResponse::err(id, -32000, format!("Save: {e}"));
        }

        let cell_id_out = {
            self.nb
                .lock()
                .await
                .doc
                .as_ref()
                .and_then(|d| d.cells.get(idx))
                .map(|c| c.id.clone())
                .unwrap_or(cell_id)
        };
        RpcResponse::ok(
            id,
            json!({"content":[{"type":"text",
            "text":format!("Cell '{cell_id_out}' (run #{run}): {outcome}\nOutputs: {out_count}")}],
            "isError":is_err}),
        )
    }

    async fn eval_all(&self, id: Option<Value>) -> RpcResponse {
        let count = {
            self.nb
                .lock()
                .await
                .doc
                .as_ref()
                .map(|d| d.cells.len())
                .unwrap_or(0)
        };
        if count == 0 {
            return RpcResponse::err(id, -32000, "No notebook loaded");
        }
        let mut results = Vec::new();
        for i in 0..count {
            let args = json!({"cell_id": i.to_string()});
            let resp = self.eval_cell(id.clone(), &args).await;
            // eval_cell always returns a JSON-RPC-level Ok envelope, even when the
            // cell itself failed to compile/run — the real per-cell success signal
            // is the nested "isError" flag in the content payload, not whether a
            // `result` is present at all. Checking `resp.result.is_some()` here was
            // unconditionally true for any existing cell, so this summary reported
            // "ok" for cells that had genuinely errored.
            let ok = resp
                .result
                .as_ref()
                .and_then(|r| r.get("isError"))
                .and_then(Value::as_bool)
                .map(|is_err| !is_err)
                .unwrap_or(false);
            results.push(format!("cell[{}]: {}", i, if ok { "ok" } else { "err" }));
        }
        RpcResponse::ok(
            id,
            json!({"content":[{"type":"text",
            "text":format!("Evaluated {} cells:\n{}", count, results.join("\n"))}],"isError":false}),
        )
    }

    async fn list_vars(&self, id: Option<Value>) -> RpcResponse {
        let vars = self.vars.lock().await;
        let lines: Vec<String> = vars
            .bindings
            .iter()
            .map(|(k, v)| format!("  {k}: {} = {}", v.type_name(), v.to_json()))
            .collect();
        RpcResponse::ok(
            id,
            json!({"content":[{"type":"text",
            "text":if lines.is_empty() { "No variables.".into() } else { format!("{} vars:\n{}", lines.len(), lines.join("\n")) }}],
            "isError":false}),
        )
    }

    async fn insert_cell(&self, id: Option<Value>, args: &Value) -> RpcResponse {
        let source = args
            .get("source")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        if source.is_empty() {
            return RpcResponse::err(id, -32602, "Missing: source");
        }
        let kind_str = args.get("kind").and_then(Value::as_str).unwrap_or("crush");
        let index = args
            .get("index")
            .and_then(Value::as_u64)
            .map(|v| v as usize);
        let kind = match kind_str {
            "crush" => crush_notebook_core::CellKind::Crush,
            "nepali" => crush_notebook_core::CellKind::Nepali,
            "sona" => crush_notebook_core::CellKind::Sona,
            "markdown" => crush_notebook_core::CellKind::Markdown,
            "python" => crush_notebook_core::CellKind::Python,
            "javascript" => crush_notebook_core::CellKind::JavaScript,
            "ai_query" => crush_notebook_core::CellKind::AiQuery {
                query: source.clone(),
            },
            "ai_agent_delegate" => {
                let agent = match required_agent(args, "delegate_to") {
                    Ok(agent) => agent,
                    Err(message) => return RpcResponse::err(id, -32602, message),
                };
                let task = args
                    .get("task")
                    .and_then(Value::as_str)
                    .filter(|task| !task.trim().is_empty())
                    .unwrap_or(&source)
                    .to_owned();
                crush_notebook_core::CellKind::AiAgentDelegate { agent, task }
            }
            _ => return RpcResponse::err(id, -32602, format!("Unknown kind: {kind_str}")),
        };
        let agent = args
            .get("agent")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|agent| !agent.is_empty())
            .map(str::to_owned);
        let meta = match agent {
            Some(agent) => CellMeta {
                wip: Some(WipAnnotation {
                    intent: format!("Work on {} cell", kind.label()),
                    started_by: Some(agent),
                    done: vec![],
                    todo: vec![],
                    unresolved: vec![],
                }),
                ..CellMeta::default()
            },
            None => CellMeta::default(),
        };
        let cell = Cell {
            id: format!("cell-{}", &uuid::Uuid::new_v4().to_string()[..8]),
            kind,
            source,
            state: CellState::Pending,
            meta,
            outputs: vec![],
            execution: None,
        };
        let mut nb = self.nb.lock().await;
        let doc = match &mut nb.doc {
            Some(d) => d,
            None => return RpcResponse::err(id, -32000, "No notebook loaded"),
        };
        match index {
            Some(i) if i < doc.cells.len() => doc.cells.insert(i, cell),
            _ => doc.cells.push(cell),
        }
        drop(nb);
        // Auto-save so other agents see the new cell
        if let Err(e) = self.save().await {
            return RpcResponse::err(id, -32000, format!("Save: {e}"));
        }
        RpcResponse::ok(
            id,
            json!({"content":[{"type":"text","text":"Cell inserted."}],"isError":false}),
        )
    }

    async fn delete_cell(&self, id: Option<Value>, args: &Value) -> RpcResponse {
        let cell_id = args
            .get("cell_id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        if cell_id.is_empty() {
            return RpcResponse::err(id, -32602, "Missing: cell_id");
        }
        let mut nb = self.nb.lock().await;
        let doc = match &mut nb.doc {
            Some(d) => d,
            None => return RpcResponse::err(id, -32000, "No notebook loaded"),
        };
        let idx = find_cell_index(doc, &cell_id);
        if idx >= doc.cells.len() {
            return RpcResponse::err(id, -32000, format!("Not found: {cell_id}"));
        }
        doc.cells.remove(idx);
        drop(nb);
        // Auto-save so other agents see the removal
        if let Err(e) = self.save().await {
            return RpcResponse::err(id, -32000, format!("Save: {e}"));
        }
        RpcResponse::ok(
            id,
            json!({"content":[{"type":"text","text":format!("Removed '{cell_id}'.")}],"isError":false}),
        )
    }

    /// Hand a delegation request to the external messaging command
    /// (`CRUSH_NOTEBOOK_SQUAD_MSG`, default `squad-msg`).
    async fn delegate_cell(
        &self,
        cell_id: &str,
        agent: &str,
        task: &str,
        run: usize,
    ) -> (CellState, Vec<CellOutput>, Option<ExecutionStats>) {
        let command =
            std::env::var("CRUSH_NOTEBOOK_SQUAD_MSG").unwrap_or_else(|_| "squad-msg".to_owned());
        let sender = std::env::var("AGENT_NAME")
            .or_else(|_| std::env::var("CRUSH_NOTEBOOK_AGENT"))
            .unwrap_or_else(|_| "crush-notebook".to_owned());
        let message = delegation_message(cell_id, &sender, task, run);
        let started = std::time::Instant::now();
        let result = tokio::process::Command::new(&command)
            .arg(format!("@{agent}"))
            .arg(&message)
            .env("AGENT_NAME", &sender)
            .output()
            .await;
        match result {
            Ok(output) if output.status.success() => {
                let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
                let response = if stdout.is_empty() {
                    format!("Delegation sent to '{agent}'.")
                } else {
                    format!("Delegation sent to '{agent}': {stdout}")
                };
                let cell_output = CellOutput {
                    id: format!("delegate-{run}"),
                    kind: OutputKind::Text { text: response },
                    data: json!({"target":agent,"task":task,"sender":sender}),
                    timestamp: None,
                };
                let stats = ExecutionStats {
                    steps: 1,
                    duration_ms: started.elapsed().as_millis() as u64,
                    tier: ExecutionTier::None,
                    frontend: "delegate".into(),
                    jit_compiled: false,
                };
                (CellState::AiPending, vec![cell_output], Some(stats))
            }
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
                error_cell(
                    run,
                    format!(
                        "delegation failed ({}): {}",
                        output.status,
                        if stderr.is_empty() {
                            "no error output"
                        } else {
                            &stderr
                        }
                    ),
                )
            }
            Err(error) => error_cell(run, format!("delegation command '{command}': {error}")),
        }
    }

    /// Claim a cell for an agent. Existing claims are exclusive until the
    /// current owner explicitly releases the WIP item.
    async fn claim_cell(&self, id: Option<Value>, args: &Value) -> RpcResponse {
        if let Err(e) = self.maybe_reload().await {
            return RpcResponse::err(id, -32000, format!("Reload before claim: {e}"));
        }
        let cell_id = match required_string(args, "cell_id") {
            Ok(value) => value,
            Err(message) => return RpcResponse::err(id, -32602, message),
        };
        let agent = match required_string(args, "agent") {
            Ok(value) => value,
            Err(message) => return RpcResponse::err(id, -32602, message),
        };
        let intent = args
            .get("intent")
            .and_then(Value::as_str)
            .unwrap_or("Work in progress")
            .to_owned();

        {
            let mut nb = self.nb.lock().await;
            let doc = match &mut nb.doc {
                Some(doc) => doc,
                None => return RpcResponse::err(id, -32000, "No notebook loaded"),
            };
            let index = find_cell_index(doc, &cell_id);
            if index >= doc.cells.len() {
                return RpcResponse::err(id, -32000, format!("Not found: {cell_id}"));
            }
            let cell = &mut doc.cells[index];
            if let Some(wip) = &cell.meta.wip {
                if let Some(owner) = &wip.started_by {
                    if owner != &agent {
                        return RpcResponse::err(
                            id,
                            -32009,
                            format!("Cell '{cell_id}' is already claimed by '{owner}'"),
                        );
                    }
                }
            }
            let wip = cell.meta.wip.get_or_insert_with(|| WipAnnotation {
                intent: intent.clone(),
                started_by: None,
                done: vec![],
                todo: vec![],
                unresolved: vec![],
            });
            wip.started_by = Some(agent.clone());
            if args.get("intent").is_some() {
                wip.intent = intent;
            }
        }
        if let Err(e) = self.save().await {
            return RpcResponse::err(id, -32000, format!("Save: {e}"));
        }
        RpcResponse::ok(
            id,
            json!({"content":[{"type":"text",
            "text":format!("Cell '{cell_id}' claimed by '{agent}'.")}],"isError":false}),
        )
    }

    /// Update WIP details while enforcing the claim owner.
    async fn update_wip(&self, id: Option<Value>, args: &Value) -> RpcResponse {
        if let Err(e) = self.maybe_reload().await {
            return RpcResponse::err(id, -32000, format!("Reload before WIP update: {e}"));
        }
        let cell_id = match required_string(args, "cell_id") {
            Ok(value) => value,
            Err(message) => return RpcResponse::err(id, -32602, message),
        };
        let agent = match required_string(args, "agent") {
            Ok(value) => value,
            Err(message) => return RpcResponse::err(id, -32602, message),
        };
        let done = match optional_string_array(args, "done") {
            Ok(value) => value,
            Err(message) => return RpcResponse::err(id, -32602, message),
        };
        let todo = match optional_string_array(args, "todo") {
            Ok(value) => value,
            Err(message) => return RpcResponse::err(id, -32602, message),
        };
        let unresolved = match optional_string_array(args, "unresolved") {
            Ok(value) => value,
            Err(message) => return RpcResponse::err(id, -32602, message),
        };
        let intent = args
            .get("intent")
            .and_then(Value::as_str)
            .map(str::to_owned);

        {
            let mut nb = self.nb.lock().await;
            let doc = match &mut nb.doc {
                Some(doc) => doc,
                None => return RpcResponse::err(id, -32000, "No notebook loaded"),
            };
            let index = find_cell_index(doc, &cell_id);
            if index >= doc.cells.len() {
                return RpcResponse::err(id, -32000, format!("Not found: {cell_id}"));
            }
            let wip = match doc.cells[index].meta.wip.as_mut() {
                Some(wip) => wip,
                None => {
                    return RpcResponse::err(
                        id,
                        -32000,
                        format!("Cell '{cell_id}' has no @wip claim"),
                    )
                }
            };
            match &wip.started_by {
                Some(owner) if owner == &agent => {}
                Some(owner) => {
                    return RpcResponse::err(
                        id,
                        -32009,
                        format!("Cell '{cell_id}' is owned by '{owner}'"),
                    )
                }
                None => {
                    return RpcResponse::err(
                        id,
                        -32009,
                        format!("Cell '{cell_id}' has no owner; claim it before updating"),
                    )
                }
            }
            if let Some(value) = done {
                wip.done = value;
            }
            if let Some(value) = todo {
                wip.todo = value;
            }
            if let Some(value) = unresolved {
                wip.unresolved = value;
            }
            if let Some(value) = intent {
                wip.intent = value;
            }
        }
        if let Err(e) = self.save().await {
            return RpcResponse::err(id, -32000, format!("Save: {e}"));
        }
        RpcResponse::ok(
            id,
            json!({"content":[{"type":"text",
            "text":format!("WIP for cell '{cell_id}' updated by '{agent}'.")}],"isError":false}),
        )
    }

    /// Release a cell claim without deleting its WIP history.
    async fn release_cell(&self, id: Option<Value>, args: &Value) -> RpcResponse {
        if let Err(e) = self.maybe_reload().await {
            return RpcResponse::err(id, -32000, format!("Reload before release: {e}"));
        }
        let cell_id = match required_string(args, "cell_id") {
            Ok(value) => value,
            Err(message) => return RpcResponse::err(id, -32602, message),
        };
        let agent = match required_string(args, "agent") {
            Ok(value) => value,
            Err(message) => return RpcResponse::err(id, -32602, message),
        };

        {
            let mut nb = self.nb.lock().await;
            let doc = match &mut nb.doc {
                Some(doc) => doc,
                None => return RpcResponse::err(id, -32000, "No notebook loaded"),
            };
            let index = find_cell_index(doc, &cell_id);
            if index >= doc.cells.len() {
                return RpcResponse::err(id, -32000, format!("Not found: {cell_id}"));
            }
            let wip = match doc.cells[index].meta.wip.as_mut() {
                Some(wip) => wip,
                None => {
                    return RpcResponse::err(
                        id,
                        -32000,
                        format!("Cell '{cell_id}' has no @wip claim"),
                    )
                }
            };
            match &wip.started_by {
                Some(owner) if owner == &agent => wip.started_by = None,
                Some(owner) => {
                    return RpcResponse::err(
                        id,
                        -32009,
                        format!("Cell '{cell_id}' is owned by '{owner}'"),
                    )
                }
                None => {
                    return RpcResponse::err(id, -32009, format!("Cell '{cell_id}' is not claimed"))
                }
            }
        }
        if let Err(e) = self.save().await {
            return RpcResponse::err(id, -32000, format!("Save: {e}"));
        }
        RpcResponse::ok(
            id,
            json!({"content":[{"type":"text",
            "text":format!("Cell '{cell_id}' released by '{agent}'.")}],"isError":false}),
        )
    }

    async fn save_tool(&self, id: Option<Value>) -> RpcResponse {
        match self.save().await {
            Ok(()) => RpcResponse::ok(
                id,
                json!({"content":[{"type":"text","text":"Notebook saved."}],"isError":false}),
            ),
            Err(e) => RpcResponse::err(id, -32000, format!("Save: {e}")),
        }
    }

    async fn reload_tool(&self, id: Option<Value>) -> RpcResponse {
        match self.maybe_reload().await {
            Ok(true) => RpcResponse::ok(
                id,
                json!({"content":[{"type":"text","text":"Reloaded from disk (external changes detected)."}],"isError":false}),
            ),
            Ok(false) => RpcResponse::ok(
                id,
                json!({"content":[{"type":"text","text":"No external changes detected."}],"isError":false}),
            ),
            Err(e) => RpcResponse::err(id, -32000, format!("Reload: {e}")),
        }
    }

    async fn get_state(&self, id: Option<Value>) -> RpcResponse {
        // Auto-reload to pick up external changes from other agents
        if let Err(e) = self.maybe_reload().await {
            tracing::warn!("reload: {e}");
        }

        let nb = self.nb.lock().await;
        let vars = self.vars.lock().await;
        let doc = match &nb.doc {
            Some(d) => d,
            None => return RpcResponse::err(id, -32000, "No notebook loaded"),
        };
        let cells: Vec<Value> = doc
            .cells
            .iter()
            .map(|c| {
                json!({
            "id":c.id, "state":serde_json::to_value(&c.state).unwrap_or(json!(null)),
            "source_preview":source_preview(&c.source, 80),
            "outputs_count":c.outputs.len() })
            })
            .collect();
        let vars_out: Vec<Value> = vars
            .bindings
            .iter()
            .map(|(k, v)| json!({"name":k,"value":v.to_json(),"type":v.type_name()}))
            .collect();
        RpcResponse::ok(
            id,
            json!({"content":[{"type":"text",
            "text":serde_json::to_string_pretty(&json!({"title":doc.meta.title,"cells":cells,
            "variables":vars_out,"run_count":nb.run_count})).unwrap_or_default()}],"isError":false}),
        )
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────

const SAVE_LOCK_TIMEOUT: Duration = Duration::from_secs(5);
const SAVE_LOCK_RETRY: Duration = Duration::from_millis(10);

struct SaveLock {
    path: PathBuf,
    // Keeping the file open documents ownership and prevents accidental reuse
    // while the guard is alive; exclusivity comes from create_new below.
    _file: tokio::fs::File,
}

impl Drop for SaveLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn save_lock_path(path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.lock", path.display()))
}

async fn acquire_save_lock(path: &Path, timeout: Duration) -> Result<SaveLock, String> {
    let lock_path = save_lock_path(path);
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        match tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&lock_path)
            .await
        {
            Ok(mut file) => {
                let owner = format!("pid={}\\n", std::process::id());
                if let Err(error) =
                    tokio::io::AsyncWriteExt::write_all(&mut file, owner.as_bytes()).await
                {
                    let _ = tokio::fs::remove_file(&lock_path).await;
                    return Err(format!("write lock {}: {error}", lock_path.display()));
                }
                return Ok(SaveLock {
                    path: lock_path,
                    _file: file,
                });
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                if tokio::time::Instant::now() >= deadline {
                    return Err(format!(
                        "timed out waiting for notebook lock {}",
                        lock_path.display()
                    ));
                }
                tokio::time::sleep(SAVE_LOCK_RETRY).await;
            }
            Err(error) => return Err(format!("create lock {}: {error}", lock_path.display())),
        }
    }
}

async fn atomic_write(path: &Path, contents: &[u8]) -> Result<(), String> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("invalid notebook filename: {}", path.display()))?;
    let temp_path = parent.join(format!(
        ".{filename}.tmp-{}-{}",
        std::process::id(),
        uuid::Uuid::new_v4()
    ));
    let result = async {
        let mut file = tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)
            .await
            .map_err(|e| format!("create temp file {}: {e}", temp_path.display()))?;
        file.write_all(contents)
            .await
            .map_err(|e| format!("write temp file {}: {e}", temp_path.display()))?;
        file.sync_all()
            .await
            .map_err(|e| format!("flush temp file {}: {e}", temp_path.display()))?;
        drop(file);
        tokio::fs::rename(&temp_path, path)
            .await
            .map_err(|e| format!("replace {}: {e}", path.display()))?;
        Ok::<(), String>(())
    }
    .await;
    if result.is_err() {
        let _ = tokio::fs::remove_file(&temp_path).await;
    }
    result
}

fn chrono_now() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let (y, m, d, h, min, s) = unix_to_ymdhms(secs);
    format!("{y:04}-{m:02}-{d:02}T{h:02}:{min:02}:{s:02}Z")
}

fn unix_to_ymdhms(ts: u64) -> (u64, u64, u64, u64, u64, u64) {
    let s = ts % 60;
    let m = (ts / 60) % 60;
    let h = (ts / 3600) % 24;
    let days = ts / 86400;
    let (y, mth, d) = days_to_ymd(days);
    (y, mth, d, h, m, s)
}

fn days_to_ymd(mut days: u64) -> (u64, u64, u64) {
    // Howard Hinnant's civil-from-days algorithm
    days += 719468;
    let era = days / 146097;
    let doe = days - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

fn source_preview(source: &str, max_chars: usize) -> &str {
    source
        .char_indices()
        .nth(max_chars)
        .map(|(index, _)| &source[..index])
        .unwrap_or(source)
}

fn delegation_message(cell_id: &str, sender: &str, task: &str, run: usize) -> String {
    format!("Notebook delegation cell={cell_id} run={run} from={sender}: {task}")
}

fn required_agent(args: &Value, key: &str) -> Result<String, String> {
    let agent = required_string(args, key)?;
    if agent
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-')
    {
        Ok(agent)
    } else {
        Err(format!("Invalid: {key} must be a bare agent name"))
    }
}

fn required_string(args: &Value, key: &str) -> Result<String, String> {
    match args
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        Some(value) => Ok(value.to_owned()),
        None => Err(format!("Missing: {key}")),
    }
}

fn optional_string_array(args: &Value, key: &str) -> Result<Option<Vec<String>>, String> {
    match args.get(key) {
        None => Ok(None),
        Some(value) => serde_json::from_value(value.clone())
            .map(Some)
            .map_err(|_| format!("Invalid: {key} must be an array of strings")),
    }
}

fn find_cell_index(doc: &NotebookDocument, cell_id: &str) -> usize {
    if let Ok(i) = cell_id.parse::<usize>() {
        if i < doc.cells.len() {
            return i;
        }
    }
    doc.cells
        .iter()
        .position(|c| c.id == cell_id)
        .unwrap_or(usize::MAX)
}

// ── Eval engine (MVP — real crush-frontend is Phase 2) ───────────────────

// ── Eval engine (crush-frontend + crush-vm — Phase 2 / M1) ──────────────

fn eval_crush_source(
    source: &str,
    run: usize,
    lang: &str,
) -> (CellState, Vec<CellOutput>, Option<ExecutionStats>) {
    let wrapped = if source.contains("fn main") {
        source.to_string()
    } else {
        format!("fn main() {{\n{}\n}}", source)
    };
    let t0 = std::time::Instant::now();

    match crush_frontend::compile_crush_source(&wrapped) {
        Ok(mut casm_program) => {
            casm_program.lang = Some(lang.to_string());
            match run_cvm1(&casm_program) {
                Ok(result) => {
                    let mut out = Vec::new();
                    if !result.output.is_empty() {
                        out.push(CellOutput {
                            id: format!("out-{run}"),
                            kind: OutputKind::Text {
                                text: result.output.clone(),
                            },
                            data: json!({"output": result.output}),
                            timestamp: None,
                        });
                    }
                    let stats = ExecutionStats {
                        steps: result.steps,
                        duration_ms: t0.elapsed().as_millis() as u64,
                        tier: ExecutionTier::Cvm1,
                        frontend: lang.into(),
                        jit_compiled: false,
                    };
                    (CellState::Done, out, Some(stats))
                }
                Err(e) => error_cell(run, e),
            }
        }
        Err(e) => error_cell(run, format!("Compile: {e}")),
    }
}

/// Lower a compiled program to CVM1 assembly and run it on the PortableVm.
fn run_cvm1(casm_program: &casm::Program) -> Result<crush_vm::VmResult, String> {
    // Convert casm::Program to crush_vm assembly text, then assemble + run via CVM1
    let assembly = casm_to_assembly(casm_program).map_err(|e| format!("Lower: {e}"))?;
    let program =
        crush_vm::assemble(&assembly, None, None).map_err(|e| format!("Assemble: {e}"))?;
    let mut vm = crush_vm::PortableVm::new(program);
    vm.set_quotas(crush_vm::Quotas {
        max_steps: 1_000_000,
        ..Default::default()
    });
    vm.run().map_err(|e| format!("VM error: {e}"))
}

#[cfg(feature = "jit")]
/// JIT-compiled cell evaluation via Cranelift (Phase 1 ops only).
/// Falls back to FastVM if the program uses unsupported opcodes
/// (capability calls, function calls, arrays/maps, exceptions).
fn eval_jit_source(
    source: &str,
    run: usize,
) -> (CellState, Vec<CellOutput>, Option<ExecutionStats>) {
    let wrapped = if source.contains("fn main") {
        source.to_string()
    } else {
        format!("fn main() {{\n{}\n}}", source)
    };
    let t0 = std::time::Instant::now();

    match crush_frontend::compile_crush_source(&wrapped) {
        Ok(casm_program) => {
            match crush_vm::fastvm::lower_program(&casm_program) {
                Ok(lowered) => {
                    let engine = crush_jit::JitEngine::default();
                    match engine.run(&lowered) {
                        Ok(result) => {
                            let out = fastvm_output(result, run);
                            let stats = ExecutionStats {
                                steps: 0,
                                duration_ms: t0.elapsed().as_millis() as u64,
                                tier: ExecutionTier::Jit,
                                frontend: "crush".into(),
                                jit_compiled: true,
                            };
                            (CellState::Done, out, Some(stats))
                        }
                        Err(e) => {
                            // JIT failed — fall back to FastVM interpretation
                            let msg = format!("JIT not supported (falling back to FastVM): {e}");
                            let warn = vec![CellOutput {
                                id: format!("warn-{run}"),
                                kind: OutputKind::Text { text: msg },
                                data: json!({}),
                                timestamp: None,
                            }];
                            match crush_vm::run_fastvm(&casm_program) {
                                Ok(result) => {
                                    let mut out = fastvm_output(result, run);
                                    let mut merged = warn;
                                    merged.append(&mut out);
                                    let stats = ExecutionStats {
                                        steps: 0,
                                        duration_ms: t0.elapsed().as_millis() as u64,
                                        tier: ExecutionTier::FastVM,
                                        frontend: "crush".into(),
                                        jit_compiled: false,
                                    };
                                    (CellState::Done, merged, Some(stats))
                                }
                                Err(e) => error_cell(run, format!("FastVM: {e}")),
                            }
                        }
                    }
                }
                Err(e) => error_cell(run, format!("Lower: {e}")),
            }
        }
        Err(e) => error_cell(run, format!("Compile: {e}")),
    }
}

fn eval_sim_source(
    source: &str,
    run: usize,
) -> (CellState, Vec<CellOutput>, Option<ExecutionStats>) {
    // Fallback: simple expression eval for languages without a compiler
    use std::fmt::Write;
    let mut output = String::new();
    for line in source.lines() {
        let t = line.trim().trim_end_matches(';').trim();
        if t.is_empty() || t.starts_with("//") || t.starts_with('#') {
            continue;
        }
        let _ = writeln!(output, "  {t}");
    }
    let stats = ExecutionStats {
        steps: source.lines().count(),
        duration_ms: 1,
        tier: ExecutionTier::None,
        frontend: "fallback".into(),
        jit_compiled: false,
    };
    let out = if output.is_empty() {
        vec![]
    } else {
        vec![CellOutput {
            id: format!("out-{run}"),
            kind: OutputKind::Text { text: output },
            data: json!({}),
            timestamp: None,
        }]
    };
    (CellState::Done, out, Some(stats))
}

/// Polyglot cell execution — wraps source in a `@<lang> { ... }` block,
/// compiles through crush-frontend → CASM, then runs via FastVM which
/// spawns the native interpreter (python3/node) as a subprocess via exec_lang.
fn eval_polyglot(
    source: &str,
    run: usize,
    lang: &str,
) -> (CellState, Vec<CellOutput>, Option<ExecutionStats>) {
    // Escape braces in source so they don't confuse crush's parser
    let escaped = source.replace('{', "\\{").replace('}', "\\}");
    // Wrap in @lang block: @python { ... } or @javascript { ... }
    let crush_source = format!("fn main() {{ @{lang} {{ {escaped} }} }}");
    let t0 = std::time::Instant::now();

    match crush_frontend::compile_crush_source(&crush_source) {
        Ok(casm_program) => match crush_vm::run_fastvm(&casm_program) {
            // FastVM doesn't spawn the interpreter itself: it yields an
            // exec_lang request for the host to service, and this kernel
            // doesn't service it yet. Say so instead of reporting a cell
            // that never ran as done.
            Ok(crush_vm::fastvm::FastYield::Request(crush_vm::fastvm::HostRequest::ExecLang { .. })) => {
                error_cell(run, format!(
                    "{lang} cells are not executed yet: the kernel does not run polyglot code (no exec_lang host)"))
            }
            Ok(result) => {
                let out = fastvm_output(result, run);
                let stats = ExecutionStats {
                    steps: 0,
                    duration_ms: t0.elapsed().as_millis() as u64,
                    tier: ExecutionTier::FastVM,
                    frontend: lang.into(),
                    jit_compiled: false,
                };
                (CellState::Done, out, Some(stats))
            }
            Err(e) => error_cell(run, format!("{lang}: FastVM: {e}")),
        },
        Err(e) => error_cell(run, format!("{lang}: compile: {e}")),
    }
}

fn error_cell(run: usize, msg: String) -> (CellState, Vec<CellOutput>, Option<ExecutionStats>) {
    let out = vec![CellOutput {
        id: format!("err-{run}"),
        kind: OutputKind::Error {
            message: msg.clone(),
            trace: None,
        },
        data: json!({}),
        timestamp: None,
    }];
    (CellState::Error { message: msg }, out, None)
}

/// Extract CellOutput from a FastVM/JIT `FastYield` result.
fn fastvm_output(result: crush_vm::fastvm::FastYield, run: usize) -> Vec<CellOutput> {
    match result {
        crush_vm::fastvm::FastYield::Finished(Some(val)) => {
            vec![CellOutput {
                id: format!("out-{run}"),
                kind: OutputKind::Text {
                    text: format!("{val:?}"),
                },
                data: json!({"value": format!("{val:?}")}),
                timestamp: None,
            }]
        }
        crush_vm::fastvm::FastYield::Value(val) => {
            vec![CellOutput {
                id: format!("out-{run}"),
                kind: OutputKind::Text {
                    text: format!("{val:?}"),
                },
                data: json!({"value": format!("{val:?}")}),
                timestamp: None,
            }]
        }
        _ => vec![],
    }
}

/// Lower a `casm::Program` to crush_vm assembly text.
///
/// Uses `Instruction::to_opcode()` for typed matching — a new opcode variant
/// in the `casm` crate produces a **compile-time** error here (non-exhaustive
/// match) rather than silently emitting wrong programs.
fn casm_to_assembly(casm_program: &casm::Program) -> Result<String, String> {
    use std::sync::atomic::{AtomicU32, Ordering};
    static LABEL_COUNTER: AtomicU32 = AtomicU32::new(0);
    fn ulabel() -> String {
        format!("L{}", LABEL_COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    let mut lines = Vec::new();
    let local_funcs: std::collections::HashSet<String> =
        casm_program.functions.keys().cloned().collect();

    for (fname, func) in &casm_program.functions {
        let mut slot_map = std::collections::HashMap::new();
        let mut next_slot: u16 = 0;
        let mut targets = std::collections::HashMap::<usize, String>::new();

        for instr in &func.body {
            if let Some(t) = instr.args.get("target").and_then(|v| v.as_u64()) {
                targets.entry(t as usize).or_insert_with(ulabel);
            }
        }

        lines.push(format!(".func {fname}"));
        for (i, instr) in func.body.iter().enumerate() {
            if let Some(l) = targets.get(&i) {
                lines.push(format!("{l}:"));
            }

            let opcode = instr
                .to_opcode()
                .map_err(|e| format!("bad opcode in fn {fname}: {e}"))?;

            let op = match opcode {
                // Stack
                casm::OpCode::PushInt(v) => format!("PUSH {v}"),
                casm::OpCode::PushFloat(v) => format!("PUSH_F64 {v}"),
                casm::OpCode::PushStr(ref s) => format!("PUSH_STR {s:?}"),
                casm::OpCode::PushBool(b) => format!("PUSH_BOOL {b}"),
                casm::OpCode::PushNull => "PUSH_NULL".into(),
                casm::OpCode::Pop => "POP".into(),
                casm::OpCode::Dup => "DUP".into(),
                casm::OpCode::Swap => "SWAP".into(),
                casm::OpCode::Rot => "ROT".into(),
                casm::OpCode::Pick(n) => format!("PICK {n}"),
                casm::OpCode::Roll(n) => format!("ROLL {n}"),

                // Memory
                casm::OpCode::Store(ref n) => {
                    let s = if let Some(&slot) = slot_map.get(n) {
                        slot
                    } else {
                        let s = next_slot;
                        slot_map.insert(n.clone(), s);
                        next_slot += 1;
                        s
                    };
                    format!("STORE {s}")
                }
                casm::OpCode::Load(ref n) => {
                    let s = if let Some(&slot) = slot_map.get(n) {
                        slot
                    } else {
                        let s = next_slot;
                        slot_map.insert(n.clone(), s);
                        next_slot += 1;
                        s
                    };
                    format!("LOAD {s}")
                }
                casm::OpCode::ExportVar(ref n) => format!("EXPORT_VAR {n}"),
                casm::OpCode::ImportVar(ref n) => format!("IMPORT_VAR {n}"),

                // Arithmetic
                casm::OpCode::Add => "ADD".into(),
                casm::OpCode::Sub => "SUB".into(),
                casm::OpCode::Mul => "MUL".into(),
                casm::OpCode::Div => "DIV".into(),
                casm::OpCode::Mod => "MOD".into(),
                casm::OpCode::Neg => "NEG".into(),

                // Comparison
                casm::OpCode::Eq => "EQ".into(),
                casm::OpCode::Ne => "NE".into(),
                casm::OpCode::Lt => "LT".into(),
                casm::OpCode::Gt => "GT".into(),
                casm::OpCode::Le => "LE".into(),
                casm::OpCode::Ge => "GE".into(),

                // Logical
                casm::OpCode::And => "AND".into(),
                casm::OpCode::Or => "OR".into(),
                casm::OpCode::Not => "NOT".into(),

                // Bitwise
                casm::OpCode::BitAnd => "BITAND".into(),
                casm::OpCode::BitOr => "BITOR".into(),
                casm::OpCode::BitXor => "BITXOR".into(),
                casm::OpCode::BitNot => "BITNOT".into(),
                casm::OpCode::Shl => "SHL".into(),
                casm::OpCode::Shr => "SHR".into(),

                // Control flow
                casm::OpCode::Jmp(t) => format!(
                    "JMP {}",
                    targets.get(&t).cloned().unwrap_or("UNKNOWN".into())
                ),
                casm::OpCode::JmpIf(t) => format!(
                    "JNZ {}",
                    targets.get(&t).cloned().unwrap_or("UNKNOWN".into())
                ),
                casm::OpCode::JmpIfNot(t) => format!(
                    "JZ {}",
                    targets.get(&t).cloned().unwrap_or("UNKNOWN".into())
                ),
                casm::OpCode::Call(ref n) => {
                    if local_funcs.contains(n) {
                        format!("CALL {n}")
                    } else {
                        format!("CAP_CALL {n:?} 0")
                    }
                }
                casm::OpCode::Ret => "RET".into(),
                casm::OpCode::Break => "BREAK".into(),
                casm::OpCode::Continue => "CONTINUE".into(),
                casm::OpCode::Spawn => "SPAWN".into(),
                casm::OpCode::Yield => "YIELD".into(),
                casm::OpCode::Await { ref handle } => format!("AWAIT {handle}"),
                casm::OpCode::EnterTry => "ENTER_TRY".into(),
                casm::OpCode::ExitTry => "EXIT_TRY".into(),
                casm::OpCode::Throw => "THROW".into(),

                // Arrays
                // crush-frontend emits `new_array <capacity>` followed by one
                // `array_push` per element; CVM1's `NEW_ARRAY n` instead pops n
                // elements, so start from an empty array.
                casm::OpCode::NewArray(_) => "NEW_ARRAY 0".to_string(),
                casm::OpCode::ArrGet => "ARR_GET".into(),
                casm::OpCode::ArrSet => "ARR_SET".into(),
                casm::OpCode::ArrLen => "ARR_LEN".into(),
                casm::OpCode::ArrPush => "ARR_PUSH".into(),
                casm::OpCode::ArrPop => "ARR_POP".into(),
                casm::OpCode::Index => "ARR_GET".into(),
                casm::OpCode::Len => "ARR_LEN".into(),
                casm::OpCode::ArrayPush => "ARR_PUSH".into(),
                casm::OpCode::ArrayPop => "ARR_POP".into(),
                casm::OpCode::MakeRange => "MAKE_RANGE".into(),

                // Collections
                casm::OpCode::NewTuple(sz) => format!("NEW_TUPLE {sz}"),
                casm::OpCode::TuplePush => "TUPLE_PUSH".into(),
                casm::OpCode::NewList(sz) => format!("NEW_LIST {sz}"),
                casm::OpCode::ListPush => "LIST_PUSH".into(),
                casm::OpCode::NewVector(sz) => format!("NEW_VECTOR {sz}"),
                casm::OpCode::VectorPush => "VECTOR_PUSH".into(),
                casm::OpCode::NewSet(sz) => format!("NEW_SET {sz}"),
                casm::OpCode::SetPush => "SET_PUSH".into(),

                // Objects
                casm::OpCode::NewObj => "NEW_OBJ".into(),
                casm::OpCode::NewStruct(ref n) => format!("NEW_STRUCT {n}"),
                casm::OpCode::GetField(ref n) => format!("GET_FIELD {n:?}"),
                casm::OpCode::SetField(ref n) => format!("SET_FIELD {n:?}"),

                // Types
                casm::OpCode::TypeOf => "TYPEOF".into(),
                casm::OpCode::Cast(ref t) => format!("CAST {t}"),

                // Capability / polyglot
                casm::OpCode::CapCall { ref name, argc } => format!("CAP_CALL {name:?} {argc}"),
                casm::OpCode::CallHost {
                    ref capsule,
                    ref method,
                    argc,
                    ..
                } => {
                    format!("CALL_HOST {capsule:?} {method:?} {argc}")
                }
                casm::OpCode::CallInterface {
                    ref handle,
                    ref method,
                    argc,
                } => {
                    format!("CALL_INTERFACE {handle} {method} {argc}")
                }
                casm::OpCode::ExecLang {
                    ref lang,
                    ref code,
                    var_count,
                } => {
                    format!("EXEC_LANG {lang:?} {code:?} {var_count}")
                }

                // String intrinsics
                casm::OpCode::StrContains => "STR_CONTAINS".into(),
                casm::OpCode::StrSplit => "STR_SPLIT".into(),
                casm::OpCode::StrReplace => "STR_REPLACE".into(),
                casm::OpCode::StrJoin => "STR_JOIN".into(),
                casm::OpCode::StrStartsWith => "STR_STARTS_WITH".into(),
                casm::OpCode::StrEndsWith => "STR_ENDS_WITH".into(),
                casm::OpCode::StrToUpper => "STR_TO_UPPER".into(),
                casm::OpCode::StrToLower => "STR_TO_LOWER".into(),
                casm::OpCode::StrTrim => "STR_TRIM".into(),

                // Math
                casm::OpCode::MathPow => "MATH_POW".into(),
                casm::OpCode::MathSqrt => "MATH_SQRT".into(),
                casm::OpCode::MathAbs => "MATH_ABS".into(),
                casm::OpCode::MathRound => "MATH_ROUND".into(),
                casm::OpCode::MathFloor => "MATH_FLOOR".into(),
                casm::OpCode::MathCeil => "MATH_CEIL".into(),

                // Program control
                casm::OpCode::Halt => "HALT".into(),

                // AI / DOM / catch-all: translate as NOP with a comment so
                // assembly still round-trips through the assembler, but these
                // opcodes aren't meaningful in the notebook's CVM1 path.
                casm::OpCode::AiQuery(_)
                | casm::OpCode::AiAdaptationRequest(_)
                | casm::OpCode::AiCapabilityDiscovery(_)
                | casm::OpCode::AiSemanticSwitch(_)
                | casm::OpCode::AiToolchain(_)
                | casm::OpCode::AiAgentDelegation(_)
                | casm::OpCode::AiLearningLoop(_)
                | casm::OpCode::AiContextAware(_)
                | casm::OpCode::AiSemanticMatch(_)
                | casm::OpCode::AiSynthesize(_)
                | casm::OpCode::AiGoalDeclaration(_)
                | casm::OpCode::AiProgressUpdate(_)
                | casm::OpCode::AiKnowledgeSharing(_)
                | casm::OpCode::DomQuery(_)
                | casm::OpCode::DomMutate(_)
                | casm::OpCode::DomEventListener(_) => "NOP  ; ai/dom opcode".to_string(),
            };
            lines.push(format!("    {op}"));
        }
    }
    Ok(lines.join("\n"))
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    // stdout carries the MCP JSON-RPC stream; logs must go to stderr.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(if args.verbose {
            "crush_notebook=debug"
        } else {
            "crush_notebook=info"
        })
        .init();
    tracing::info!("Crush-Notebook Kernel v0.1.0");
    run().await
}

async fn run() -> Result<()> {
    let kernel = Kernel::new();
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();
    let mut reader = BufReader::new(stdin);
    let mut writer = stdout;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).await? == 0 {
            return Ok(());
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let msg: RpcMessage = match serde_json::from_str(trimmed) {
            Ok(m) => m,
            Err(_) => {
                continue;
            }
        };
        let id = msg.id.clone();
        if id.is_none() {
            continue;
        }
        let resp = kernel.dispatch(&msg.method, id, &msg.params).await;
        let mut body = serde_json::to_string(&resp)?;
        body.push('\n');
        writer.write_all(body.as_bytes()).await?;
        writer.flush().await?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crush_notebook_core::CellKind;

    // ── Variables persist across cells ──────────────────────────────────

    fn binding<'a>(vars: &'a Vars, name: &str) -> Option<&'a SessionValue> {
        vars.bindings
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v)
    }

    #[tokio::test]
    async fn nb_041_hello_example_runs_top_to_bottom() {
        let example = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/hello.crush-nb");
        let nb_path =
            std::env::temp_dir().join(format!("test_hello-{}.crush-nb", uuid::Uuid::new_v4()));
        std::fs::copy(&example, &nb_path).expect(
            "tests/fixtures/hello.crush-nb (a symlink to examples/hello.crush-nb) should exist",
        );

        let kernel = Kernel::new();
        kernel
            .open(Some(json!(1)), &json!({"path": nb_path.to_str().unwrap()}))
            .await;
        let resp = kernel.eval_all(Some(json!(2))).await;
        let text = resp.result.unwrap()["content"][0]["text"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(
            !text.contains(": err"),
            "every hello cell should succeed: {text}"
        );

        let saved: Value =
            serde_json::from_str(&std::fs::read_to_string(&nb_path).unwrap()).unwrap();
        for cell in saved["cells"].as_array().unwrap() {
            assert_eq!(
                cell["state"]["status"], "done",
                "cell {} should be done: {}",
                cell["id"], cell["state"]
            );
        }

        let vars = kernel.vars.lock().await;
        assert_eq!(binding(&vars, "x"), Some(&SessionValue::Int(42)));
        assert_eq!(binding(&vars, "total"), Some(&SessionValue::Int(52)));
        assert_eq!(
            binding(&vars, "name"),
            Some(&SessionValue::Str("Crush Notebook".into()))
        );
        assert_eq!(binding(&vars, "multiplier"), Some(&SessionValue::Int(3)));
        assert_eq!(binding(&vars, "scaled"), Some(&SessionValue::Int(156)));
        drop(vars);

        let resp = kernel.list_vars(Some(json!(3))).await;
        let text = resp.result.unwrap()["content"][0]["text"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(
            text.contains("scaled: int = 156"),
            "list_vars should report session bindings: {text}"
        );
        let _ = std::fs::remove_file(&nb_path);
    }

    #[test]
    fn nb_041_variables_flow_between_cells_and_can_be_reassigned() {
        let mut vars = Vars::new();
        let (state, _, _) = vars.eval(
            "let x = 1\nlet xs = [1, 2, 3]\nlet m = {a: 1, b: \"z\"}",
            &CellKind::Crush,
            1,
        );
        assert_eq!(state, CellState::Done);
        let (state, _, _) = vars.eval("x = x + 10\nlet n = len(xs)", &CellKind::Crush, 2);
        assert_eq!(state, CellState::Done, "second cell should see x and xs");
        assert_eq!(binding(&vars, "x"), Some(&SessionValue::Int(11)));
        assert_eq!(binding(&vars, "n"), Some(&SessionValue::Int(3)));
        assert_eq!(
            binding(&vars, "xs"),
            Some(&SessionValue::Array(vec![
                SessionValue::Int(1),
                SessionValue::Int(2),
                SessionValue::Int(3)
            ]))
        );
        assert_eq!(
            binding(&vars, "m"),
            Some(&SessionValue::Map(vec![
                ("a".into(), SessionValue::Int(1)),
                ("b".into(), SessionValue::Str("z".into()))
            ]))
        );
        // Re-running a cell that redeclares a session variable shadows it.
        let (state, _, _) = vars.eval("let x = x * 2", &CellKind::Crush, 3);
        assert_eq!(state, CellState::Done);
        assert_eq!(binding(&vars, "x"), Some(&SessionValue::Int(22)));
    }

    #[test]
    fn polyglot_cells_report_that_they_did_not_run() {
        let mut vars = Vars::new();
        for (kind, lang) in [
            (CellKind::Python, "python"),
            (CellKind::JavaScript, "javascript"),
        ] {
            let (state, outputs, _) = vars.eval("1 + 2", &kind, 1);
            assert!(
                matches!(state, CellState::Error { ref message } if message.contains("not executed")),
                "{lang}: {state:?}"
            );
            assert_eq!(outputs.len(), 1);
        }
    }

    #[test]
    fn nb_041_undefined_variable_is_still_a_compile_error() {
        let mut vars = Vars::new();
        let (state, _, _) = vars.eval("let y = missing + 1", &CellKind::Crush, 1);
        assert!(
            matches!(state, CellState::Error { ref message } if message.contains("missing")),
            "{state:?}"
        );
        assert!(vars.bindings.is_empty());
    }

    #[test]
    fn nb_041_user_return_is_not_mistaken_for_captured_bindings() {
        let mut vars = Vars::new();
        vars.eval("let x = 5", &CellKind::Crush, 1);
        // A cell with its own return skips the capture: x keeps its old
        // value, and the returned array (same length as a capture) is ignored.
        let (state, _, _) = vars.eval("x = 6\nreturn [x]", &CellKind::Crush, 2);
        assert_eq!(state, CellState::Done);
        assert_eq!(binding(&vars, "x"), Some(&SessionValue::Int(5)));
        let (state, _, _) = vars.eval(
            "let y = x\nif y > 1 {\nreturn 0\n}\nlet z = 1",
            &CellKind::Crush,
            3,
        );
        assert_eq!(
            state,
            CellState::Done,
            "a nested return still sees session variables"
        );
        assert_eq!(binding(&vars, "y"), None);
    }

    #[test]
    fn nb_041_standalone_program_runs_without_the_session() {
        let mut vars = Vars::new();
        vars.eval("let x = 5", &CellKind::Crush, 1);
        let (state, _, _) = vars.eval("fn main() {\nlet x = 1\n}", &CellKind::Crush, 2);
        assert_eq!(state, CellState::Done);
        assert_eq!(binding(&vars, "x"), Some(&SessionValue::Int(5)));
    }

    #[test]
    fn nb_041_array_literals_lower_to_cvm1() {
        // crush-frontend emits `new_array <capacity>` + one `array_push` per
        // element; lowering it as CVM1 `NEW_ARRAY n` popped n stack slots.
        let program =
            crush_frontend::compile_crush_source("fn main() {\nreturn [\"a\", 2]\n}").unwrap();
        let result = run_cvm1(&program).expect("array literal should run on CVM1");
        let value = SessionValue::from_vm(result.stack.last().unwrap()).unwrap();
        assert_eq!(
            value,
            SessionValue::Array(vec![SessionValue::Str("a".into()), SessionValue::Int(2)])
        );
    }

    #[tokio::test]
    async fn nb_041_open_starts_a_fresh_session() {
        let nb_path = std::env::temp_dir().join(format!(
            "test_fresh_session-{}.crush-nb",
            uuid::Uuid::new_v4()
        ));
        std::fs::write(
            &nb_path,
            json!({"meta": {"title": "Fresh"}, "cells": []}).to_string(),
        )
        .unwrap();
        let kernel = Kernel::new();
        kernel
            .vars
            .lock()
            .await
            .set("stale".into(), SessionValue::Int(1));
        kernel
            .open(Some(json!(1)), &json!({"path": nb_path.to_str().unwrap()}))
            .await;
        assert!(kernel.vars.lock().await.bindings.is_empty());
        let _ = std::fs::remove_file(&nb_path);
    }

    // ── Unit: casm_to_assembly ──────────────────────────────────────────

    #[test]
    fn casm_to_assembly_valid_program() {
        let program = casm::Program {
            version: "1.0".into(),
            functions: [(
                "main".into(),
                casm::Function {
                    params: vec![],
                    locals: vec![],
                    type_hints: None,
                    body: vec![
                        casm::Instruction {
                            op: "push_int".into(),
                            lang: None,
                            meta: None,
                            args: serde_json::json!({"value": 42}),
                        },
                        casm::Instruction {
                            op: "halt".into(),
                            lang: None,
                            meta: None,
                            args: serde_json::json!({}),
                        },
                    ],
                },
            )]
            .into_iter()
            .collect(),
            manifest: casm::Manifest::default(),
            lang: Some("crush".into()),
        };

        let asm = casm_to_assembly(&program).expect("valid program should lower");
        assert!(
            asm.contains(".func main"),
            "asm should contain .func main: {asm}"
        );
        assert!(asm.contains("PUSH 42"), "asm should contain PUSH 42: {asm}");
        assert!(asm.contains("HALT"), "asm should contain HALT: {asm}");
    }

    #[test]
    fn casm_to_assembly_unknown_opcode_is_error() {
        let program = casm::Program {
            version: "1.0".into(),
            functions: [(
                "main".into(),
                casm::Function {
                    params: vec![],
                    locals: vec![],
                    type_hints: None,
                    body: vec![
                        casm::Instruction {
                            op: "push_int".into(),
                            lang: None,
                            meta: None,
                            args: serde_json::json!({"value": 1}),
                        },
                        casm::Instruction {
                            op: "bogus_opcode".into(),
                            lang: None,
                            meta: None,
                            args: serde_json::json!({}),
                        },
                    ],
                },
            )]
            .into_iter()
            .collect(),
            manifest: casm::Manifest::default(),
            lang: None,
        };

        let err = casm_to_assembly(&program).expect_err("bogus opcode should error");
        assert!(
            err.contains("bad opcode") || err.contains("UnknownOpcode"),
            "error should mention bad/unknown opcode: {err}"
        );
    }

    #[test]
    fn casm_to_assembly_jmp_targets() {
        let program = casm::Program {
            version: "1.0".into(),
            functions: [(
                "main".into(),
                casm::Function {
                    params: vec![],
                    locals: vec![],
                    type_hints: None,
                    body: vec![
                        casm::Instruction {
                            op: "push_int".into(),
                            lang: None,
                            meta: None,
                            args: serde_json::json!({"value": 0}),
                        },
                        casm::Instruction {
                            op: "store".into(),
                            lang: None,
                            meta: None,
                            args: serde_json::json!({"name": "i"}),
                        },
                        casm::Instruction {
                            op: "load".into(),
                            lang: None,
                            meta: None,
                            args: serde_json::json!({"name": "i"}),
                        },
                        casm::Instruction {
                            op: "push_int".into(),
                            lang: None,
                            meta: None,
                            args: serde_json::json!({"value": 10}),
                        },
                        casm::Instruction {
                            op: "lt".into(),
                            lang: None,
                            meta: None,
                            args: serde_json::json!({}),
                        },
                        casm::Instruction {
                            op: "jmp_if_not".into(),
                            lang: None,
                            meta: None,
                            args: serde_json::json!({"target": 8}),
                        },
                        casm::Instruction {
                            op: "load".into(),
                            lang: None,
                            meta: None,
                            args: serde_json::json!({"name": "i"}),
                        },
                        casm::Instruction {
                            op: "push_int".into(),
                            lang: None,
                            meta: None,
                            args: serde_json::json!({"value": 1}),
                        },
                        casm::Instruction {
                            op: "add".into(),
                            lang: None,
                            meta: None,
                            args: serde_json::json!({}),
                        },
                        casm::Instruction {
                            op: "store".into(),
                            lang: None,
                            meta: None,
                            args: serde_json::json!({"name": "i"}),
                        },
                        casm::Instruction {
                            op: "jmp".into(),
                            lang: None,
                            meta: None,
                            args: serde_json::json!({"target": 2}),
                        },
                        casm::Instruction {
                            op: "halt".into(),
                            lang: None,
                            meta: None,
                            args: serde_json::json!({}),
                        },
                    ],
                },
            )]
            .into_iter()
            .collect(),
            manifest: casm::Manifest::default(),
            lang: None,
        };

        let asm = casm_to_assembly(&program).expect("jmp program should lower");
        // Labels should be generated for jump targets
        assert!(asm.contains("JMP"), "asm should contain JMP: {asm}");
        assert!(
            asm.contains("JZ"),
            "asm should contain JZ for jmp_if_not: {asm}"
        );
    }

    #[test]
    fn casm_to_assembly_load_store_maps_slots() {
        let program = casm::Program {
            version: "1.0".into(),
            functions: [(
                "main".into(),
                casm::Function {
                    params: vec![],
                    locals: vec![],
                    type_hints: None,
                    body: vec![
                        casm::Instruction {
                            op: "push_int".into(),
                            lang: None,
                            meta: None,
                            args: serde_json::json!({"value": 5}),
                        },
                        casm::Instruction {
                            op: "store".into(),
                            lang: None,
                            meta: None,
                            args: serde_json::json!({"name": "x"}),
                        },
                        casm::Instruction {
                            op: "load".into(),
                            lang: None,
                            meta: None,
                            args: serde_json::json!({"name": "x"}),
                        },
                        casm::Instruction {
                            op: "halt".into(),
                            lang: None,
                            meta: None,
                            args: serde_json::json!({}),
                        },
                    ],
                },
            )]
            .into_iter()
            .collect(),
            manifest: casm::Manifest::default(),
            lang: None,
        };

        let asm = casm_to_assembly(&program).expect("load/store program should lower");
        assert!(
            asm.contains("STORE 0"),
            "first store should be slot 0: {asm}"
        );
        assert!(asm.contains("LOAD 0"), "first load should be slot 0: {asm}");
    }

    // ── Unit: fastvm_output ──────────────────────────────────────────────

    #[test]
    fn fastvm_output_finished_with_value() {
        let result = crush_vm::fastvm::FastYield::Finished(Some(crush_vm::RuntimeValue::Int(42)));
        let outputs = fastvm_output(result, 1);
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].id.contains("out-1"));
    }

    #[test]
    fn fastvm_output_finished_null() {
        let result = crush_vm::fastvm::FastYield::Finished(None);
        let outputs = fastvm_output(result, 2);
        assert!(outputs.is_empty(), "Finished(None) produces no output");
    }

    #[test]
    fn fastvm_output_yielded_is_empty() {
        let result = crush_vm::fastvm::FastYield::Yielded;
        let outputs = fastvm_output(result, 3);
        assert!(outputs.is_empty(), "Yielded produces no output");
    }

    #[test]
    fn fastvm_output_budget_exhausted_is_empty() {
        let result = crush_vm::fastvm::FastYield::BudgetExhausted;
        let outputs = fastvm_output(result, 4);
        assert!(outputs.is_empty(), "BudgetExhausted produces no output");
    }

    // ── Unit: error_cell ─────────────────────────────────────────────────

    #[test]
    fn error_cell_state_is_error() {
        let (state, outputs, stats) = error_cell(1, "boom".into());
        assert!(matches!(state, CellState::Error { .. }));
        if let CellState::Error { message } = state {
            assert_eq!(message, "boom");
        }
        assert_eq!(outputs.len(), 1);
        assert!(matches!(outputs[0].kind, OutputKind::Error { .. }));
        assert!(stats.is_none(), "error cells have no execution stats");
    }

    // ── Unit: eval_sim_source ────────────────────────────────────────────

    #[test]
    fn eval_sim_source_basic() {
        let (state, outputs, stats_opt) = eval_sim_source("let x = 1\nlet y = 2", 1);
        assert_eq!(state, CellState::Done);
        assert_eq!(outputs.len(), 1, "both lines are echoed as one text output");
        let stats = stats_opt.expect("sim eval produces stats");
        assert_eq!(stats.tier, ExecutionTier::None);
    }

    #[test]
    fn eval_sim_source_empty() {
        let (state, outputs, stats_opt) = eval_sim_source("", 1);
        assert_eq!(state, CellState::Done);
        assert!(outputs.is_empty());
        assert!(stats_opt.is_some());
    }

    // ── Unit: find_cell_index ────────────────────────────────────────────

    #[test]
    fn find_cell_index_by_position() {
        let doc = notebook_with_cells(&["cell-a", "cell-b", "cell-c"]);
        assert_eq!(find_cell_index(&doc, "0"), 0);
        assert_eq!(find_cell_index(&doc, "1"), 1);
        assert_eq!(find_cell_index(&doc, "2"), 2);
    }

    #[test]
    fn find_cell_index_by_id() {
        let doc = notebook_with_cells(&["cell-a", "cell-b", "cell-c"]);
        assert_eq!(find_cell_index(&doc, "cell-a"), 0);
        assert_eq!(find_cell_index(&doc, "cell-c"), 2);
    }

    #[test]
    fn find_cell_index_not_found() {
        let doc = notebook_with_cells(&["cell-a"]);
        assert_eq!(find_cell_index(&doc, "zzz"), usize::MAX);
        assert_eq!(find_cell_index(&doc, "99"), usize::MAX);
    }

    // ── Async: kernel tool dispatch ──────────────────────────────────────

    #[tokio::test]
    async fn kernel_dispatch_initialize() {
        let kernel = Kernel::new();
        let resp = kernel
            .dispatch(
                "initialize",
                Some(serde_json::json!(1)),
                &serde_json::json!({}),
            )
            .await;
        assert!(resp.result.is_some(), "init should succeed");
        let result = resp.result.unwrap();
        assert_eq!(result["serverInfo"]["name"], "crush-notebook-kernel");
    }

    #[tokio::test]
    async fn kernel_dispatch_ping() {
        let kernel = Kernel::new();
        let resp = kernel
            .dispatch("ping", Some(serde_json::json!(1)), &serde_json::json!({}))
            .await;
        assert!(resp.result.is_some());
    }

    #[tokio::test]
    async fn kernel_tools_list() {
        let kernel = Kernel::new();
        let resp = kernel
            .dispatch(
                "tools/list",
                Some(serde_json::json!(1)),
                &serde_json::json!({}),
            )
            .await;
        let tools = resp.result.unwrap()["tools"].as_array().unwrap().clone();
        assert_eq!(tools.len(), 12, "should register 12 tools");
        let names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
        assert!(names.contains(&"notebook_open"));
        assert!(names.contains(&"notebook_eval_cell"));
        assert!(names.contains(&"notebook_eval_all"));
        assert!(names.contains(&"notebook_insert_cell"));
        assert!(names.contains(&"notebook_delete_cell"));
        assert!(names.contains(&"notebook_get_state"));
        assert!(names.contains(&"notebook_list_vars"));
        assert!(names.contains(&"notebook_save"));
        assert!(names.contains(&"notebook_reload"));
        assert!(names.contains(&"notebook_claim_cell"));
        assert!(names.contains(&"notebook_update_wip"));
        assert!(names.contains(&"notebook_release_cell"));
    }

    #[tokio::test]
    async fn kernel_no_notebook_errors_on_ops() {
        let kernel = Kernel::new();

        // eval_cell with no notebook loaded
        let resp = kernel
            .eval_cell(
                Some(serde_json::json!(1)),
                &serde_json::json!({"cell_id":"0"}),
            )
            .await;
        assert!(resp.error.is_some());

        // insert_cell
        let resp = kernel
            .insert_cell(
                Some(serde_json::json!(2)),
                &serde_json::json!({"source":"let x=1","kind":"crush"}),
            )
            .await;
        assert!(resp.error.is_some());

        // delete_cell
        let resp = kernel
            .delete_cell(
                Some(serde_json::json!(3)),
                &serde_json::json!({"cell_id":"0"}),
            )
            .await;
        assert!(resp.error.is_some());

        // get_state
        let resp = kernel.get_state(Some(serde_json::json!(4))).await;
        assert!(resp.error.is_some());
    }

    #[tokio::test]
    async fn kernel_open_missing_file() {
        let kernel = Kernel::new();
        let resp = kernel
            .open(
                Some(serde_json::json!(1)),
                &serde_json::json!({"path":"/tmp/no/such/notebook.crush-nb"}),
            )
            .await;
        assert!(resp.error.is_some(), "missing file should error");
    }

    #[tokio::test]
    async fn kernel_open_valid_notebook_then_get_state() {
        let kernel = Kernel::new();
        let nb_path = std::env::temp_dir().join("test_kernel.crush-nb");
        let doc = serde_json::json!({
            "meta": {"title": "Test Notebook"},
            "cells": [
                {"id":"cell-a","kind":{"type":"crush"},"source":"let x = 1","state":{"status":"pending"},"meta":{},"outputs":[]},
                {"id":"cell-b","kind":{"type":"markdown"},"source":"# Hello","state":{"status":"done"},"meta":{},"outputs":[]}
            ]
        });
        std::fs::write(&nb_path, serde_json::to_string(&doc).unwrap()).unwrap();

        let resp = kernel
            .open(
                Some(serde_json::json!(1)),
                &serde_json::json!({"path": nb_path.to_str().unwrap()}),
            )
            .await;
        assert!(
            resp.result.is_some(),
            "open should succeed: {:?}",
            resp.error
        );

        // get_state should work
        let resp = kernel.get_state(Some(serde_json::json!(2))).await;
        assert!(
            resp.result.is_some(),
            "get_state after open: {:?}",
            resp.error
        );

        let _ = std::fs::remove_file(&nb_path);
    }

    #[tokio::test]
    async fn kernel_insert_and_delete_cell() {
        let kernel = Kernel::new();
        let nb_path = std::env::temp_dir().join("test_insert.crush-nb");
        let doc = serde_json::json!({
            "meta": {"title": "Test"},
            "cells": []
        });
        std::fs::write(&nb_path, serde_json::to_string(&doc).unwrap()).unwrap();

        kernel
            .open(
                Some(serde_json::json!(1)),
                &serde_json::json!({"path": nb_path.to_str().unwrap()}),
            )
            .await;

        // Insert a cell
        let resp = kernel
            .insert_cell(
                Some(serde_json::json!(2)),
                &serde_json::json!({"source":"let x=1","kind":"crush"}),
            )
            .await;
        assert!(
            resp.result.is_some(),
            "insert should succeed: {:?}",
            resp.error
        );

        // Verify via get_state
        let resp = kernel.get_state(Some(serde_json::json!(3))).await;
        let text = resp.result.unwrap()["content"][0]["text"]
            .as_str()
            .unwrap()
            .to_string();
        let state: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(state["cells"].as_array().unwrap().len(), 1);

        // Delete that cell by index
        let resp = kernel
            .delete_cell(
                Some(serde_json::json!(4)),
                &serde_json::json!({"cell_id":"0"}),
            )
            .await;
        assert!(resp.result.is_some());

        // Verify empty
        let resp = kernel.get_state(Some(serde_json::json!(5))).await;
        let text = resp.result.unwrap()["content"][0]["text"]
            .as_str()
            .unwrap()
            .to_string();
        let state: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert!(state["cells"].as_array().unwrap().is_empty());

        let _ = std::fs::remove_file(&nb_path);
    }

    #[tokio::test]
    async fn kernel_eval_cell_markdown_returns_done() {
        let kernel = Kernel::new();
        let nb_path = std::env::temp_dir().join("test_eval_md.crush-nb");
        let doc = serde_json::json!({
            "meta": {"title": "Markdown Test"},
            "cells": [{
                "id":"cell-md",
                "kind":{"type":"markdown"},
                "source":"# Hello",
                "state":{"status":"pending"},
                "meta":{},
                "outputs":[]
            }]
        });
        std::fs::write(&nb_path, serde_json::to_string(&doc).unwrap()).unwrap();

        kernel
            .open(
                Some(serde_json::json!(1)),
                &serde_json::json!({"path": nb_path.to_str().unwrap()}),
            )
            .await;

        // Eval the markdown cell by id
        let resp = kernel
            .eval_cell(
                Some(serde_json::json!(2)),
                &serde_json::json!({"cell_id":"cell-md"}),
            )
            .await;
        assert!(resp.result.is_some(), "eval markdown: {:?}", resp.error);
        assert_eq!(resp.error, None);

        let _ = std::fs::remove_file(&nb_path);
    }

    #[tokio::test]
    async fn kernel_unknown_method_returns_method_not_found() {
        let kernel = Kernel::new();
        let resp = kernel
            .dispatch("bogus", Some(serde_json::json!(1)), &serde_json::json!({}))
            .await;
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, -32601);
    }

    #[tokio::test]
    async fn kernel_unknown_tool_returns_error() {
        let kernel = Kernel::new();
        let resp = kernel
            .tool_call(
                Some(serde_json::json!(1)),
                &serde_json::json!({"name": "bogus_tool", "arguments":{}}),
            )
            .await;
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, -32602);
    }

    #[tokio::test]
    async fn kernel_eval_all_on_empty_notebook() {
        let kernel = Kernel::new();
        let nb_path = std::env::temp_dir().join("test_eval_all.crush-nb");
        let doc = serde_json::json!({
            "meta": {"title": "Empty"},
            "cells": []
        });
        std::fs::write(&nb_path, serde_json::to_string(&doc).unwrap()).unwrap();

        kernel
            .open(
                Some(serde_json::json!(1)),
                &serde_json::json!({"path": nb_path.to_str().unwrap()}),
            )
            .await;

        let resp = kernel.eval_all(Some(serde_json::json!(2))).await;
        assert!(resp.error.is_some(), "empty eval_all should error");

        let _ = std::fs::remove_file(&nb_path);
    }

    #[tokio::test]
    async fn kernel_list_vars_empty() {
        let kernel = Kernel::new();
        let resp = kernel.list_vars(Some(serde_json::json!(1))).await;
        assert!(resp.result.is_some());
        let text = resp.result.unwrap()["content"][0]["text"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(text.contains("No variables"));
    }

    #[tokio::test]
    async fn nb_021_claim_cell_records_owner_and_rejects_competing_agent() {
        let kernel = Kernel::new();
        let nb_path = std::env::temp_dir().join("test_wip_claim.crush-nb");
        let doc = serde_json::json!({
            "meta": {"title": "WIP Claim"},
            "cells": [{"id":"cell-wip","kind":{"type":"crush"},"source":"let x=1","state":{"status":"pending"},"meta":{},"outputs":[]}]
        });
        std::fs::write(&nb_path, serde_json::to_string(&doc).unwrap()).unwrap();
        kernel
            .open(
                Some(serde_json::json!(1)),
                &serde_json::json!({"path": nb_path.to_str().unwrap()}),
            )
            .await;

        let response = kernel
            .claim_cell(
                Some(serde_json::json!(2)),
                &serde_json::json!({
                    "cell_id":"cell-wip", "agent":"agent-a", "intent":"Finish the example"
                }),
            )
            .await;
        assert!(
            response.result.is_some(),
            "claim should succeed: {:?}",
            response.error
        );

        let response = kernel
            .claim_cell(
                Some(serde_json::json!(3)),
                &serde_json::json!({
                    "cell_id":"cell-wip", "agent":"agent-b"
                }),
            )
            .await;
        assert_eq!(
            response.error.as_ref().map(|error| error.code),
            Some(-32009)
        );
        assert!(response.error.unwrap().message.contains("agent-a"));

        let saved: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&nb_path).unwrap()).unwrap();
        assert_eq!(saved["cells"][0]["meta"]["wip"]["started_by"], "agent-a");
        assert_eq!(
            saved["cells"][0]["meta"]["wip"]["intent"],
            "Finish the example"
        );
        let _ = std::fs::remove_file(&nb_path);
    }

    #[tokio::test]
    async fn nb_021_update_wip_requires_owner_and_updates_checklists() {
        let kernel = Kernel::new();
        let nb_path = std::env::temp_dir().join("test_wip_update.crush-nb");
        let doc = serde_json::json!({
            "meta": {"title": "WIP Update"},
            "cells": [{"id":"cell-wip","kind":{"type":"crush"},"source":"let x=1","state":{"status":"pending"},"meta":{},"outputs":[]}]
        });
        std::fs::write(&nb_path, serde_json::to_string(&doc).unwrap()).unwrap();
        kernel
            .open(
                Some(serde_json::json!(1)),
                &serde_json::json!({"path": nb_path.to_str().unwrap()}),
            )
            .await;
        kernel
            .claim_cell(
                Some(serde_json::json!(2)),
                &serde_json::json!({
                    "cell_id":"cell-wip", "agent":"agent-a"
                }),
            )
            .await;

        let response = kernel
            .update_wip(
                Some(serde_json::json!(3)),
                &serde_json::json!({
                    "cell_id":"cell-wip", "agent":"agent-b", "todo":["review output"]
                }),
            )
            .await;
        assert_eq!(
            response.error.as_ref().map(|error| error.code),
            Some(-32009)
        );

        let response = kernel.update_wip(Some(serde_json::json!(4)), &serde_json::json!({
            "cell_id":"cell-wip", "agent":"agent-a", "done":["compile"],
            "todo":["review output"], "unresolved":["needs fixture"], "intent":"Finish safely"
        })).await;
        assert!(
            response.result.is_some(),
            "owner update should succeed: {:?}",
            response.error
        );

        let saved: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&nb_path).unwrap()).unwrap();
        let wip = &saved["cells"][0]["meta"]["wip"];
        assert_eq!(wip["started_by"], "agent-a");
        assert_eq!(wip["done"][0], "compile");
        assert_eq!(wip["todo"][0], "review output");
        assert_eq!(wip["unresolved"][0], "needs fixture");
        assert_eq!(wip["intent"], "Finish safely");

        let response = kernel
            .release_cell(
                Some(serde_json::json!(5)),
                &serde_json::json!({
                    "cell_id":"cell-wip", "agent":"agent-b"
                }),
            )
            .await;
        assert_eq!(
            response.error.as_ref().map(|error| error.code),
            Some(-32009)
        );
        let response = kernel
            .release_cell(
                Some(serde_json::json!(6)),
                &serde_json::json!({
                    "cell_id":"cell-wip", "agent":"agent-a"
                }),
            )
            .await;
        assert!(
            response.result.is_some(),
            "owner release should succeed: {:?}",
            response.error
        );
        let saved: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&nb_path).unwrap()).unwrap();
        assert!(!saved["cells"][0]["meta"]["wip"]
            .as_object()
            .unwrap()
            .contains_key("started_by"));
        let _ = std::fs::remove_file(&nb_path);
    }

    #[test]
    fn source_preview_respects_character_boundaries() {
        let source = "é".repeat(81);
        let preview = source_preview(&source, 80);
        assert_eq!(preview.chars().count(), 80);
        assert!(preview.is_char_boundary(preview.len()));
    }

    #[tokio::test]
    async fn nb_021_agent_insert_adds_wip_owner() {
        let kernel = Kernel::new();
        let nb_path = std::env::temp_dir().join("test_wip_insert.crush-nb");
        std::fs::write(
            &nb_path,
            serde_json::json!({
                "meta": {"title": "WIP Insert"}, "cells": []
            })
            .to_string(),
        )
        .unwrap();
        kernel
            .open(
                Some(serde_json::json!(1)),
                &serde_json::json!({"path": nb_path.to_str().unwrap()}),
            )
            .await;

        let response = kernel
            .insert_cell(
                Some(serde_json::json!(2)),
                &serde_json::json!({
                    "source":"let x = 1", "kind":"crush", "agent":"agent-a"
                }),
            )
            .await;
        assert!(
            response.result.is_some(),
            "agent insert should succeed: {:?}",
            response.error
        );
        let saved: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&nb_path).unwrap()).unwrap();
        assert_eq!(saved["cells"][0]["meta"]["wip"]["started_by"], "agent-a");
        let _ = std::fs::remove_file(&nb_path);
    }

    #[test]
    fn delegation_message_contains_routing_context() {
        let message = delegation_message("cell-5", "agent-a", "Review output: $HOME", 3);
        assert_eq!(
            message,
            "Notebook delegation cell=cell-5 run=3 from=agent-a: Review output: $HOME"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn nb_022_delegate_cell_sends_target_and_task_to_messenger() {
        use std::os::unix::fs::PermissionsExt;

        let kernel = Kernel::new();
        let suffix = uuid::Uuid::new_v4().to_string();
        let command_path = std::env::temp_dir().join(format!("test_delegate_command-{suffix}"));
        let args_path = std::env::temp_dir().join(format!("test_delegate_args-{suffix}"));
        let script = format!(
            "#!/bin/sh\nprintf '%s\\n%s\\n' \"$1\" \"$2\" > '{}'\n",
            args_path.display()
        );
        std::fs::write(&command_path, script).unwrap();
        std::fs::set_permissions(&command_path, std::fs::Permissions::from_mode(0o755)).unwrap();

        let previous = std::env::var_os("CRUSH_NOTEBOOK_SQUAD_MSG");
        std::env::set_var("CRUSH_NOTEBOOK_SQUAD_MSG", &command_path);
        // The sender comes from AGENT_NAME when set (as it is in any agent shell).
        let sender = std::env::var("AGENT_NAME").unwrap_or_else(|_| "crush-notebook".to_owned());
        let result = kernel
            .delegate_cell("cell-5", "agent-b", "Review $HOME", 3)
            .await;
        match previous {
            Some(value) => std::env::set_var("CRUSH_NOTEBOOK_SQUAD_MSG", value),
            None => std::env::remove_var("CRUSH_NOTEBOOK_SQUAD_MSG"),
        }

        let (state, outputs, stats) = result;
        assert_eq!(state, CellState::AiPending);
        assert_eq!(stats.unwrap().frontend, "delegate");
        assert_eq!(outputs[0].data["target"], "agent-b");
        assert_eq!(outputs[0].data["task"], "Review $HOME");
        let args = std::fs::read_to_string(&args_path).unwrap();
        assert_eq!(args.lines().next(), Some("@agent-b"));
        assert!(args.contains(&format!(
            "Notebook delegation cell=cell-5 run=3 from={sender}: Review $HOME"
        )));

        let _ = std::fs::remove_file(&command_path);
        let _ = std::fs::remove_file(&args_path);
    }

    #[tokio::test]
    async fn nb_022_delegate_cell_records_target_and_task() {
        let kernel = Kernel::new();
        let nb_path = std::env::temp_dir().join("test_delegate_cell.crush-nb");
        std::fs::write(
            &nb_path,
            serde_json::json!({
                "meta": {"title": "Delegate"}, "cells": []
            })
            .to_string(),
        )
        .unwrap();
        kernel
            .open(
                Some(serde_json::json!(1)),
                &serde_json::json!({"path": nb_path.to_str().unwrap()}),
            )
            .await;

        let response = kernel
            .insert_cell(
                Some(serde_json::json!(2)),
                &serde_json::json!({
                    "source":"Review the FastVM output", "kind":"ai_agent_delegate",
                    "delegate_to":"agent-b", "task":"Review the FastVM output"
                }),
            )
            .await;
        assert!(
            response.result.is_some(),
            "delegate insert should succeed: {:?}",
            response.error
        );

        let nb = kernel.nb.lock().await;
        let cell = &nb.doc.as_ref().unwrap().cells[0];
        assert_eq!(
            cell.kind,
            crush_notebook_core::CellKind::AiAgentDelegate {
                agent: "agent-b".into(),
                task: "Review the FastVM output".into()
            }
        );
        let _ = std::fs::remove_file(&nb_path);
    }

    #[tokio::test]
    async fn nb_022_delegate_cell_rejects_invalid_target_agent() {
        let kernel = Kernel::new();
        let nb_path = std::env::temp_dir().join("test_delegate_invalid_target.crush-nb");
        std::fs::write(
            &nb_path,
            serde_json::json!({
                "meta": {"title": "Delegate"}, "cells": []
            })
            .to_string(),
        )
        .unwrap();
        kernel
            .open(
                Some(serde_json::json!(1)),
                &serde_json::json!({"path": nb_path.to_str().unwrap()}),
            )
            .await;

        let response = kernel
            .insert_cell(
                Some(serde_json::json!(2)),
                &serde_json::json!({
                    "source":"Do the review", "kind":"ai_agent_delegate", "delegate_to":"agent name"
                }),
            )
            .await;
        assert_eq!(
            response.error.as_ref().map(|error| error.code),
            Some(-32602)
        );
        assert!(response.error.unwrap().message.contains("bare agent name"));
        let _ = std::fs::remove_file(&nb_path);
    }

    #[tokio::test]
    async fn nb_022_delegate_cell_requires_target_agent() {
        let kernel = Kernel::new();
        let nb_path = std::env::temp_dir().join("test_delegate_missing_target.crush-nb");
        std::fs::write(
            &nb_path,
            serde_json::json!({
                "meta": {"title": "Delegate"}, "cells": []
            })
            .to_string(),
        )
        .unwrap();
        kernel
            .open(
                Some(serde_json::json!(1)),
                &serde_json::json!({"path": nb_path.to_str().unwrap()}),
            )
            .await;

        let response = kernel
            .insert_cell(
                Some(serde_json::json!(2)),
                &serde_json::json!({
                    "source":"Do the review", "kind":"ai_agent_delegate"
                }),
            )
            .await;
        assert_eq!(
            response.error.as_ref().map(|error| error.code),
            Some(-32602)
        );
        let _ = std::fs::remove_file(&nb_path);
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    fn notebook_with_cells(ids: &[&str]) -> NotebookDocument {
        NotebookDocument {
            meta: crush_notebook_core::NotebookMeta::default(),
            cells: ids
                .iter()
                .map(|id| Cell {
                    id: id.to_string(),
                    kind: crush_notebook_core::CellKind::Crush,
                    source: String::new(),
                    state: CellState::Pending,
                    meta: Default::default(),
                    outputs: vec![],
                    execution: None,
                })
                .collect(),
        }
    }

    #[tokio::test]
    async fn nb_020_save_and_reload_tools() {
        let kernel = Kernel::new();
        let nb_path = std::env::temp_dir().join("test_save_reload.crush-nb");
        let doc = serde_json::json!({
            "meta": {"title": "Shared"},
            "cells": [{"id":"cell-a","kind":{"type":"crush"},"source":"let x=1","state":{"status":"pending"},"meta":{},"outputs":[]}]
        });
        std::fs::write(&nb_path, serde_json::to_string(&doc).unwrap()).unwrap();

        kernel
            .open(
                Some(serde_json::json!(1)),
                &serde_json::json!({"path": nb_path.to_str().unwrap()}),
            )
            .await;

        // save_tool should succeed
        let resp = kernel.save_tool(Some(serde_json::json!(2))).await;
        assert!(resp.result.is_some(), "save_tool failed: {:?}", resp.error);

        // reload_tool should say no changes (we just saved)
        let resp = kernel.reload_tool(Some(serde_json::json!(3))).await;
        assert!(resp.result.is_some());
        let text = resp.result.unwrap()["content"][0]["text"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(
            text.contains("No external changes"),
            "should detect no changes: {text}"
        );

        let _ = std::fs::remove_file(&nb_path);
    }

    #[tokio::test]
    async fn nb_020_two_kernels_share_notebook_via_file() {
        let nb_path = std::env::temp_dir().join("test_shared.crush-nb");
        let doc = serde_json::json!({
            "meta": {"title": "Agent A + B"},
            "cells": [{"id":"cell-orig","kind":{"type":"crush"},"source":"let x=1","state":{"status":"pending"},"meta":{},"outputs":[]}]
        });
        std::fs::write(&nb_path, serde_json::to_string(&doc).unwrap()).unwrap();

        // Agent A: opens notebook, sees 1 cell
        let kernel_a = Kernel::new();
        kernel_a
            .open(
                Some(serde_json::json!(1)),
                &serde_json::json!({"path": nb_path.to_str().unwrap()}),
            )
            .await;

        let resp = kernel_a.get_state(Some(serde_json::json!(2))).await;
        let text = resp.result.unwrap()["content"][0]["text"]
            .as_str()
            .unwrap()
            .to_string();
        let state: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(state["cells"].as_array().unwrap().len(), 1);

        // Agent B: opens same notebook, inserts a cell, saves
        let kernel_b = Kernel::new();
        kernel_b
            .open(
                Some(serde_json::json!(3)),
                &serde_json::json!({"path": nb_path.to_str().unwrap()}),
            )
            .await;
        kernel_b
            .insert_cell(
                Some(serde_json::json!(4)),
                &serde_json::json!({"source":"let y=2","kind":"crush"}),
            )
            .await;
        // insert_cell auto-saves now

        // Agent A: reloads → sees 2 cells
        let resp = kernel_a.reload_tool(Some(serde_json::json!(5))).await;
        let text = resp.result.unwrap()["content"][0]["text"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(
            text.contains("Reloaded"),
            "reload should detect external change: {text}"
        );

        let resp = kernel_a.get_state(Some(serde_json::json!(6))).await;
        let text = resp.result.unwrap()["content"][0]["text"]
            .as_str()
            .unwrap()
            .to_string();
        let state: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(
            state["cells"].as_array().unwrap().len(),
            2,
            "agent A should see 2 cells after reload"
        );

        let _ = std::fs::remove_file(&nb_path);
    }

    #[tokio::test]
    async fn nb_020_auto_reload_on_get_state_detects_external_changes() {
        let nb_path = std::env::temp_dir().join("test_auto_reload.crush-nb");
        let doc = serde_json::json!({
            "meta": {"title": "Auto-Reload"},
            "cells": [{"id":"cell-1","kind":{"type":"crush"},"source":"let a=1","state":{"status":"pending"},"meta":{},"outputs":[]}]
        });
        std::fs::write(&nb_path, serde_json::to_string(&doc).unwrap()).unwrap();

        let kernel = Kernel::new();
        kernel
            .open(
                Some(serde_json::json!(1)),
                &serde_json::json!({"path": nb_path.to_str().unwrap()}),
            )
            .await;

        // Another process writes a modified notebook
        let modified = serde_json::json!({
            "meta": {"title": "Auto-Reload"},
            "cells": [
                {"id":"cell-1","kind":{"type":"crush"},"source":"let a=1","state":{"status":"pending"},"meta":{},"outputs":[]},
                {"id":"cell-2","kind":{"type":"markdown"},"source":"# Added externally","state":{"status":"done"},"meta":{},"outputs":[]}
            ]
        });
        std::fs::write(&nb_path, serde_json::to_string(&modified).unwrap()).unwrap();

        // get_state should auto-reload and show 2 cells
        let resp = kernel.get_state(Some(serde_json::json!(2))).await;
        let text = resp.result.unwrap()["content"][0]["text"]
            .as_str()
            .unwrap()
            .to_string();
        let state: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(
            state["cells"].as_array().unwrap().len(),
            2,
            "auto-reload should pick up external changes"
        );

        let _ = std::fs::remove_file(&nb_path);
    }

    #[tokio::test]
    async fn nb_020_save_updates_modified_timestamp() {
        let nb_path = std::env::temp_dir().join("test_modified.crush-nb");
        let doc = serde_json::json!({
            "meta": {"title": "Modified Test"},
            "cells": []
        });
        std::fs::write(&nb_path, serde_json::to_string(&doc).unwrap()).unwrap();

        let kernel = Kernel::new();
        kernel
            .open(
                Some(serde_json::json!(1)),
                &serde_json::json!({"path": nb_path.to_str().unwrap()}),
            )
            .await;

        // Mutate (insert → auto-save)
        kernel
            .insert_cell(
                Some(serde_json::json!(2)),
                &serde_json::json!({"source":"let z=3","kind":"crush"}),
            )
            .await;

        // Read the file back — it should have a modified field
        let raw = std::fs::read_to_string(&nb_path).unwrap();
        let saved: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert!(
            saved["meta"]["modified"].is_string(),
            "saved notebook should have modified timestamp: {raw}"
        );

        let _ = std::fs::remove_file(&nb_path);
    }

    #[tokio::test]
    async fn nb_020_reload_with_no_file_path_is_noop() {
        let kernel = Kernel::new();
        // No notebook opened — maybe_reload should return Ok(false)
        let result = kernel.maybe_reload().await;
        assert!(result.is_ok());
        assert!(!result.unwrap(), "reload without file path should be noop");
    }

    #[tokio::test]
    async fn nb_020_save_sets_last_save_stamp() {
        let nb_path = std::env::temp_dir().join("test_mtime.crush-nb");
        let doc = serde_json::json!({
            "meta": {"title": "Mtime"},
            "cells": []
        });
        std::fs::write(&nb_path, serde_json::to_string(&doc).unwrap()).unwrap();

        let kernel = Kernel::new();
        kernel
            .open(
                Some(serde_json::json!(1)),
                &serde_json::json!({"path": nb_path.to_str().unwrap()}),
            )
            .await;

        // Before save, mtime should be set from open
        assert!(
            kernel.nb.lock().await.last_save_stamp.is_some(),
            "open should set last_save_stamp"
        );

        // Save should update it
        kernel.save_tool(Some(serde_json::json!(2))).await;
        assert!(
            kernel.nb.lock().await.last_save_stamp.is_some(),
            "save should update last_save_stamp"
        );

        let _ = std::fs::remove_file(&nb_path);
    }

    #[tokio::test]
    async fn nb_020_save_lock_is_exclusive_across_kernel_instances() {
        let nb_path =
            std::env::temp_dir().join(format!("test_save_lock-{}.crush-nb", uuid::Uuid::new_v4()));
        let lock_path = save_lock_path(&nb_path);
        let _ = std::fs::remove_file(&lock_path);

        let lock = acquire_save_lock(&nb_path, Duration::from_millis(100))
            .await
            .expect("first writer should acquire the notebook lock");
        let blocked = acquire_save_lock(&nb_path, Duration::from_millis(40)).await;
        assert!(
            blocked.as_ref().is_err(),
            "second writer must wait for the existing lock"
        );
        assert!(blocked.err().unwrap().contains("timed out"));

        drop(lock);
        let next = acquire_save_lock(&nb_path, Duration::from_millis(100)).await;
        assert!(
            next.is_ok(),
            "lock should be released after the first writer finishes"
        );
        drop(next);
        let _ = std::fs::remove_file(&lock_path);
    }

    #[tokio::test]
    async fn nb_020_save_replaces_file_atomically_and_cleans_temp_file() {
        let nb_path = std::env::temp_dir().join(format!(
            "test_atomic_save-{}.crush-nb",
            uuid::Uuid::new_v4()
        ));
        let filename = nb_path.file_name().unwrap().to_str().unwrap().to_owned();
        std::fs::write(
            &nb_path,
            serde_json::json!({
                "meta": {"title": "Atomic"}, "cells": []
            })
            .to_string(),
        )
        .unwrap();

        let kernel = Kernel::new();
        kernel
            .open(
                Some(serde_json::json!(1)),
                &serde_json::json!({"path": nb_path.to_str().unwrap()}),
            )
            .await;
        let response = kernel
            .insert_cell(
                Some(serde_json::json!(2)),
                &serde_json::json!({
                    "source": "atomic save", "kind": "markdown"
                }),
            )
            .await;
        assert!(
            response.result.is_some(),
            "insert should save atomically: {:?}",
            response.error
        );

        let saved: NotebookDocument =
            serde_json::from_str(&std::fs::read_to_string(&nb_path).unwrap())
                .expect("the replaced notebook must always be valid JSON");
        assert_eq!(saved.cells.len(), 1);
        let leftovers: Vec<_> = std::fs::read_dir(nb_path.parent().unwrap())
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with(&format!(".{filename}.tmp-"))
            })
            .collect();
        assert!(
            leftovers.is_empty(),
            "save should remove temporary files after rename"
        );
        assert!(
            !save_lock_path(&nb_path).exists(),
            "save lock should be released after saving"
        );

        let _ = std::fs::remove_file(&nb_path);
    }
}

// ── Main ─────────────────────────────────────────────────────────────────
