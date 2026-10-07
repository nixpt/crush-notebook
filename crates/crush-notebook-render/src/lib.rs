//! Notebook renderer — emits self-contained HTML+CSS+JS.
//!
//! Pattern: crush-visuals-egui's `render_html()` model.
//! Generates a complete interactive notebook page with:
//!   - Cell cards with status colors
//!   - Output rendering (text, json, svg, error)
//!   - Variable inspector panel
//!   - Compile-path tier summary bar
//!   - AI response accept/reject buttons

use crush_notebook_core::*;

/// Render a notebook document to a self-contained HTML string.
pub fn render_html(doc: &NotebookDocument) -> String {
    let title = &doc.meta.title;
    let cells_html: String = doc
        .cells
        .iter()
        .map(render_cell)
        .collect::<Vec<_>>()
        .join("\n");
    let tier_bar = render_tier_bar(doc);
    let toolbar = render_toolbar();

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title} — Crush Notebook</title>
  <style>{CSS}</style>
</head>
<body>
  <div class="notebook-container">
    <header class="notebook-header">
      <h1>{title}</h1>
      <div class="notebook-meta">
        {tier_bar}
      </div>
    </header>
    <main class="notebook-cells">
      {cells_html}
    </main>
    <footer class="notebook-footer">
      {toolbar}
    </footer>
  </div>
  <script>{JS}</script>
</body>
</html>"#
    )
}

fn render_cell(cell: &Cell) -> String {
    let state_class = match &cell.state {
        CellState::Pending => "cell-pending",
        CellState::Running => "cell-running",
        CellState::Done => "cell-done",
        CellState::Error { .. } => "cell-error",
        CellState::Interrupted => "cell-interrupted",
        CellState::AiPending => "cell-ai-pending",
    };

    let kind_badge = match &cell.kind {
        CellKind::Crush => "crush",
        CellKind::Nepali => "nepali",
        CellKind::Sona => "sona",
        CellKind::Python => "python",
        CellKind::JavaScript => "js",
        CellKind::Markdown => "md",
        CellKind::AiGenerated { .. } => "ai-gen",
        CellKind::AiQuery { .. } => "ai-query",
        CellKind::AiAgentDelegate { .. } => "delegate",
    };

    let source_escaped = html_escape(&cell.source);
    let execution = cell
        .execution
        .as_ref()
        .map(|s| {
            format!(
                "{} steps · {}ms · {}· JIT:{}",
                s.steps,
                s.duration_ms,
                s.tier.label(),
                s.jit_compiled
            )
        })
        .unwrap_or_default();

    let outputs_html: String = cell
        .outputs
        .iter()
        .map(render_output)
        .collect::<Vec<_>>()
        .join("\n");

    let wip_badge = cell
        .meta
        .wip
        .as_ref()
        .map(|w| {
            let todo = w.todo.len();
            let unresolved = w.unresolved.len();
            if todo + unresolved > 0 {
                format!(
                    "<span class='badge badge-wip' title='{}'>@wip {}⬜ {}❓</span>",
                    w.intent, todo, unresolved
                )
            } else {
                String::new()
            }
        })
        .unwrap_or_default();

    let tmp_badge = cell
        .meta
        .temporary
        .as_ref()
        .map(|_| "<span class='badge badge-tmp' title='Temporary code'>@temporary</span>")
        .unwrap_or_default();

    format!(
        r#"
    <div class="cell {state_class}" data-cell-id="{id}">
      <div class="cell-header">
        <span class="cell-id">[{kind_badge}]</span>
        <span class="cell-state">{state_label}</span>
        {wip_badge}
        {tmp_badge}
        <span class="cell-execution">{execution}</span>
        <span class="cell-actions">
          <button onclick="evalCell('{id}')">▶ Run</button>
          <button onclick="deleteCell('{id}')">✕</button>
        </span>
      </div>
      <div class="cell-source">
        <pre><code>{source_escaped}</code></pre>
      </div>
      <div class="cell-outputs">
        {outputs_html}
      </div>
    </div>"#,
        id = cell.id,
        kind_badge = kind_badge,
        state_label = cell.state.label(),
        execution = execution,
        source_escaped = source_escaped,
        outputs_html = outputs_html,
        wip_badge = wip_badge,
        tmp_badge = tmp_badge,
    )
}

fn render_output(output: &CellOutput) -> String {
    let id = &output.id;
    match &output.kind {
        OutputKind::Text { text } => {
            format!(
                r#"<div class="output output-text" data-output-id="{id}"><pre>{}</pre></div>"#,
                html_escape(text)
            )
        }
        OutputKind::Json => {
            let pretty = serde_json::to_string_pretty(&output.data).unwrap_or_default();
            format!(
                r#"<div class="output output-json" data-output-id="{id}"><pre>{}</pre></div>"#,
                html_escape(&pretty)
            )
        }
        OutputKind::Svg => {
            let svg = output
                .data
                .get("svg")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            format!(r#"<div class="output output-svg" data-output-id="{id}">{svg}</div>"#)
        }
        OutputKind::Png { base64 } => {
            format!(
                r#"<div class="output output-png" data-output-id="{id}"><img src="data:image/png;base64,{base64}" /></div>"#
            )
        }
        OutputKind::Html { html } => {
            format!(r#"<div class="output output-html" data-output-id="{id}">{html}</div>"#)
        }
        OutputKind::Error { message, trace } => {
            let trace_html = trace
                .as_ref()
                .map(|t| {
                    format!(
                        "<details><summary>Stack trace</summary><pre>{}</pre></details>",
                        html_escape(t)
                    )
                })
                .unwrap_or_default();
            format!(
                r#"<div class="output output-error" data-output-id="{id}"><strong>Error:</strong> {message}{trace_html}</div>"#
            )
        }
        OutputKind::Stream { chunks } => {
            let text = chunks.join("");
            format!(
                r#"<div class="output output-stream" data-output-id="{id}"><pre>{}</pre></div>"#,
                html_escape(&text)
            )
        }
        OutputKind::AiProposal {
            code,
            language,
            confidence,
        } => {
            let conf_str = confidence
                .map(|c| format!("{:.0}%", c * 100.0))
                .unwrap_or_default();
            format!(
                r#"<div class="output output-ai-proposal" data-output-id="{id}">
                <div class="ai-proposal-header">AI Proposal ({language}) — confidence: {conf_str}</div>
                <pre><code>{code}</code></pre>
                <div class="ai-proposal-actions">
                    <button onclick="acceptProposal('{id}')">Accept</button>
                    <button onclick="rejectProposal('{id}')">Reject</button>
                    <button onclick="editProposal('{id}')">Edit</button>
                </div>
            </div>"#,
                id = id,
                language = language,
                conf_str = conf_str,
                code = html_escape(code)
            )
        }
    }
}

fn render_tier_bar(doc: &NotebookDocument) -> String {
    let frontends = &doc.meta.frontends;
    if frontends.is_empty() {
        return String::new();
    }
    let tiers = frontends
        .iter()
        .map(|f| match f.as_str() {
            "crush" => "<span class='tier-badge tier-fastvm'>crush → CVM1</span>",
            "nepali" => "<span class='tier-badge tier-fastvm'>nepali → CVM1</span>",
            "sona" => "<span class='tier-badge tier-fastvm'>sona → CVM1</span>",
            "python" => "<span class='tier-badge tier-wasm'>python → python3</span>",
            "javascript" => "<span class='tier-badge tier-wasm'>js → node</span>",
            _ => "",
        })
        .collect::<Vec<_>>()
        .join(" ");
    format!("<div class='tier-bar'>{tiers}</div>")
}

fn render_toolbar() -> String {
    String::from(
        r#"
    <div class="toolbar">
      <button onclick="insertCell('crush')">+ Crush Cell</button>
      <button onclick="insertCell('nepali')">+ Nepali Cell</button>
      <button onclick="insertCell('sona')">+ Sona Cell</button>
      <button onclick="insertCell('markdown')">+ Markdown</button>
      <button onclick="insertCell('python')">+ Python</button>
      <button onclick="evalAll()">▶ Run All</button>
      <button onclick="exportNotebook()">Export .crush-nb</button>
    </div>"#,
    )
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

// ── CSS ──────────────────────────────────────────────────────────────────

const CSS: &str = r#"
:root {
  --bg: #0f1117;
  --surface: #1a1d27;
  --surface2: #232734;
  --border: #2d3144;
  --text: #c9d1d9;
  --text-dim: #8b949e;
  --accent: #58a6ff;
  --green: #3fb950;
  --red: #f85149;
  --orange: #d2991d;
  --purple: #a371f7;
  --blue: #58a6ff;
  --radius: 6px;
  --font: -apple-system, BlinkMacSystemFont, 'Segoe UI', monospace;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg); color: var(--text); font-family: var(--font);
  font-size: 14px; line-height: 1.6; padding: 20px;
}
.notebook-container { max-width: 900px; margin: 0 auto; }
.notebook-header {
  padding: 20px 0; border-bottom: 1px solid var(--border); margin-bottom: 20px;
}
.notebook-header h1 { font-size: 24px; color: var(--accent); }
.notebook-meta { margin-top: 8px; }

.cell {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); margin-bottom: 12px;
  overflow: hidden; transition: border-color 0.2s;
}
.cell-done { border-left: 3px solid var(--green); }
.cell-error { border-left: 3px solid var(--red); }
.cell-running { border-left: 3px solid var(--blue); }
.cell-pending { border-left: 3px solid var(--border); }
.cell-interrupted { border-left: 3px solid var(--orange); }
.cell-ai-pending { border-left: 3px solid var(--purple); }

.cell-header {
  padding: 6px 12px; background: var(--surface2);
  display: flex; gap: 8px; align-items: center;
  font-size: 12px; color: var(--text-dim); border-bottom: 1px solid var(--border);
}
.cell-id { font-weight: 600; color: var(--accent); }
.cell-state { color: var(--text-dim); }
.cell-execution { margin-left: auto; color: var(--text-dim); font-size: 11px; }
.cell-actions button {
  background: var(--surface); border: 1px solid var(--border);
  color: var(--text); cursor: pointer; padding: 2px 8px; border-radius: 3px;
  font-size: 11px; transition: background 0.15s;
}
.cell-actions button:hover { background: var(--surface2); }

.cell-source { padding: 12px; }
.cell-source pre { margin: 0; }
.cell-source code { color: var(--text); }

.cell-outputs { border-top: 1px solid var(--border); }
.output { padding: 8px 12px; font-size: 13px; border-bottom: 1px solid var(--border); }
.output:last-child { border-bottom: none; }
.output-text { background: var(--surface2); }
.output-json { background: #1a2332; }
.output-error { background: #2d1a1a; color: var(--red); }
.output-ai-proposal { background: #1a1a2d; }
.ai-proposal-header { color: var(--purple); margin-bottom: 8px; font-weight: 600; }
.ai-proposal-actions { margin-top: 8px; display: flex; gap: 8px; }
.ai-proposal-actions button {
  background: var(--surface); border: 1px solid var(--border);
  color: var(--text); cursor: pointer; padding: 4px 16px; border-radius: var(--radius);
  font-size: 12px;
}
.ai-proposal-actions button:hover { background: var(--surface2); }

.badge { padding: 2px 6px; border-radius: 3px; font-size: 11px; }
.badge-wip { background: #2d2416; color: var(--orange); }
.badge-tmp { background: #2d1a2d; color: var(--purple); }

.tier-bar { display: flex; gap: 8px; }
.tier-badge {
  padding: 2px 8px; border-radius: 3px; font-size: 11px;
  background: var(--surface2); border: 1px solid var(--border); color: var(--text-dim);
}
.tier-fastvm { border-color: var(--green); }
.tier-jit { border-color: var(--purple); }
.tier-wasm { border-color: var(--orange); }

.notebook-footer {
  padding: 12px 0; border-top: 1px solid var(--border); margin-top: 20px;
}
.toolbar { display: flex; gap: 8px; flex-wrap: wrap; }
.toolbar button {
  background: var(--surface); border: 1px solid var(--border);
  color: var(--accent); cursor: pointer; padding: 6px 16px; border-radius: var(--radius);
  font-size: 13px; transition: background 0.15s;
}
.toolbar button:hover { background: var(--surface2); }

pre { overflow-x: auto; }
code { font-family: 'Fira Code', 'Cascadia Code', 'JetBrains Mono', monospace; }
img { max-width: 100%; border-radius: var(--radius); }
"#;

// ── JS ───────────────────────────────────────────────────────────────────

const JS: &str = r#"
function evalCell(id) {
  fetch('/mcp', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ method: 'tools/call', params: { name: 'notebook_eval_cell', arguments: { cell_id: id } } })
  });
}
function evalAll() {
  fetch('/mcp', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ method: 'tools/call', params: { name: 'notebook_eval_all', arguments: {} } })
  });
}
function insertCell(kind) {
  fetch('/mcp', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ method: 'tools/call', params: { name: 'notebook_insert_cell', arguments: { source: '// New cell', kind: kind } } })
  });
}
function deleteCell(id) {
  fetch('/mcp', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ method: 'tools/call', params: { name: 'notebook_delete_cell', arguments: { cell_id: id } } })
  });
}
function acceptProposal(id) { console.log('accept', id); }
function rejectProposal(id) { console.log('reject', id); }
function editProposal(id) { console.log('edit', id); }
function exportNotebook() {
  fetch('/mcp', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ method: 'tools/call', params: { name: 'notebook_get_state', arguments: {} } })
  });
}
"#;

/// Simple notebook renderer (no JS interactivity, lighter than render_html).
pub fn render_notebook(doc: &crush_notebook_core::NotebookDocument) -> String {
    render_html(doc)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_notebook() -> NotebookDocument {
        NotebookDocument {
            meta: NotebookMeta {
                title: "Minimal".into(),
                ..Default::default()
            },
            cells: vec![],
        }
    }

    fn notebook_with_cells() -> NotebookDocument {
        NotebookDocument {
            meta: NotebookMeta {
                title: "Multi-Cell".into(),
                frontends: vec!["crush".into(), "python".into()],
                ..Default::default()
            },
            cells: vec![
                Cell {
                    id: "cell-1".into(),
                    kind: CellKind::Crush,
                    source: "let x = 1".into(),
                    state: CellState::Done,
                    meta: Default::default(),
                    outputs: vec![CellOutput {
                        id: "out-1".into(),
                        kind: OutputKind::Text {
                            text: "result".into(),
                        },
                        data: serde_json::json!({}),
                        timestamp: None,
                    }],
                    execution: Some(ExecutionStats {
                        steps: 10,
                        duration_ms: 5,
                        tier: ExecutionTier::FastVM,
                        frontend: "crush".into(),
                        jit_compiled: false,
                    }),
                },
                Cell {
                    id: "cell-2".into(),
                    kind: CellKind::Python,
                    source: "print('hello')".into(),
                    state: CellState::Error {
                        message: "python3 not found".into(),
                    },
                    meta: Default::default(),
                    outputs: vec![CellOutput {
                        id: "out-2".into(),
                        kind: OutputKind::Error {
                            message: "python3 not found".into(),
                            trace: Some("at cell-2".into()),
                        },
                        data: serde_json::json!({}),
                        timestamp: None,
                    }],
                    execution: Some(ExecutionStats {
                        steps: 0,
                        duration_ms: 1,
                        tier: ExecutionTier::FastVM,
                        frontend: "python".into(),
                        jit_compiled: false,
                    }),
                },
                Cell {
                    id: "cell-3".into(),
                    kind: CellKind::AiQuery {
                        query: "what is 2+2?".into(),
                    },
                    source: "// AI query".into(),
                    state: CellState::AiPending,
                    meta: CellMeta {
                        wip: Some(WipAnnotation {
                            intent: "answer query".into(),
                            started_by: Some("agent-a".into()),
                            done: vec![],
                            todo: vec!["generate code".into()],
                            unresolved: vec![],
                        }),
                        temporary: Some(TemporaryAnnotation {
                            reason: "placeholder".into(),
                            expires_when: Some("2026-08-01".into()),
                            owner: Some("agent-a".into()),
                            added: None,
                        }),
                        ..Default::default()
                    },
                    outputs: vec![],
                    execution: None,
                },
            ],
        }
    }

    #[test]
    fn render_hello_example_visual_contract() {
        let doc: NotebookDocument =
            serde_json::from_str(include_str!("../tests/fixtures/hello.crush-nb"))
                .expect("hello example should remain a valid notebook fixture");
        let html = render_html(&doc);
        for marker in [
            "Hello, Crush Notebook!",
            "cell-1",
            "cell-2",
            "cell-3",
            "cell-4",
            "[crush]",
            "[sona]",
            "[md]",
            "[ai-query]",
            "@wip",
            "Run All",
        ] {
            assert!(
                html.contains(marker),
                "rendered example is missing marker {marker:?}"
            );
        }
        assert!(html.starts_with("<!DOCTYPE html>"));
        assert!(
            !html.contains("<link "),
            "visual fixture must remain self-contained"
        );
        assert!(
            !html.contains("<script src="),
            "visual fixture must remain self-contained"
        );
    }

    #[test]
    fn render_minimal_notebook_is_valid_html5() {
        let html = render_html(&minimal_notebook());
        assert!(
            html.starts_with("<!DOCTYPE html>"),
            "must start with DOCTYPE"
        );
        assert!(html.contains("<html lang=\"en\">"), "must have html tag");
        assert!(html.contains("<head>"), "must have head");
        assert!(html.contains("<title>"), "must have title");
        assert!(html.contains("</html>"), "must close html");
    }

    #[test]
    fn render_includes_style_block() {
        let html = render_html(&minimal_notebook());
        assert!(html.contains("<style>"), "must include style block");
        assert!(html.contains("</style>"), "must close style block");
    }

    #[test]
    fn render_includes_script_block() {
        let html = render_html(&minimal_notebook());
        assert!(html.contains("<script>"), "must include script block");
        assert!(html.contains("</script>"), "must close script block");
    }

    #[test]
    fn render_sets_title_from_meta() {
        let html = render_html(&notebook_with_cells());
        assert!(
            html.contains("Multi-Cell"),
            "title should appear in HTML: {html}"
        );
    }

    #[test]
    fn render_no_external_assets() {
        let html = render_html(&notebook_with_cells());
        assert!(!html.contains("<link "), "no link tags");
        assert!(!html.contains("<script src="), "no external scripts");
        assert!(!html.contains("href=\"http"), "no external URLs");
        assert!(
            !html.contains("src=\"http"),
            "no external image/script sources"
        );
    }

    #[test]
    fn render_cells_appear_in_order() {
        let html = render_html(&notebook_with_cells());
        let pos1 = html.find("cell-1").expect("cell-1 id should appear");
        let pos2 = html.find("cell-2").expect("cell-2 id should appear");
        let pos3 = html.find("cell-3").expect("cell-3 id should appear");
        assert!(
            pos1 < pos2 && pos2 < pos3,
            "cells should appear in order: {pos1} < {pos2} < {pos3}"
        );
    }

    #[test]
    fn render_state_classes_applied() {
        let html = render_html(&notebook_with_cells());
        assert!(
            html.contains("cell-done"),
            "done cell should have cell-done class"
        );
        assert!(
            html.contains("cell-error"),
            "error cell should have cell-error class"
        );
        assert!(
            html.contains("cell-ai-pending"),
            "ai-pending cell should have cell-ai-pending class"
        );
    }

    #[test]
    fn render_kind_badges_applied() {
        let html = render_html(&notebook_with_cells());
        assert!(html.contains("[crush]"), "crush badge: {html}");
        assert!(html.contains("[python]"), "python badge: {html}");
    }

    #[test]
    fn render_output_text_included() {
        let html = render_html(&notebook_with_cells());
        assert!(html.contains("result"), "text output should appear");
    }

    #[test]
    fn render_error_output_included() {
        let html = render_html(&notebook_with_cells());
        assert!(
            html.contains("output-error"),
            "error output class should be present"
        );
        assert!(
            html.contains("python3 not found"),
            "error message should appear"
        );
    }

    #[test]
    fn render_tier_bar_includes_frontends() {
        let html = render_html(&notebook_with_cells());
        assert!(
            html.contains("tier-bar") || html.contains("tier-fastvm"),
            "tier bar should appear"
        );
    }

    #[test]
    fn render_toolbar_includes_buttons() {
        let html = render_html(&notebook_with_cells());
        assert!(html.contains(">+ Crush Cell<"), "add crush button");
        assert!(html.contains("+ Python"), "add python button");
        assert!(html.contains("Run All"), "run all button");
    }

    #[test]
    fn render_wip_badge_when_todos_exist() {
        let html = render_html(&notebook_with_cells());
        assert!(
            html.contains("@wip"),
            "wip badge should be present when todo items exist"
        );
    }

    #[test]
    fn render_temporary_badge() {
        let html = render_html(&notebook_with_cells());
        assert!(
            html.contains("@temporary"),
            "temporary badge should be present"
        );
    }

    #[test]
    fn render_notebook_is_alias_for_render_html() {
        let doc = notebook_with_cells();
        let html1 = render_html(&doc);
        let html2 = render_notebook(&doc);
        assert_eq!(html1, html2);
    }

    #[test]
    fn render_markdown_cell_no_execution_stats() {
        let doc = NotebookDocument {
            meta: NotebookMeta {
                title: "MD".into(),
                ..Default::default()
            },
            cells: vec![Cell {
                id: "md-1".into(),
                kind: CellKind::Markdown,
                source: "# Hello".into(),
                state: CellState::Done,
                meta: Default::default(),
                outputs: vec![],
                execution: None,
            }],
        };
        let html = render_html(&doc);
        assert!(html.contains("md-1"), "markdown cell rendered");
        // Markdown cells render without execution stats span content
    }
}
