//! Core types for Crush-Notebook.
//!
//! Defines the notebook document format (`.crush-nb`), cell types,
//! execution state, and output types. Zero dependencies on the
//! compiler or VM — pure data model.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Notebook Document ───────────────────────────────────────────────────

/// A complete notebook document — the `.crush-nb` file format.
///
/// Serialized as JSON. A notebook is a sequence of cells with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotebookDocument {
    /// Notebook-wide metadata.
    pub meta: NotebookMeta,
    /// Ordered cells.
    pub cells: Vec<Cell>,
}

/// Notebook-level metadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NotebookMeta {
    /// Human-readable title.
    #[serde(default)]
    pub title: String,
    /// Description / purpose.
    #[serde(default)]
    pub description: String,
    /// Language frontends used (crush, sona, python, etc.).
    #[serde(default)]
    pub frontends: Vec<String>,
    /// Creation timestamp (ISO 8601).
    #[serde(default)]
    pub created: Option<String>,
    /// Last modified timestamp (ISO 8601).
    #[serde(default)]
    pub modified: Option<String>,
    /// Author / owner.
    #[serde(default)]
    pub author: Option<String>,
    /// Arbitrary key-value metadata.
    #[serde(default)]
    pub tags: HashMap<String, String>,
}

// ── Cell ────────────────────────────────────────────────────────────────

/// A single notebook cell.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cell {
    /// Unique cell ID (UUID).
    pub id: String,
    /// What kind of cell this is.
    pub kind: CellKind,
    /// The source code for this cell.
    pub source: String,
    /// Execution state.
    #[serde(default)]
    pub state: CellState,
    /// Cell-level metadata.
    #[serde(default)]
    pub meta: CellMeta,
    /// Outputs produced by this cell (empty if not yet executed).
    #[serde(default)]
    pub outputs: Vec<CellOutput>,
    /// Execution statistics (set after execution).
    #[serde(default)]
    pub execution: Option<ExecutionStats>,
}

/// What kind of cell — determines which frontend/compiler is used.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CellKind {
    /// Standard Crush language cell.
    Crush,
    /// Sona C-like frontend cell.
    Sona,
    /// Nepali frontend cell.
    Nepali,
    /// Python polyglot cell.
    Python,
    /// JavaScript polyglot cell. Serialized as `"javascript"`, matching the
    /// schema and `label()`; `"java_script"` (what `rename_all` used to emit)
    /// still loads.
    #[serde(rename = "javascript", alias = "java_script")]
    JavaScript,
    /// Raw text / documentation cell (Markdown).
    Markdown,
    /// AI-generated cell (with confidence).
    AiGenerated {
        confidence: Option<f64>,
        prompt: Option<String>,
    },
    /// AI query cell (natural language → generates code).
    AiQuery { query: String },
    /// Delegation cell that asks another agent to perform a task.
    AiAgentDelegate { agent: String, task: String },
}

impl CellKind {
    /// Which execution tier this cell kind targets.
    pub fn default_tier(&self) -> ExecutionTier {
        match self {
            CellKind::Crush
            | CellKind::Sona
            | CellKind::Nepali
            | CellKind::AiGenerated { .. }
            | CellKind::AiQuery { .. } => ExecutionTier::FastVM,
            CellKind::AiAgentDelegate { .. } => ExecutionTier::None,
            CellKind::Python | CellKind::JavaScript => ExecutionTier::Wasm,
            CellKind::Markdown => ExecutionTier::None,
        }
    }

    /// Human-readable label.
    pub fn label(&self) -> &str {
        match self {
            CellKind::Crush => "crush",
            CellKind::Sona => "sona",
            CellKind::Nepali => "nepali",
            CellKind::Python => "python",
            CellKind::JavaScript => "javascript",
            CellKind::Markdown => "markdown",
            CellKind::AiGenerated { .. } => "ai-generated",
            CellKind::AiQuery { .. } => "ai-query",
            CellKind::AiAgentDelegate { .. } => "ai-agent-delegate",
        }
    }
}

/// Per-cell metadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CellMeta {
    /// Collapsed state in the UI.
    #[serde(default)]
    pub collapsed: bool,
    /// Tags for filtering.
    #[serde(default)]
    pub tags: Vec<String>,
    /// @wip annotation (work-in-progress tracking).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wip: Option<WipAnnotation>,
    /// @temporary annotation (technical debt).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temporary: Option<TemporaryAnnotation>,
    /// @decision annotations (architectural decisions).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub decisions: Vec<DecisionAnnotation>,
    /// Arbitrary key-value extensions.
    #[serde(default)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// Work-in-progress annotation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WipAnnotation {
    pub intent: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_by: Option<String>,
    #[serde(default)]
    pub done: Vec<String>,
    #[serde(default)]
    pub todo: Vec<String>,
    #[serde(default)]
    pub unresolved: Vec<String>,
}

/// Temporary-code annotation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporaryAnnotation {
    pub reason: String,
    #[serde(default)]
    pub expires_when: Option<String>,
    #[serde(default)]
    pub owner: Option<String>,
    #[serde(default)]
    pub added: Option<String>,
}

/// Architectural decision annotation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionAnnotation {
    pub name: String,
    pub chose: String,
    #[serde(default)]
    pub over: Vec<String>,
    pub because: String,
    #[serde(default)]
    pub revisit_if: Vec<String>,
}

// ── Cell State ───��───────────────────────────────────────────────────────

/// Execution state of a cell.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum CellState {
    /// Never executed.
    #[default]
    Pending,
    /// Currently running.
    Running,
    /// Executed successfully.
    Done,
    /// Execution errored.
    Error { message: String },
    /// Execution was interrupted by the user.
    Interrupted,
    /// Waiting for AI response.
    AiPending,
}

impl CellState {
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            CellState::Done | CellState::Error { .. } | CellState::Interrupted
        )
    }
    pub fn label(&self) -> &str {
        match self {
            CellState::Pending => "pending",
            CellState::Running => "running",
            CellState::Done => "done",
            CellState::Error { .. } => "error",
            CellState::Interrupted => "interrupted",
            CellState::AiPending => "ai-pending",
        }
    }
    pub fn icon(&self) -> &str {
        match self {
            CellState::Pending => "pending",
            CellState::Running => "running",
            CellState::Done => "done",
            CellState::Error { .. } => "error",
            CellState::Interrupted => "interrupted",
            CellState::AiPending => "ai-pending",
        }
    }
}

// ── Output Types ─────────────────────────────────────────────────────────

/// Output produced by a cell execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellOutput {
    /// Unique output ID.
    pub id: String,
    /// Type of output.
    pub kind: OutputKind,
    /// Output data.
    pub data: serde_json::Value,
    /// Timestamp when this output was produced.
    #[serde(default)]
    pub timestamp: Option<String>,
}

/// Types of cell output.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "mime", rename_all = "snake_case")]
pub enum OutputKind {
    /// Plain text.
    Text { text: String },
    /// JSON structured data.
    Json,
    /// SVG image.
    Svg,
    /// PNG image (base64).
    Png { base64: String },
    /// HTML fragment.
    Html { html: String },
    /// Error output.
    Error {
        message: String,
        trace: Option<String>,
    },
    /// Streamed text (chunks appended over time).
    Stream { chunks: Vec<String> },
    /// AI-generated code proposal.
    AiProposal {
        code: String,
        language: String,
        confidence: Option<f64>,
    },
}

// ── Execution ────────────────────────────────────────────────────────────

/// Statistics from a cell execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStats {
    /// Number of VM steps executed.
    pub steps: usize,
    /// Wall-clock duration.
    pub duration_ms: u64,
    /// Which execution tier was used.
    pub tier: ExecutionTier,
    /// Which frontend compiled this cell.
    pub frontend: String,
    /// Whether the cell used JIT compilation.
    #[serde(default)]
    pub jit_compiled: bool,
}

/// Execution tier used for a cell.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionTier {
    /// Not executable (e.g., Markdown).
    None,
    /// CVM1 — portable interpreted VM (debuggable).
    Cvm1,
    /// FastVM — lowered bytecode hot path. Serialized as `"fastvm"`, matching
    /// the schema; `"fast_v_m"` (what `rename_all` used to emit) still loads.
    #[serde(rename = "fastvm", alias = "fast_v_m")]
    FastVM,
    /// crush-jit — Cranelift native code.
    Jit,
    /// WASM/WASI sandbox (polyglot cells).
    Wasm,
}

impl ExecutionTier {
    pub fn label(&self) -> &str {
        match self {
            ExecutionTier::None => "none",
            ExecutionTier::Cvm1 => "CVM1",
            ExecutionTier::FastVM => "FastVM",
            ExecutionTier::Jit => "JIT",
            ExecutionTier::Wasm => "WASM",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn javascript_kind_matches_schema_and_reads_legacy_name() {
        let json = serde_json::to_value(CellKind::JavaScript).unwrap();
        assert_eq!(json, serde_json::json!({"type": "javascript"}));
        assert_eq!(CellKind::JavaScript.label(), "javascript");
        for name in ["javascript", "java_script"] {
            let kind: CellKind =
                serde_json::from_value(serde_json::json!({ "type": name })).unwrap();
            assert_eq!(kind, CellKind::JavaScript);
        }
    }

    #[test]
    fn fastvm_tier_matches_schema_and_reads_legacy_name() {
        assert_eq!(
            serde_json::to_value(ExecutionTier::FastVM).unwrap(),
            serde_json::json!("fastvm")
        );
        for name in ["fastvm", "fast_v_m"] {
            let tier: ExecutionTier = serde_json::from_value(serde_json::json!(name)).unwrap();
            assert_eq!(tier, ExecutionTier::FastVM);
        }
    }
}
