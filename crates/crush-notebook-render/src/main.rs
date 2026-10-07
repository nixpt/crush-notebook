//! Crush-Notebook Render CLI.
//!
//! Usage:
//!   crush-notebook-render hello.crush-nb          -> hello.html
//!   crush-notebook-render hello.crush-nb --out result.html

use anyhow::Result;
use clap::Parser;
use crush_notebook_core::NotebookDocument;

#[derive(Parser, Debug)]
#[command(name = "crush-notebook-render")]
struct Args {
    input: String,
    #[arg(short, long)]
    out: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let content = std::fs::read_to_string(&args.input)?;
    let doc: NotebookDocument = serde_json::from_str(&content)?;
    let html = crush_notebook_render::render_html(&doc);
    let out = args
        .out
        .unwrap_or_else(|| args.input.replace(".crush-nb", ".html"));
    std::fs::write(&out, html)?;
    println!("Rendered {} -> {}", args.input, out);
    Ok(())
}
