//! `holm-index` command line tool.
//!
//! ```text
//! holm-index build [--schema F] [--taxonomy F]... [--facts F]... [--meta F]...
//!                  --out F [--tmp DIR] [--chunk-mb N]
//! holm-index stats F
//! ```

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use hg_core::container::{Container, Verify};
use hg_index::build::{build_index, BuildConfig, Predicates};
use hg_index::index::IndexView;

const USAGE: &str = "usage:\n  holm-index build [--schema F] [--taxonomy F]... [--facts F]... [--meta F]... --out F [--tmp DIR] [--chunk-mb N]\n  holm-index stats F";

fn parse_build(args: &[String]) -> Result<BuildConfig, String> {
    let mut cfg = BuildConfig {
        schema: None,
        taxonomy: Vec::new(),
        facts: Vec::new(),
        meta: Vec::new(),
        out: PathBuf::new(),
        tmp_dir: std::env::temp_dir(),
        chunk_bytes: 256 << 20,
        predicates: Predicates::yago46(),
    };
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let value = it
            .next()
            .ok_or_else(|| format!("missing value after {flag}"))?;
        match flag.as_str() {
            "--schema" => cfg.schema = Some(value.into()),
            "--taxonomy" => cfg.taxonomy.push(value.into()),
            "--facts" => cfg.facts.push(value.into()),
            "--meta" => cfg.meta.push(value.into()),
            "--out" => cfg.out = value.into(),
            "--tmp" => cfg.tmp_dir = value.into(),
            "--chunk-mb" => {
                let mb: usize = value
                    .parse()
                    .map_err(|_| format!("bad --chunk-mb `{value}`"))?;
                cfg.chunk_bytes = mb.max(1) << 20;
            }
            _ => return Err(format!("unknown flag {flag}")),
        }
    }
    if cfg.out.as_os_str().is_empty() {
        return Err("--out is required".into());
    }
    Ok(cfg)
}

fn run(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("build") => {
            let cfg = parse_build(&args[1..])?;
            let t0 = Instant::now();
            let s = build_index(&cfg).map_err(|e| e.to_string())?;
            println!("nodes {}", s.nodes);
            println!("edges {}", s.edges);
            println!("relations {}", s.relations);
            println!("classes {}", s.classes);
            println!("type_pairs {}", s.type_pairs);
            println!("typed_nodes {}", s.typed_nodes);
            println!("intervals {}", s.intervals);
            println!("duplicate_edges {}", s.duplicate_edges);
            println!("qids {}", s.qids);
            println!("ignored_meta {}", s.ignored_meta);
            println!("seconds {:.3}", t0.elapsed().as_secs_f64());
            Ok(())
        }
        Some("stats") => {
            let path = args.get(1).ok_or("stats needs a file")?;
            let c = Container::open(path.as_ref(), Verify::Header).map_err(|e| e.to_string())?;
            let ix = IndexView::new(&c).map_err(|e| e.to_string())?;
            println!("nodes {}", ix.graph.n_nodes());
            println!("edges {}", ix.graph.n_edges());
            println!("incidences {}", ix.graph.n_incidences());
            println!("relations {}", ix.n_relations());
            println!("classes {}", ix.n_classes());
            println!("intervals {}", ix.intervals.len());
            Ok(())
        }
        _ => Err("unknown command".into()),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}\n{USAGE}");
            ExitCode::from(2)
        }
    }
}
