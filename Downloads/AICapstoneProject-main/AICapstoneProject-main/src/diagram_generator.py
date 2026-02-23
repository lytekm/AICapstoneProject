# src/diagram_generator.py
"""
Diagram Generator (Graphviz-first)

Purpose
- Generates submission-ready architecture diagrams automatically as:
  - Graphviz DOT source (.dot)
  - Rendered images (.png and .svg)
- Optional: also saves Mermaid-safe (.mmd) for draw.io, if you want.

Requirements
- Graphviz installed and on PATH (you already verified: `dot -V`)
- Python package:
    py -m pip install graphviz

Outputs (default folder: docs/diagrams)
- ai_capability_architecture.(dot|png|svg)
- data_flow_pipes_filters.(dot|png|svg)
- component_interaction.(dot|png|svg)

How to use
- In your runner:
    from src.diagram_generator import DiagramGenerator
    DiagramGenerator().generate_all()
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Dict, Optional

try:
    from graphviz import Source
except ImportError:
    Source = None  # type: ignore


# -----------------------------
# Config
# -----------------------------

@dataclass
class DiagramPaths:
    out_dir: str = "docs/diagrams"


# -----------------------------
# Generator
# -----------------------------

class DiagramGenerator:
    def __init__(self, paths: Optional[DiagramPaths] = None):
        self.paths = paths or DiagramPaths()

    # ---------- helpers ----------

    def _ensure_dir(self) -> None:
        os.makedirs(self.paths.out_dir, exist_ok=True)

    def _dot_available(self) -> bool:
        return shutil.which("dot") is not None

    def _graphviz_py_available(self) -> bool:
        return Source is not None

    def _write_text(self, filename: str, content: str) -> str:
        self._ensure_dir()
        path = os.path.join(self.paths.out_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content.strip() + "\n")
        return path

    def _render_dot(self, name_no_ext: str, dot_text: str) -> Dict[str, str]:
        """
        Render DOT to PNG + SVG using graphviz python package (calls `dot`).
        Returns dict of output paths. Raises if requirements missing.
        """
        if not self._dot_available():
            raise RuntimeError(
                "Graphviz 'dot' not found in PATH. Install Graphviz and ensure it's on PATH."
            )
        if not self._graphviz_py_available():
            raise RuntimeError(
                "Python package 'graphviz' not installed. Run: py -m pip install graphviz"
            )

        self._ensure_dir()

        outputs: Dict[str, str] = {}
        base = os.path.join(self.paths.out_dir, name_no_ext)

        # PNG
        src = Source(dot_text)
        src.format = "png"
        out_png = src.render(filename=base, cleanup=True)  # returns full path
        outputs["png"] = out_png

        # SVG
        src = Source(dot_text)
        src.format = "svg"
        out_svg = src.render(filename=base, cleanup=True)
        outputs["svg"] = out_svg

        return outputs

    # -----------------------------
    # DOT templates (edit labels here)
    # -----------------------------

    def ai_capability_architecture_dot(self) -> str:
        # High-level: data sources -> modules -> artifacts
        return r"""
digraph AI_Capability_Architecture {
  rankdir=LR;
  graph [fontsize=12, fontname="Arial", labelloc="t",
         label="AI Capability Architecture (High-Level)"];
  node  [fontname="Arial", fontsize=10, shape=box, style="rounded,filled"];
  edge  [fontname="Arial", fontsize=10];

  // ---- External data sources ----
  subgraph cluster_external {
    label="External Data Sources";
    color="#888888";
    style="rounded";
    RSS [label="Public RSS Feeds\n(CBC/BBC/Reuters)", fillcolor="#f2f2f2"];
    HF  [label="HuggingFace Datasets\n(CNN/DailyMail)", fillcolor="#f2f2f2"];
  }

  // ---- Frameworks / Libraries ----
  subgraph cluster_fw {
    label="Frameworks / Libraries";
    color="#888888";
    style="rounded";
    PY   [label="Python 3.x", fillcolor="#f2f2f2"];
    SK   [label="scikit-learn\nTF-IDF, cosine", fillcolor="#f2f2f2"];
    NLTK [label="NLTK\nsentence tokenize", fillcolor="#f2f2f2"];
    ROUGE[label="rouge-score", fillcolor="#f2f2f2"];
    FP   [label="feedparser\n(RSS parse)", fillcolor="#f2f2f2"];
    TR   [label="trafilatura\n(article extraction)", fillcolor="#f2f2f2"];
    DS   [label="datasets\n(HF loader)", fillcolor="#f2f2f2"];
  }

  // ---- Software modules (classes) ----
  subgraph cluster_mod {
    label="Software Modules (Classes)";
    color="#1a7f37";
    style="rounded";

    PIPE [label="NewsDataPipeline\n- fetch RSS\n- extract article\n- normalize\n- tokenize stats",
          fillcolor="#eaffea"];

    LOADER [label="CNNDailyMailDatasetLoader\n- load train/test splits",
            fillcolor="#eaffea"];

    MODEL [label="TextRankMMRSummarizer\n- sentence tokenize\n- TextRank(PageRank)\n- MMR selection",
           fillcolor="#eaffea"];

    TRAIN [label="SummarizerTrainer\n- hyperparameter tuning",
           fillcolor="#eaffea"];

    EVAL [label="RougeEvaluator\n- ROUGE-1 / ROUGE-2 / ROUGE-L",
          fillcolor="#eaffea"];
  }

  // ---- Artifacts ----
  subgraph cluster_art {
    label="Artifacts (Saved Outputs)";
    color="#a36a00";
    style="rounded";
    SUMCSV [label="outputs/summaries.csv", fillcolor="#fff3d6"];
    BEST   [label="outputs/best_config_cnndm.json", fillcolor="#fff3d6"];
    REP    [label="outputs/eval_report_cnndm.json", fillcolor="#fff3d6"];
    DIAG   [label="docs/diagrams/*.png|*.svg|*.dot", fillcolor="#fff3d6"];
  }

  // ---- Connections ----
  RSS -> PIPE;
  PIPE -> MODEL;
  MODEL -> SUMCSV;

  HF -> LOADER;
  LOADER -> TRAIN;
  TRAIN -> BEST;

  TRAIN -> MODEL;
  MODEL -> EVAL;
  EVAL -> REP;

  PIPE -> DIAG;
  LOADER -> DIAG;
  MODEL -> DIAG;
  TRAIN -> DIAG;
  EVAL -> DIAG;

  // Framework relationships (dotted for documentation)
  PY -> SK   [style=dashed, color="#666666"];
  PY -> NLTK [style=dashed, color="#666666"];
  PY -> ROUGE[style=dashed, color="#666666"];
  PY -> FP   [style=dashed, color="#666666"];
  PY -> TR   [style=dashed, color="#666666"];
  PY -> DS   [style=dashed, color="#666666"];

  FP -> PIPE [style=dashed, color="#666666"];
  TR -> PIPE [style=dashed, color="#666666"];
  DS -> LOADER [style=dashed, color="#666666"];
  SK -> MODEL [style=dashed, color="#666666"];
  NLTK -> MODEL [style=dashed, color="#666666"];
  ROUGE -> EVAL [style=dashed, color="#666666"];
}
"""

    def data_flow_pipes_filters_dot(self) -> str:
        # Pipes & filters view of the pipeline (good for "Data Flow Architecture" requirement)
        return r"""
digraph Data_Flow_Pipes_Filters {
  rankdir=LR;
  graph [fontsize=12, fontname="Arial", labelloc="t",
         label="Data Flow Architecture (Pipes & Filters)"];
  node  [fontname="Arial", fontsize=10, shape=box, style="rounded,filled",
         fillcolor="#f2f2f2"];
  edge  [fontname="Arial", fontsize=10, penwidth=2];

  // --- Start filters (left) ---
  In   [label="Input Source\n(RSS / Dataset)"];
  F1   [label="Filter:\nFetch / Load"];
  F2   [label="Filter:\nExtract Text"];

  In -> F1 -> F2;

  // --- Split into 3 parallel branches ---
  // Use invisible nodes to force cleaner alignment
  split [shape=point, width=0.01, label=""];
  F2 -> split;

  // Branch A (top): Normalize -> Tokenize
  A1 [label="Filter:\nNormalize"];
  A2 [label="Filter:\nTokenize\n(sentences/words)"];

  // Branch B (middle): Vectorize -> Similarity
  B1 [label="Filter:\nVectorize\n(TF-IDF)"];
  B2 [label="Filter:\nSimilarity Graph\n(cosine matrix)"];

  // Branch C (bottom): Quality checks -> Clean-up
  C1 [label="Filter:\nQuality Checks\n(drop short/junk)"];
  C2 [label="Filter:\nDe-dup / Clean"];

  split -> A1;
  split -> B1;
  split -> C1;

  A1 -> A2;
  B1 -> B2;
  C1 -> C2;

  // --- Merge back ---
  merge [shape=point, width=0.01, label=""];
  A2 -> merge;
  B2 -> merge;
  C2 -> merge;

  // --- Final stage (right) ---
  F3 [label="Filter:\nTextRank\n(PageRank scores)"];
  F4 [label="Filter:\nMMR Selection\n(diversity)"];
  Out [label="Output:\nSummary"];
  Persist [label="Persist:\nCSV / JSON"];

  merge -> F3 -> F4 -> Out -> Persist;

  // Optional: keep branch ordering visually consistent
  { rank=same; A1; B1; C1; }
  { rank=same; A2; B2; C2; }
}
"""

    def component_interaction_dot(self) -> str:
        # Component interaction view (who calls what)
        return r"""
digraph Component_Interaction {
  rankdir=TB;
  graph [fontsize=12, fontname="Arial", labelloc="t",
         label="Component Interaction (Runner → Modules → Artifacts)"];
  node  [fontname="Arial", fontsize=10, shape=box, style="rounded,filled", fillcolor="#e7f0ff"];
  edge  [fontname="Arial", fontsize=10];

  Runner [label="Runner\nrun_iteration1 / run_cnndm_training_eval"];
  Pipe   [label="NewsDataPipeline\nbuild_articles()"];
  Loader [label="CNNDailyMailDatasetLoader\nload(split, limit, shuffle, seed)"];
  Train  [label="SummarizerTrainer\ntune(texts, refs)"];
  Model  [label="TextRankMMRSummarizer\nsummarize(text, k)"];
  Eval   [label="RougeEvaluator\nevaluate(preds, refs)"];

  FS1 [label="outputs/summaries.csv", fillcolor="#fff3d6"];
  FS2 [label="outputs/best_config_cnndm.json", fillcolor="#fff3d6"];
  FS3 [label="outputs/eval_report_cnndm.json", fillcolor="#fff3d6"];
  DGS [label="docs/diagrams/*", fillcolor="#fff3d6"];

  Runner -> Pipe;
  Pipe -> Model;
  Model -> FS1;

  Runner -> Loader;
  Loader -> Train;
  Train -> Model;
  Train -> FS2;

  Model -> Eval;
  Eval -> FS3;

  Runner -> DGS [label="DiagramGenerator.generate_all()"];
}
"""

    # -----------------------------
    # Public API
    # -----------------------------

    def generate_all(self, also_write_mermaid: bool = False) -> Dict[str, str]:
        """
        Generates DOT + PNG + SVG into docs/diagrams.
        Returns a mapping of artifact keys -> paths.

        also_write_mermaid:
          If True, also writes Mermaid (.mmd) placeholders so you can paste into draw.io.
          (You can keep False if you only want Graphviz outputs.)
        """
        self._ensure_dir()

        outputs: Dict[str, str] = {}

        diagrams = {
            "ai_capability_architecture": self.ai_capability_architecture_dot(),
            "data_flow_pipes_filters": self.data_flow_pipes_filters_dot(),
            "component_interaction": self.component_interaction_dot(),
        }

        for name, dot_text in diagrams.items():
            # write DOT
            dot_path = self._write_text(f"{name}.dot", dot_text)
            outputs[f"{name}_dot"] = dot_path

            # render PNG + SVG
            rendered = self._render_dot(name, dot_text)
            outputs[f"{name}_png"] = rendered["png"]
            outputs[f"{name}_svg"] = rendered["svg"]

        if also_write_mermaid:
            # Minimal Mermaid versions (optional). Keeps your old workflow available.
            for name in diagrams.keys():
                mmd_path = self._write_text(
                    f"{name}.mmd",
                    f"%% Mermaid version not generated for {name} in Graphviz-first mode.\n"
                    f"%% Use the DOT/PNG/SVG outputs in this folder.\n"
                    f"flowchart LR\n  A[{name}] --> B[See {name}.png]\n"
                )
                outputs[f"{name}_mmd"] = mmd_path

        return outputs