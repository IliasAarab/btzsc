"""BTZSC Leaderboard — Gradio Space app."""

from __future__ import annotations

import glob
import json
import os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import gradio as gr
import gradio.themes as gr_themes
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_RESULTS_REPO = os.getenv("BTZSC_RESULTS_REPO", "btzsc/btzsc-results")

FAMILY_LABELS: dict[str, str] = {
    "embedding": "Embedding Model",
    "nli": "NLI Cross-Encoder",
    "reranker": "Reranker",
    "llm": "Instruction-tuned LLM",
}

FAMILY_COLORS: dict[str, str] = {
    "Embedding Model": "#4b7a9b",
    "NLI Cross-Encoder": "#9b6464",
    "Reranker": "#8b7355",
    "Instruction-tuned LLM": "#7b6b9b",
}

SORT_COLUMNS: dict[str, str] = {
    "Avg F1": "avg_f1",
    "Avg Acc": "avg_acc",
    "Topic": "topic",
    "Sentiment": "sentiment",
    "Intent": "intent",
    "Emotion": "emotion",
}

REQUIRED_DATASETS = [
    "amazonpolarity",
    "imdb",
    "appreviews",
    "yelpreviews",
    "rottentomatoes",
    "financialphrasebank",
    "emotiondair",
    "empathetic",
    "banking77",
    "biasframes_intent",
    "massive",
    "agnews",
    "yahootopics",
    "trueteacher",
    "manifesto",
    "capsotu",
    "biasframes_offensive",
    "biasframes_sex",
    "wikitoxic_toxicaggregated",
    "wikitoxic_obscene",
    "wikitoxic_threat",
    "wikitoxic_insult",
]

# ---------------------------------------------------------------------------
# Markdown content
# ---------------------------------------------------------------------------

HEADER_MD = f"""
# 🏆 BTZSC Leaderboard

**Benchmark for Zero-Shot Text Classification** across Cross-Encoders, Embedding Models, Rerankers and LLMs.

📄 [Paper](https://openreview.net/forum?id=IxMryAz2p3) &nbsp;|&nbsp;\
💻 [Eval Harness](https://github.com/btzsc/btzsc) &nbsp;|&nbsp;\
📊 [Results Dataset](https://huggingface.co/datasets/{DEFAULT_RESULTS_REPO}) &nbsp;|&nbsp;\
🤗 [How to Submit](https://huggingface.co/datasets/{DEFAULT_RESULTS_REPO}/blob/main/SUBMISSION.md)

**Primary metric:** Macro F1 &nbsp;|&nbsp; **22 datasets** &nbsp;|&nbsp; **4 task types** (Sentiment · Topic · Intent · Emotion)
"""

ABOUT_MD = """
## About BTZSC

BTZSC evaluates **genuine zero-shot** text classification — no labeled examples, no few-shot prompts.

| Task | # Datasets | Examples |
|------|-----------|----------|
| Sentiment | 6 | Amazon Polarity, IMDB, Yelp Reviews, … |
| Emotion | 2 | Emotion (DAIR), Empathetic Dialogues |
| Intent | 3 | BANKING77, MASSIVE, BiasFrames (intent) |
| Topic | 11 | AG News, Yahoo Topics, CapsOTU, … |

**Metrics:** Macro F1 (primary), Accuracy, Macro Precision, Macro Recall

**How to submit:** Install the `btzsc` eval harness, run your model, export with `--output-json`,\
and open a PR on the results dataset repo. See \
[SUBMISSION.md](https://huggingface.co/datasets/btzsc/btzsc-results/blob/main/SUBMISSION.md) for full instructions.
"""

# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------


def _family_badge(family: str) -> str:
    color = FAMILY_COLORS.get(family, "#6b7280")
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:12px;font-size:0.85em">{family}</span>'
    )


def _model_link(model_id: str) -> str:
    url = f"https://huggingface.co/{model_id}"
    short = model_id.split("/")[-1]
    return f'<a href="{url}" target="_blank" style="color:#60a5fa">{short}</a>'


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_results_from_local(local_root: Path) -> list[dict[str, Any]]:
    files = glob.glob(str(local_root / "results" / "**" / "*.json"), recursive=True)
    return [_read_json(Path(p)) for p in files]


def _load_results_from_hub(repo_id: str) -> list[dict[str, Any]]:
    repo_files = list_repo_files(repo_id=repo_id, repo_type="dataset")
    json_files = [f for f in repo_files if f.startswith("results/") and f.endswith(".json")]
    payloads: list[dict[str, Any]] = []
    for file_name in json_files:
        local_path = hf_hub_download(repo_id=repo_id, filename=file_name, repo_type="dataset")
        payloads.append(_read_json(Path(local_path)))
    return payloads


def load_payloads() -> list[dict[str, Any]]:
    local_results_root = os.getenv("BTZSC_RESULTS_LOCAL_DIR")
    if local_results_root:
        return _load_results_from_local(Path(local_results_root))
    return _load_results_from_hub(DEFAULT_RESULTS_REPO)


# ---------------------------------------------------------------------------
# DataFrame builders
# ---------------------------------------------------------------------------

EMPTY_COLS = [
    "Rank",
    "Model",
    "Family",
    "Params",
    "Avg F1",
    "Avg Acc",
    "Topic",
    "Sentiment",
    "Intent",
    "Emotion",
    "Badge",
]


def _build_main_df(payloads: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        model = payload.get("model", {})
        result = payload.get("results", {})
        by_task = result.get("by_task", {})
        overall = result.get("overall", {})

        model_type = model.get("model_type", "unknown")
        if model_type not in FAMILY_LABELS:
            continue

        rows.append({
            "model_id": model.get("name", "unknown"),
            "family": FAMILY_LABELS[model_type],
            "params": model.get("params", "unknown"),
            "avg_f1": float(overall.get("macro_f1", 0.0)),
            "avg_acc": float(overall.get("accuracy", 0.0)),
            "topic": float(by_task.get("topic", {}).get("macro_f1", 0.0)),
            "sentiment": float(by_task.get("sentiment", {}).get("macro_f1", 0.0)),
            "intent": float(by_task.get("intent", {}).get("macro_f1", 0.0)),
            "emotion": float(by_task.get("emotion", {}).get("macro_f1", 0.0)),
        })

    if not rows:
        return pd.DataFrame(columns=EMPTY_COLS)
    return pd.DataFrame(rows)


def _build_per_dataset_df(
    payloads: list[dict[str, Any]],
    dataset_name: str,
) -> pd.DataFrame:
    """Build a per-model table for a single dataset (F1, Acc, Precision, Recall)."""
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        model = payload.get("model", {})
        by_dataset = payload.get("results", {}).get("by_dataset", {})
        ds_metrics = by_dataset.get(dataset_name, {})
        if not ds_metrics:
            continue
        family_raw = model.get("model_type", "")
        family_label = FAMILY_LABELS.get(family_raw) or family_raw
        rows.append({
            "Model": _model_link(model.get("name", "unknown")),
            "Family": _family_badge(family_label),
            "F1": f"{float(ds_metrics.get('macro_f1', 0.0)):.2f}",
            "Accuracy": f"{float(ds_metrics.get('accuracy', 0.0)):.2f}",
            "Precision": f"{float(ds_metrics.get('macro_precision', 0.0)):.2f}",
            "Recall": f"{float(ds_metrics.get('macro_recall', 0.0)):.2f}",
        })
    if not rows:
        return pd.DataFrame(columns=["Model", "Family", "F1", "Accuracy", "Precision", "Recall"])
    return pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Global state (mutable — refreshed by background thread)
# ---------------------------------------------------------------------------

ALL_PAYLOADS = load_payloads()
MAIN_DF = _build_main_df(ALL_PAYLOADS)
_LAST_LOADED = datetime.now(UTC)


def _auto_refresh(interval: int = 300) -> None:
    """Reload results from HF Hub/local every *interval* seconds."""
    global ALL_PAYLOADS, MAIN_DF, _LAST_LOADED  # noqa: PLW0603
    while True:
        time.sleep(interval)
        try:
            new = load_payloads()
            ALL_PAYLOADS = new
            MAIN_DF = _build_main_df(new)
            _LAST_LOADED = datetime.now(UTC)
        except Exception:  # noqa: BLE001, S110
            pass  # keep serving stale data rather than crashing


threading.Thread(target=_auto_refresh, daemon=True).start()


# ---------------------------------------------------------------------------
# Overall leaderboard render
# ---------------------------------------------------------------------------


def render_table(
    search: str,
    families: list[str],
    sort_label: str,
    order: str,
) -> pd.DataFrame:
    if MAIN_DF.empty:
        return pd.DataFrame(columns=EMPTY_COLS)

    selected = cast(pd.DataFrame, MAIN_DF.copy())

    if families:
        selected = selected[selected["family"].isin(families)]

    if search:
        search_lower = search.lower()
        model_mask = selected["model_id"].map(lambda v: search_lower in str(v).lower())
        selected = selected[model_mask]

    sort_col = SORT_COLUMNS[sort_label]
    descending = order == "Descending"
    selected = selected.sort_values(sort_col, ascending=not descending).reset_index(drop=True)

    selected["Rank"] = selected.index + 1
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    selected["Badge"] = selected["Rank"].map(lambda rank: medals.get(int(rank), ""))
    selected["Model"] = selected["model_id"].map(_model_link)
    selected["Family"] = selected["family"].map(_family_badge)

    display = cast(
        pd.DataFrame,
        selected[
            [
                "Rank",
                "Model",
                "Family",
                "params",
                "avg_f1",
                "avg_acc",
                "topic",
                "sentiment",
                "intent",
                "emotion",
                "Badge",
            ]
        ],
    )
    display = display.rename(
        columns={
            "params": "Params",
            "avg_f1": "Avg F1",
            "avg_acc": "Avg Acc",
            "topic": "Topic",
            "sentiment": "Sentiment",
            "intent": "Intent",
            "emotion": "Emotion",
        }
    )

    float_cols = ["Avg F1", "Avg Acc", "Topic", "Sentiment", "Intent", "Emotion"]
    for col in float_cols:
        display[col] = display[col].map(lambda x: f"{x:.2f}")

    return display


# ---------------------------------------------------------------------------
# Per-dataset render
# ---------------------------------------------------------------------------


def render_per_dataset(dataset_name: str) -> pd.DataFrame:
    return _build_per_dataset_df(ALL_PAYLOADS, dataset_name)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

FORCE_DARK_JS = """
() => {
    document.querySelector('body').classList.add('dark');
    document.querySelector('html').setAttribute('data-theme', 'dark');
}
"""

COLUMN_WIDTH_CSS = """
/* Widen Model and Family columns */
table.svelte-table th:nth-child(2),
table.svelte-table td:nth-child(2) { min-width: 220px; }
table.svelte-table th:nth-child(3),
table.svelte-table td:nth-child(3) { min-width: 200px; }
"""

with gr.Blocks(title="BTZSC Leaderboard") as demo:
    gr.Markdown(HEADER_MD)

    with gr.Tabs():
        # ---- Tab 1: Overall leaderboard ----
        with gr.TabItem("🏆 Overall"):
            with gr.Row():
                search = gr.Textbox(label="Search model", placeholder="Type model name…")
                sort = gr.Dropdown(
                    choices=list(SORT_COLUMNS.keys()),
                    value="Avg F1",
                    label="Sort by",
                )
                order = gr.Dropdown(
                    choices=["Descending", "Ascending"],
                    value="Descending",
                    label="Order",
                )

            family_choices = list(FAMILY_LABELS.values())
            families = gr.CheckboxGroup(
                choices=family_choices,
                value=family_choices,
                label="Model families",
            )

            leaderboard = gr.Dataframe(
                value=render_table("", family_choices, "Avg F1", "Descending"),
                label="Leaderboard",
                interactive=False,
                wrap=True,
                datatype="html",
            )

            search.change(render_table, [search, families, sort, order], leaderboard)
            sort.change(render_table, [search, families, sort, order], leaderboard)
            order.change(render_table, [search, families, sort, order], leaderboard)
            families.change(render_table, [search, families, sort, order], leaderboard)

        # ---- Tab 2: Per-dataset detail ----
        with gr.TabItem("📋 Per-Dataset"):
            dataset_dropdown = gr.Dropdown(
                choices=sorted(REQUIRED_DATASETS),
                value=REQUIRED_DATASETS[0],
                label="Dataset",
            )
            per_dataset_table = gr.Dataframe(
                value=render_per_dataset(REQUIRED_DATASETS[0]),
                label="Per-Dataset Results",
                interactive=False,
                wrap=True,
                datatype="html",
            )
            dataset_dropdown.change(
                fn=render_per_dataset,
                inputs=[dataset_dropdown],
                outputs=[per_dataset_table],
            )

        # ---- Tab 3: About ----
        with gr.TabItem("ℹ️ About"):  # noqa: RUF001
            gr.Markdown(ABOUT_MD)

    # ---- Footer ----
    gr.Markdown(
        f"*Last loaded: {_LAST_LOADED.strftime('%Y-%m-%d %H:%M UTC')} · "
        f"{len(ALL_PAYLOADS)} models evaluated · "
        f"Results sourced from "
        f"[{DEFAULT_RESULTS_REPO}](https://huggingface.co/datasets/{DEFAULT_RESULTS_REPO})*"
    )

if __name__ == "__main__":
    demo.launch(
        theme=gr_themes.Default(primary_hue="blue", neutral_hue="slate"),
        js=FORCE_DARK_JS,
        css=COLUMN_WIDTH_CSS,
    )
