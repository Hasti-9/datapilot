"""LangGraph workflows.

analysis:  ingest -> clean -> eda -> visualize -> [train_models] -> build_kb -> insights
chat:      route -> sql | explain | rag
"""
from __future__ import annotations

import io
import json
import re
from typing import Any, Literal, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from .cleaning import clean_dataframe
from .eda import run_eda
from .llm import get_llm
from .ml import suggest_target, train_and_compare
from .rag import RAGAssistant, build_knowledge_base
from .sql import SQLEngine
from .viz import auto_charts, chart_for_result


# analysis
class AnalysisState(TypedDict, total=False):
    source: Any                 # path, bytes or DataFrame
    raw_df: pd.DataFrame
    df: pd.DataFrame
    options: dict
    target: str | None
    cleaning_log: str
    eda: dict
    charts: list
    ml: Any
    ml_error: str | None
    kb: Any
    insights: str
    trace: list[str]


def _log(state: AnalysisState, msg: str) -> list[str]:
    return [*state.get("trace", []), msg]


def ingest(state: AnalysisState) -> AnalysisState:
    src = state["source"]
    if isinstance(src, pd.DataFrame):
        df = src
    elif isinstance(src, (bytes, bytearray)):
        df = pd.read_csv(io.BytesIO(src), sep=None, engine="python")
    else:
        df = pd.read_csv(src, sep=None, engine="python")
    return {"raw_df": df, "trace": _log(state, f"ingest: loaded {df.shape[0]} rows x {df.shape[1]} cols")}


def clean(state: AnalysisState) -> AnalysisState:
    opts = state.get("options", {})
    df, rep = clean_dataframe(state["raw_df"],
                              drop_missing_threshold=opts.get("drop_missing_threshold", 0.6),
                              clip_outliers=opts.get("clip_outliers", False))
    target = state.get("target")
    if target == "auto":
        target = suggest_target(df)
    elif target and target not in df.columns:
        # Map a user-supplied raw name onto the snake_cased column.
        target = re.sub(r"[^0-9a-zA-Z]+", "_", target).strip("_").lower()
        target = target if target in df.columns else None
    return {"df": df, "cleaning_log": rep.summary(), "target": target,
            "trace": _log(state, f"clean: {len(rep.steps)} steps")}


def eda(state: AnalysisState) -> AnalysisState:
    res = run_eda(state["df"])
    return {"eda": res, "trace": _log(state, f"eda: {len(res['findings'])} findings")}


def visualize(state: AnalysisState) -> AnalysisState:
    charts = auto_charts(state["df"], state["eda"], state.get("target"))
    return {"charts": charts, "trace": _log(state, f"visualize: {len(charts)} charts")}


def should_train(state: AnalysisState) -> Literal["train_models", "build_kb"]:
    return "train_models" if state.get("options", {}).get("run_ml", True) and state.get("target") else "build_kb"


def train_models(state: AnalysisState) -> AnalysisState:
    try:
        res = train_and_compare(state["df"], state["target"],
                                cv_folds=state.get("options", {}).get("cv_folds", 5))
        return {"ml": res, "ml_error": None,
                "trace": _log(state, f"ml: {res.task}, best={res.best_model_name}")}
    except Exception as exc:
        return {"ml": None, "ml_error": str(exc), "trace": _log(state, f"ml: failed ({exc})")}


def build_kb(state: AnalysisState) -> AnalysisState:
    kb = build_knowledge_base(state, state.get("options", {}).get("docs"))
    return {"kb": kb, "trace": _log(state, f"build_kb: {len(kb.docs)} chunks")}


def insights(state: AnalysisState) -> AnalysisState:
    findings = state["eda"]["findings"]
    llm = get_llm()
    ml = state.get("ml")
    fallback = "### Key findings\n" + "\n".join(f"- {f}" for f in findings)
    if ml is not None:
        top = ", ".join(ml.feature_importance.head(3)["feature"])
        fallback += (f"\n\n### Model\n- Best model for `{ml.target}` ({ml.task}): **{ml.best_model_name}**."
                     f"\n- Most influential features: {top}.")
    if not llm.available:
        text = fallback
    else:
        payload = {
            "profile": {k: v for k, v in state["eda"]["profile"].items() if k != "columns"},
            "columns": state["eda"]["profile"]["columns"],
            "cleaning": state.get("cleaning_log"),
            "findings": findings,
            "outliers": state["eda"]["outliers"],
            "model": ml.summary() if ml is not None else state.get("ml_error"),
            "feature_importance": (ml.feature_importance.head(8).round(4).to_dict("records")
                                   if ml is not None else None),
        }
        system = ("You are a senior data scientist writing for business stakeholders. Produce a "
                  "concise Markdown report with sections: Executive summary (3 bullets), What the "
                  "data looks like, Key patterns, Model performance (if present - interpret metrics "
                  "against the baseline in plain words), Data quality caveats, Recommended next "
                  "steps. Only state facts supported by the provided analysis.")
        res = llm.complete(system, json.dumps(payload, default=str), max_tokens=6000)
        text = res.text if res.ok else fallback
    return {"insights": text, "trace": _log(state, "insights: report written")}


def build_analysis_graph():
    g = StateGraph(AnalysisState)
    for name, fn in [("ingest", ingest), ("clean", clean), ("eda", eda), ("visualize", visualize),
                     ("train_models", train_models), ("build_kb", build_kb), ("insights", insights)]:
        g.add_node(name, fn)
    g.add_edge(START, "ingest")
    g.add_edge("ingest", "clean")
    g.add_edge("clean", "eda")
    g.add_edge("eda", "visualize")
    g.add_conditional_edges("visualize", should_train, ["train_models", "build_kb"])
    g.add_edge("train_models", "build_kb")
    g.add_edge("build_kb", "insights")
    g.add_edge("insights", END)
    return g.compile()


def run_analysis(source, target: str | None = "auto", **options) -> AnalysisState:
    """target: a column name, "auto" to guess one, or None to skip modelling."""
    final = build_analysis_graph().invoke(
        {"source": source, "target": target, "options": options, "trace": []})
    final["kb"].add(final["insights"], "insight report")
    return final


# chat
class ChatState(TypedDict, total=False):
    question: str
    history: list[dict]
    route: str
    answer: str
    sql: Any
    chart: Any
    sources: list[dict]
    context: dict          # holds df, kb, ml from the analysis


SQL_HINTS = re.compile(r"\b(how many|count|average|avg|mean|sum|total|top \d+|list|show|which|"
                       r"max|min|highest|lowest|per|group|by|between|filter|where|rows)\b", re.I)
EXPLAIN_HINTS = re.compile(r"\b(model|predict|feature|importance|accuracy|f1|r2|rmse|driver|why)\b", re.I)


def route(state: ChatState) -> ChatState:
    q = state["question"]
    llm = get_llm()
    choice = None
    if llm.available:
        out = llm.complete_json(
            "Classify a question about a dataset. 'sql' = needs computing numbers/rows from the raw "
            "table; 'explain' = about the trained ML model, its metrics or feature importance; "
            "'rag' = about findings, data quality, cleaning, or general interpretation. "
            'Return {"route": "sql"|"explain"|"rag"}.',
            q, max_tokens=2000)
        choice = (out or {}).get("route")
    if choice not in ("sql", "explain", "rag"):
        if EXPLAIN_HINTS.search(q) and state["context"].get("ml") is not None:
            choice = "explain"
        elif SQL_HINTS.search(q):
            choice = "sql"
        else:
            choice = "rag"
    if choice == "explain" and state["context"].get("ml") is None:
        choice = "rag"
    return {"route": choice}


def sql_node(state: ChatState) -> ChatState:
    engine: SQLEngine = state["context"]["sql_engine"]
    res = engine.ask(state["question"])
    if res.error:
        # Fall back to the knowledge base rather than failing the turn.
        rag = RAGAssistant(state["context"]["kb"]).answer(state["question"], state.get("history"))
        return {"sql": res, "answer": f"_SQL couldn't answer this ({res.error}). From the analysis:_\n\n"
                                      + rag["answer"], "sources": rag["sources"]}
    llm = get_llm()
    preview = res.result.head(25).to_csv(index=False)
    if llm.available:
        out = llm.complete(
            "Explain a SQL result to a non-technical user in 2-4 sentences. Lead with the direct "
            "answer, then one notable detail. Do not restate the SQL.",
            f"Question: {state['question']}\nSQL: {res.sql}\nResult ({len(res.result)} rows):\n{preview}",
            max_tokens=1500)
        text = out.text if out.ok else ""
    else:
        text = ""
    if not text:
        text = f"Query returned {len(res.result)} row(s)."
        if res.result.shape == (1, 1):
            text = f"The answer is **{res.result.iat[0, 0]}**."
    return {"sql": res, "answer": text, "chart": chart_for_result(res.result), "sources": []}


def explain_node(state: ChatState) -> ChatState:
    ml = state["context"]["ml"]
    kb = state["context"]["kb"]
    out = RAGAssistant(kb).answer(f"{state['question']} (model results, feature importance)",
                                  state.get("history"))
    if not get_llm().available:
        out["answer"] = ml.summary()
    return {"answer": out["answer"], "sources": out["sources"]}


def rag_node(state: ChatState) -> ChatState:
    out = RAGAssistant(state["context"]["kb"]).answer(state["question"], state.get("history"))
    return {"answer": out["answer"], "sources": out["sources"]}


def build_chat_graph():
    g = StateGraph(ChatState)
    g.add_node("route", route)
    g.add_node("sql", sql_node)
    g.add_node("explain", explain_node)
    g.add_node("rag", rag_node)
    g.add_edge(START, "route")
    g.add_conditional_edges("route", lambda s: s["route"], ["sql", "explain", "rag"])
    for n in ("sql", "explain", "rag"):
        g.add_edge(n, END)
    return g.compile()


class DataPilot:
    """Convenience facade: run the analysis once, then chat with it."""

    def __init__(self, source, target: str | None = "auto", **options):
        self.state = run_analysis(source, target, **options)
        self.sql_engine = SQLEngine(self.state["df"])
        self.chat_graph = build_chat_graph()
        self.history: list[dict] = []

    def ask(self, question: str) -> ChatState:
        ctx = {"df": self.state["df"], "kb": self.state["kb"], "ml": self.state.get("ml"),
               "sql_engine": self.sql_engine}
        out = self.chat_graph.invoke({"question": question, "history": self.history, "context": ctx})
        self.history += [{"role": "user", "content": question},
                         {"role": "assistant", "content": out["answer"]}]
        return out
