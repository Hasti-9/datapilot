"""RAG over the analysis output (profile, cleaning log, findings, model results)
plus any uploaded notes. TF-IDF retrieval keeps it local - no embedding API."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .llm import LLM, get_llm


@dataclass
class Doc:
    id: int
    source: str
    text: str


def chunk_text(text: str, source: str, size: int = 700, overlap: int = 120) -> list[tuple[str, str]]:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 2 <= size:
            buf = f"{buf}\n\n{p}" if buf else p
            continue
        if buf:
            chunks.append(buf)
        while len(p) > size:
            chunks.append(p[:size])
            p = p[size - overlap:]
        buf = p
    if buf:
        chunks.append(buf)
    return [(source, c) for c in chunks]


class KnowledgeBase:
    def __init__(self):
        self.docs: list[Doc] = []
        self._vec: TfidfVectorizer | None = None
        self._mat = None

    def add(self, text: str, source: str) -> None:
        for src, chunk in chunk_text(text, source):
            self.docs.append(Doc(len(self.docs), src, chunk))
        self._vec = None

    def _index(self) -> None:
        self._vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                    token_pattern=r"(?u)\b\w[\w.]*\b")
        self._mat = self._vec.fit_transform([f"{d.source} {d.text}".replace("_", " ") for d in self.docs])

    def search(self, query: str, k: int = 5) -> list[tuple[Doc, float]]:
        if not self.docs:
            return []
        if self._vec is None:
            self._index()
        qv = self._vec.transform([query.replace("_", " ")])
        sims = cosine_similarity(qv, self._mat).ravel()
        order = np.argsort(-sims)[:k]
        return [(self.docs[i], float(sims[i])) for i in order if sims[i] > 0]


def build_knowledge_base(state: dict, extra_docs: dict[str, str] | None = None) -> KnowledgeBase:
    kb = KnowledgeBase()
    prof = state.get("eda", {}).get("profile")
    if prof:
        kb.add(f"Dataset overview: {prof['n_rows']} rows, {prof['n_cols']} columns.", "overview")
        for c in prof["columns"]:
            parts = [f"Column {c['name']}: type {c['kind']} ({c['dtype']}), "
                     f"{c['missing_pct']}% missing, {c['unique']} unique values."]
            if c["kind"] == "numeric":
                parts.append(f"mean {c['mean']:.4g}, median {c['median']:.4g}, std {c['std']:.4g}, "
                             f"min {c['min']:.4g}, max {c['max']:.4g}, skew {c['skew']:.2f}.")
            if c.get("top_values"):
                parts.append("Most common values: " +
                             ", ".join(f"{k} ({v})" for k, v in c["top_values"].items()) + ".")
            kb.add(" ".join(parts), f"column:{c['name']}")
    if state.get("cleaning_log"):
        kb.add("Data cleaning steps applied:\n\n" + state["cleaning_log"], "cleaning")
    for f in state.get("eda", {}).get("findings", []):
        kb.add(f, "eda finding")
    for p in state.get("eda", {}).get("correlated_pairs", []):
        kb.add(f"Correlation between {p['a']} and {p['b']}: r = {p['r']}.", "correlation")
    ml = state.get("ml")
    if ml is not None:
        kb.add(ml.summary(), "model results")
        fi = ml.feature_importance.head(10)
        kb.add("Feature importance (permutation, best model):\n" +
               "\n".join(f"{r.feature}: {r.importance:.4f}" for r in fi.itertuples()), "feature importance")
    if state.get("insights"):
        kb.add(state["insights"], "insight report")
    for name, text in (extra_docs or {}).items():
        kb.add(text, f"doc:{name}")
    return kb


class RAGAssistant:
    def __init__(self, kb: KnowledgeBase, llm: LLM | None = None):
        self.kb = kb
        self.llm = llm or get_llm()

    def answer(self, question: str, history: list[dict] | None = None, k: int = 6) -> dict:
        hits = self.kb.search(question, k=k)
        context = "\n\n".join(f"[{i + 1}] ({d.source}) {d.text}" for i, (d, _) in enumerate(hits))
        sources = [{"n": i + 1, "source": d.source, "text": d.text, "score": round(s, 3)}
                   for i, (d, s) in enumerate(hits)]
        if not hits:
            return {"answer": "I couldn't find anything relevant in the analysis.", "sources": []}
        if not self.llm.available:
            bullet = "\n".join(f"- {d.text}" for d, _ in hits[:4])
            return {"answer": f"Most relevant facts from the analysis:\n{bullet}", "sources": sources}
        convo = ""
        for turn in (history or [])[-6:]:
            convo += f"{turn['role'].upper()}: {turn['content']}\n"
        system = (
            "You are DataPilot, a data scientist explaining an analysis to a non-technical "
            "stakeholder. Answer only from the numbered context. Cite facts as [n]. "
            "Use plain language, explain what numbers mean in practice, and mention caveats "
            "(correlation vs causation, sample size, missing data) when relevant. If the context "
            "doesn't cover the question, say so and suggest a SQL question or analysis that would."
        )
        prompt = f"Context:\n{context}\n\nConversation so far:\n{convo}\nQuestion: {question}"
        res = self.llm.complete(system, prompt, max_tokens=4000)
        return {"answer": res.text if res.ok else f"(LLM unavailable: {res.error})", "sources": sources}
