"""Natural-language-to-SQL over the loaded dataset using DuckDB."""
from __future__ import annotations

import re
from dataclasses import dataclass

import duckdb
import pandas as pd

from .eda import column_kind
from .llm import LLM, get_llm

TABLE = "data"
FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|copy|export|import|install|load|"
    r"pragma|set|call|replace|truncate|grant|revoke)\b|read_csv|read_parquet|read_json|glob\(",
    re.I,
)


@dataclass
class SQLResult:
    question: str
    sql: str
    result: pd.DataFrame | None
    error: str | None = None
    explanation: str = ""
    method: str = "llm"


class SQLEngine:
    def __init__(self, df: pd.DataFrame, llm: LLM | None = None):
        self.df = df
        self.llm = llm or get_llm()
        self.con = duckdb.connect(":memory:")
        self.con.register(TABLE, df)
        # Prevent queries from touching the filesystem or network.
        self.con.execute("SET enable_external_access = false")

    def schema_text(self, sample_rows: int = 3) -> str:
        lines = [f"Table `{TABLE}` ({len(self.df):,} rows):"]
        for c in self.df.columns:
            s = self.df[c]
            kind = column_kind(s)
            extra = ""
            if kind in ("categorical", "boolean"):
                vals = s.dropna().astype("string").unique()[:8]
                extra = " values e.g. " + ", ".join(repr(str(v)) for v in vals)
            elif kind == "numeric":
                extra = f" range {s.min():.4g}..{s.max():.4g}"
            elif kind == "datetime":
                extra = f" range {s.min()}..{s.max()}"
            lines.append(f"  - {c} ({s.dtype}, {kind}){extra}")
        lines.append("Sample rows:\n" + self.df.head(sample_rows).to_csv(index=False))
        return "\n".join(lines)

    @staticmethod
    def validate(sql: str) -> str | None:
        stripped = sql.strip().rstrip(";")
        if ";" in stripped:
            return "Only a single statement is allowed."
        if not re.match(r"^\s*(with|select)\b", stripped, re.I):
            return "Only SELECT queries are allowed."
        if FORBIDDEN.search(stripped):
            return "Query contains a disallowed keyword."
        return None

    def run(self, sql: str, limit: int = 1000) -> pd.DataFrame:
        err = self.validate(sql)
        if err:
            raise ValueError(err)
        sql = sql.strip().rstrip(";")
        return self.con.execute(f"SELECT * FROM ({sql}) AS q LIMIT {limit}").df()

    def ask(self, question: str, max_attempts: int = 3) -> SQLResult:
        if not self.llm.available:
            return self._heuristic(question)
        system = (
            "You translate analyst questions into DuckDB SQL over a single table. "
            "Use only the columns listed. Quote identifiers with double quotes when needed. "
            "Return JSON: {\"sql\": \"...\", \"explanation\": \"one sentence on what the query does\"}. "
            "Write exactly one read-only SELECT (CTEs allowed). Prefer readable column aliases, "
            "ORDER BY for rankings and LIMIT for top-N questions."
        )
        prompt = f"{self.schema_text()}\n\nQuestion: {question}"
        last_err, sql, expl = None, "", ""
        for _ in range(max_attempts):
            out = self.llm.complete_json(system, prompt)
            if not out or "sql" not in out:
                last_err = "The model did not return a SQL query."
                break
            sql, expl = out["sql"], out.get("explanation", "")
            try:
                res = self.run(sql)
                return SQLResult(question, sql, res, None, expl, "llm")
            except Exception as exc:  # self-correct with the error message
                last_err = str(exc)
                prompt += f"\n\nYour previous query:\n{sql}\nfailed with: {last_err}\nFix it."
        heuristic = self._heuristic(question)
        if heuristic.error is None:
            return heuristic
        return SQLResult(question, sql, None, last_err, expl, "llm")

    # rule-based fallback when no LLM is configured
    def _match_col(self, text: str) -> list[str]:
        t = text.lower()
        found = []
        for c in sorted(self.df.columns, key=len, reverse=True):
            variants = {c.lower(), c.lower().replace("_", " ")}
            if any(re.search(rf"\b{re.escape(v)}\b", t) for v in variants):
                found.append(c)
                t = t.replace(c.lower().replace("_", " "), " ").replace(c.lower(), " ")
        # Preserve order of appearance in the question.
        q = text.lower()
        return sorted(found, key=lambda c: min(
            [i for i in (q.find(c.lower()), q.find(c.lower().replace("_", " "))) if i >= 0] or [10**9]))

    def _heuristic(self, question: str) -> SQLResult:
        q = question.lower()
        cols = self._match_col(question)
        nums = [c for c in cols if column_kind(self.df[c]) == "numeric"]
        cats = [c for c in cols if c not in nums]
        agg = ("AVG" if re.search(r"average|mean|avg", q) else
               "SUM" if re.search(r"total|sum", q) else
               "MAX" if re.search(r"max|highest|largest", q) else
               "MIN" if re.search(r"min|lowest|smallest", q) else None)
        top = re.search(r"top\s+(\d+)", q)
        sql = None
        if re.search(r"how many|count|number of", q) and not nums:
            sql = (f'SELECT "{cats[0]}", COUNT(*) AS n FROM {TABLE} GROUP BY 1 ORDER BY n DESC'
                   if cats else f"SELECT COUNT(*) AS n FROM {TABLE}")
        elif nums and (cats or " by " in q or " per " in q):
            grp = cats[0] if cats else (nums[1] if len(nums) > 1 else None)
            if grp:
                a = agg or "AVG"
                sql = (f'SELECT "{grp}", {a}("{nums[0]}") AS {a.lower()}_{nums[0]} FROM {TABLE} '
                       f"GROUP BY 1 ORDER BY 2 DESC")
        elif nums and agg:
            sql = f'SELECT {agg}("{nums[0]}") AS {agg.lower()}_{nums[0]} FROM {TABLE}'
        elif nums and top:
            sql = f'SELECT * FROM {TABLE} ORDER BY "{nums[0]}" DESC'
        if sql and top:
            sql += f" LIMIT {int(top.group(1))}"
        if not sql:
            return SQLResult(question, "", None,
                             "Couldn't map this question to SQL without an LLM. Mention column names, "
                             "e.g. 'average price by region'.", method="heuristic")
        try:
            return SQLResult(question, sql, self.run(sql), None, "Rule-based translation.", "heuristic")
        except Exception as exc:
            return SQLResult(question, sql, None, str(exc), method="heuristic")
