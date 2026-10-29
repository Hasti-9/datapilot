from html import escape
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from datapilot.graph import DataPilot
from datapilot.llm import get_llm
from datapilot.ml import train_and_compare
from datapilot.viz import ACCENT, SERIES, chart_for_result, hbar, heatmap, style

SAMPLE = "sample_data/customer_churn.csv"
PLOT_CFG = {"displayModeBar": False}

st.set_page_config(page_title="DataPilot", layout="wide")
st.markdown(f"<style>{Path('assets/style.css').read_text()}</style>", unsafe_allow_html=True)

llm = get_llm()


def html(s: str) -> None:
    st.markdown(s, unsafe_allow_html=True)


def label(text: str) -> None:
    html(f'<div class="dp-label">{escape(text)}</div>')


def chart(fig, key: str | None = None) -> None:
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG, key=key)


with st.sidebar:
    html('<div class="dp-brand"><i></i>DataPilot</div>')

    st.markdown("### Dataset")
    upload = st.file_uploader("CSV file", type=["csv", "txt"], label_visibility="collapsed")
    use_sample = st.toggle("Use sample churn data", value=upload is None)

    preview = None
    if upload is not None:
        preview = pd.read_csv(upload, sep=None, engine="python", nrows=200)
        upload.seek(0)
    elif use_sample:
        preview = pd.read_csv(SAMPLE, nrows=200)

    st.markdown("### Model")
    targets = ["Auto-detect", "None"] + (list(preview.columns) if preview is not None else [])
    target_choice = st.selectbox("Target column", targets)

    with st.expander("Advanced"):
        clip = st.checkbox("Clip outliers (1.5×IQR)")
        thresh = st.slider("Drop columns missing more than", 0.3, 0.95, 0.6, 0.05)
        folds = st.slider("CV folds", 2, 10, 5)
        docs = st.file_uploader("Notes for the assistant", type=["txt", "md"], accept_multiple_files=True)

    st.write("")
    run = st.button("Run analysis", type="primary", use_container_width=True, disabled=preview is None)

    st.write("")
    if llm.available:
        html(f'<div class="dp-status"><i style="background:#0ca30c"></i>Groq · {escape(llm.model)}</div>')
    else:
        html('<div class="dp-status"><i style="background:#c9c8c3"></i>Offline. Set GROQ_API_KEY for '
             'generated SQL &amp; answers</div>')

if run:
    source = upload.getvalue() if upload is not None else SAMPLE
    target = {"Auto-detect": "auto", "None": None}.get(target_choice, target_choice)
    notes = {d.name: d.getvalue().decode("utf-8", "ignore") for d in (docs or [])}
    with st.spinner("Cleaning, profiling and training models..."):
        st.session_state.pilot = DataPilot(source, target, clip_outliers=clip,
                                           drop_missing_threshold=thresh, cv_folds=folds, docs=notes)
    st.session_state.chat = []

pilot: DataPilot | None = st.session_state.get("pilot")

html('<p class="dp-eyebrow">Automated analysis</p><h1 class="dp-title">DataPilot</h1>'
     '<p class="dp-sub">Drop in a CSV. Get it cleaned, profiled, modelled, and queryable in '
     'plain English.</p>')

if pilot is None:
    html("""
    <div class="dp-steps">
      <div class="dp-step"><div class="i">01</div><div class="t">Load</div>
        <div class="d">Upload a CSV or use the sample churn dataset from the sidebar.</div></div>
      <div class="dp-step"><div class="i">02</div><div class="t">Analyse</div>
        <div class="d">Cleaning, profiling, charts and a model comparison run as one graph.</div></div>
      <div class="dp-step"><div class="i">03</div><div class="t">Ask</div>
        <div class="d">Query the table in plain English or ask why the model behaves as it does.</div></div>
    </div>""")
    if preview is not None:
        label("Preview")
        st.dataframe(preview.head(12), use_container_width=True, hide_index=True)
    st.stop()

S = pilot.state
df = S["df"]
prof = S["eda"]["profile"]
ml = S.get("ml")
removed = len(S["raw_df"]) - prof["n_rows"]
dropped = S["raw_df"].shape[1] - prof["n_cols"]
score = ""
if ml is not None:
    m = "test_f1" if ml.task == "classification" else "test_r2"
    score = f"{m.replace('test_', '').upper()} {ml.leaderboard.iloc[0][m]:.2f}"


def stat(k, v, n=""):
    return (f'<div class="dp-stat"><div class="k">{escape(k)}</div><div class="v">{escape(str(v))}</div>'
            f'<div class="n">{escape(n) or "&nbsp;"}</div></div>')


html('<div class="dp-stats">'
     + stat("Rows", f"{prof['n_rows']:,}", f"{removed} removed" if removed else "")
     + stat("Columns", prof["n_cols"], f"{dropped} dropped" if dropped else "")
     + stat("Correlations", len(S["eda"]["correlated_pairs"]), "|r| ≥ 0.5")
     + stat("Target", S.get("target") or "-", ml.task if ml else "")
     + stat("Best model", ml.best_model_name if ml else "-", score)
     + "</div>")

tabs = st.tabs(["Summary", "Cleaning", "Profile", "Charts", "Models", "SQL", "Ask"], key="main_tabs")

with tabs[0]:
    left, right = st.columns([2.2, 1], gap="large")
    with left:
        st.markdown(S["insights"])
    with right:
        label("Pipeline")
        rows = []
        for line in S["trace"]:
            node, _, msg = line.partition(":")
            rows.append(f"<div><b>{escape(node)}</b> · {escape(msg.strip())}</div>")
        html(f'<div class="dp-trace">{"".join(rows)}</div>')

with tabs[1]:
    steps = S["cleaning_log"].splitlines()
    html(f'<p>{escape(steps[0])}</p>')
    st.markdown("\n".join(steps[1:]))
    label("Before")
    st.dataframe(S["raw_df"].head(8), use_container_width=True, hide_index=True)
    label("After")
    st.dataframe(df.head(8), use_container_width=True, hide_index=True)
    st.download_button("Download cleaned CSV", df.to_csv(index=False), "cleaned.csv", "text/csv")

with tabs[2]:
    label("Findings")
    for f in S["eda"]["findings"]:
        st.markdown(f"- {f}")
    label("Columns")
    table = pd.DataFrame(prof["columns"]).drop(columns=["top_values"], errors="ignore")
    st.dataframe(table, use_container_width=True, hide_index=True)

with tabs[3]:
    charts = S["charts"]
    for i in range(0, len(charts), 2):
        row = st.columns(2, gap="medium")
        for col, (title, fig) in zip(row, charts[i:i + 2]):
            with col:
                chart(fig, key=f"auto_{i}_{title}")

    label("Explore")
    c1, c2, c3, c4 = st.columns(4)
    kind = c1.selectbox("Type", ["scatter", "histogram", "box", "bar", "line"])
    x = c2.selectbox("X", df.columns)
    y = c3.selectbox("Y", ["-"] + list(df.columns))
    color = c4.selectbox("Color by", ["-"] + list(df.columns))
    kw = {"x": x}
    if y != "-":
        kw["y"] = y
    if color != "-":
        kw["color"] = color
        kw["color_discrete_sequence"] = SERIES
    else:
        kw["color_discrete_sequence"] = [ACCENT]
    try:
        chart(style(getattr(px, kind)(df, **kw)))
    except Exception as e:
        st.info(f"Can't plot that combination: {e}")

with tabs[4]:
    if S.get("ml_error"):
        st.error(S["ml_error"])
    if ml is None:
        st.info("No model trained. Pick a target in the sidebar or train one below.")
    else:
        d = ml.details
        html(f'<p>{escape(ml.task.capitalize())} on <code>{escape(ml.target)}</code> · '
             f'{d["n_train"]:,} train / {d["n_test"]:,} test rows · {d["cv_folds"]}-fold cross-validation</p>')
        label("Leaderboard")
        st.dataframe(ml.leaderboard, use_container_width=True, hide_index=True)
        st.write("")
        left, right = st.columns(2, gap="medium")
        with left:
            chart(hbar(ml.feature_importance.head(10), "importance", "feature",
                       f"Feature importance · {ml.best_model_name}").update_yaxes(title_text=""))
        with right:
            if ml.task == "classification":
                labels = [str(v) for v in d["labels"]]
                chart(heatmap(d["confusion_matrix"], labels, labels, "Confusion matrix",
                              labels=dict(x="Predicted", y="Actual")).update_coloraxes(showscale=False))
            else:
                pv = pd.DataFrame({"actual": d["y_test"], "predicted": d["y_pred"]})
                chart(style(px.scatter(pv, x="actual", y="predicted", opacity=.55,
                                       color_discrete_sequence=[ACCENT]), "Predicted vs actual"))

    st.write("")
    with st.expander("Train on another column"):
        new_target = st.selectbox("Target", df.columns, key="retarget")
        if st.button("Train"):
            with st.spinner("Training..."):
                try:
                    S["ml"], S["ml_error"] = train_and_compare(df, new_target), None
                except Exception as e:
                    S["ml_error"] = str(e)
            st.rerun()

with tabs[5]:
    q = st.text_input("Question", placeholder="Average monthly charges by contract type",
                      label_visibility="collapsed")
    if q:
        with st.spinner("Writing query..."):
            res = pilot.sql_engine.ask(q)
        if res.sql:
            st.code(res.sql, language="sql")
        if res.error:
            st.info(res.error)
        else:
            left, right = st.columns([1, 1.3], gap="medium")
            left.dataframe(res.result, use_container_width=True, hide_index=True)
            fig = chart_for_result(res.result)
            if fig is not None:
                with right:
                    chart(fig)
    st.write("")
    with st.expander("Write SQL"):
        manual = st.text_area("Table name is `data`", "SELECT * FROM data LIMIT 10")
        if st.button("Run query"):
            try:
                st.dataframe(pilot.sql_engine.run(manual), use_container_width=True, hide_index=True)
            except Exception as e:
                st.info(str(e))

with tabs[6]:
    if not st.session_state.get("chat"):
        html('<p>Ask about the data, the cleaning, or the model. Questions that need numbers are '
             'answered with SQL; the rest come from the analysis, with sources.</p>')
    for turn in st.session_state.get("chat", []):
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("sql"):
                st.code(turn["sql"], language="sql")

    prompt = st.chat_input("What drives churn?")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner(""):
                out = pilot.ask(prompt)
            html(f'<span class="dp-pill">{escape(out["route"])}</span>')
            st.markdown(out["answer"])
            sql_text = None
            if out.get("sql") is not None and out["sql"].result is not None:
                sql_text = out["sql"].sql
                st.code(sql_text, language="sql")
                st.dataframe(out["sql"].result.head(50), use_container_width=True, hide_index=True)
            if out.get("chart") is not None:
                chart(out["chart"])
            if out.get("sources"):
                with st.expander(f"Sources ({len(out['sources'])})"):
                    html("".join(f'<div class="dp-src"><span>{s["n"]:02d}</span>{escape(s["source"])} '
                                 f'· {escape(s["text"][:220])}</div>' for s in out["sources"]))
        st.session_state.chat += [{"role": "user", "content": prompt},
                                  {"role": "assistant", "content": out["answer"], "sql": sql_text}]
