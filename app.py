import pandas as pd
import plotly.express as px
import streamlit as st

from datapilot.graph import DataPilot
from datapilot.llm import get_llm
from datapilot.ml import train_and_compare
from datapilot.viz import ACCENT, chart_for_result, style

SAMPLE = "sample_data/customer_churn.csv"

st.set_page_config(page_title="DataPilot", layout="wide")
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono&display=swap');
  html, body, [class*="css"], .stMarkdown, .stText { font-family: 'IBM Plex Sans', sans-serif; }
  code, pre { font-family: 'IBM Plex Mono', monospace !important; }
  .block-container { padding-top: 2rem; max-width: 1200px; }
  h1 { font-weight: 600; letter-spacing: -0.02em; margin-bottom: 0 !important; }
  .subtitle { color: #5f6b66; margin-top: .2rem; margin-bottom: 1.5rem; }
  div[data-testid="stMetric"] {
      background: #fff; border: 1px solid #e3e6e2; border-radius: 6px; padding: .7rem .9rem; }
  div[data-testid="stMetricLabel"] p { font-size: .75rem; text-transform: uppercase;
      letter-spacing: .05em; color: #5f6b66; }
  div[data-testid="stMetricValue"] { font-size: 1.35rem; }
  .stTabs [data-baseweb="tab"] { font-weight: 500; }
  .pill { display:inline-block; font-size:.72rem; padding:.1rem .5rem; border-radius:999px;
          background:#e6f2f0; color:#0F766E; margin-bottom:.4rem; }
  #MainMenu, footer, [data-testid="stAppDeployButton"] { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

llm = get_llm()

st.title("DataPilot")
st.markdown('<p class="subtitle">Drop in a CSV and get it cleaned, profiled, modelled and queryable '
            'in plain English.</p>', unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Data")
    upload = st.file_uploader("CSV file", type=["csv", "txt"])
    use_sample = st.toggle("Use sample churn dataset", value=upload is None)
    docs = st.file_uploader("Notes / data dictionary (optional)", type=["txt", "md"],
                            accept_multiple_files=True,
                            help="Indexed so the assistant can reference it.")

    preview = None
    if upload is not None:
        preview = pd.read_csv(upload, sep=None, engine="python", nrows=200)
        upload.seek(0)
    elif use_sample:
        preview = pd.read_csv(SAMPLE, nrows=200)

    st.subheader("Settings")
    targets = ["Auto-detect", "None"] + (list(preview.columns) if preview is not None else [])
    target_choice = st.selectbox("Target column", targets)
    with st.expander("Advanced"):
        clip = st.checkbox("Clip outliers (1.5×IQR)")
        thresh = st.slider("Drop columns missing more than", 0.3, 0.95, 0.6, 0.05)
        folds = st.slider("CV folds", 2, 10, 5)

    run = st.button("Run analysis", type="primary", use_container_width=True,
                    disabled=preview is None)

    st.divider()
    if llm.available:
        st.caption(f"LLM: Groq · `{llm.model}`")
    else:
        st.caption("LLM: offline. Set `GROQ_API_KEY` for generated SQL and written answers. "
                   "Everything else works without it.")

if run:
    source = upload.getvalue() if upload is not None else SAMPLE
    target = {"Auto-detect": "auto", "None": None}.get(target_choice, target_choice)
    notes = {d.name: d.getvalue().decode("utf-8", "ignore") for d in (docs or [])}
    with st.status("Running analysis...", expanded=True) as status:
        pilot = DataPilot(source, target, clip_outliers=clip, drop_missing_threshold=thresh,
                          cv_folds=folds, docs=notes)
        for line in pilot.state["trace"]:
            st.write(line)
        status.update(label="Done", state="complete", expanded=False)
    st.session_state.pilot = pilot
    st.session_state.chat = []

pilot: DataPilot | None = st.session_state.get("pilot")
if pilot is None:
    st.info("Choose a dataset in the sidebar and press **Run analysis**.")
    if preview is not None:
        st.caption("Preview")
        st.dataframe(preview.head(15), use_container_width=True, hide_index=True)
    st.stop()

S = pilot.state
df = S["df"]
prof = S["eda"]["profile"]
ml = S.get("ml")

cols = st.columns(5)
cols[0].metric("Rows", f"{prof['n_rows']:,}", delta=f"{prof['n_rows'] - len(S['raw_df']):,}",
               delta_color="off")
cols[1].metric("Columns", prof["n_cols"], delta=prof["n_cols"] - S["raw_df"].shape[1], delta_color="off")
cols[2].metric("Strong correlations", len(S["eda"]["correlated_pairs"]))
cols[3].metric("Target", S.get("target") or "-")
cols[4].metric("Best model", ml.best_model_name if ml else "-")

tabs = st.tabs(["Summary", "Cleaning", "Profile", "Charts", "Models", "SQL", "Ask"], key="main_tabs")

with tabs[0]:
    st.markdown(S["insights"])

with tabs[1]:
    st.markdown(S["cleaning_log"])
    left, right = st.columns(2)
    left.caption("Raw")
    left.dataframe(S["raw_df"].head(30), use_container_width=True, hide_index=True)
    right.caption("Cleaned")
    right.dataframe(df.head(30), use_container_width=True, hide_index=True)
    st.download_button("Download cleaned CSV", df.to_csv(index=False), "cleaned.csv", "text/csv")

with tabs[2]:
    for f in S["eda"]["findings"]:
        st.markdown(f"- {f}")
    st.write("")
    table = pd.DataFrame(prof["columns"]).drop(columns=["top_values"], errors="ignore")
    st.dataframe(table, use_container_width=True, hide_index=True)

with tabs[3]:
    charts = S["charts"]
    for i in range(0, len(charts), 2):
        row = st.columns(2)
        for col, (title, fig) in zip(row, charts[i:i + 2]):
            col.plotly_chart(fig, use_container_width=True, key=f"auto_{i}_{title}")

    st.markdown("#### Custom chart")
    c1, c2, c3, c4 = st.columns(4)
    kind = c1.selectbox("Type", ["scatter", "histogram", "box", "bar", "line"])
    x = c2.selectbox("X", df.columns)
    y = c3.selectbox("Y", ["-"] + list(df.columns))
    color = c4.selectbox("Color", ["-"] + list(df.columns))
    kw = {"x": x}
    if y != "-":
        kw["y"] = y
    if color != "-":
        kw["color"] = color
    else:
        kw["color_discrete_sequence"] = [ACCENT]
    try:
        st.plotly_chart(style(getattr(px, kind)(df, **kw), ""), use_container_width=True)
    except Exception as e:
        st.warning(f"Can't plot that combination: {e}")

with tabs[4]:
    if S.get("ml_error"):
        st.error(S["ml_error"])
    if ml is None:
        st.info("No model trained. Pick a target in the sidebar or train one below.")
    else:
        d = ml.details
        st.caption(f"{ml.task} on `{ml.target}` · {d['n_train']} train / {d['n_test']} test rows · "
                   f"{d['cv_folds']}-fold CV")
        st.dataframe(ml.leaderboard, use_container_width=True, hide_index=True)
        left, right = st.columns(2)
        fi = ml.feature_importance.head(12)
        fig = px.bar(fi, x="importance", y="feature", orientation="h", color_discrete_sequence=[ACCENT])
        fig.update_yaxes(categoryorder="total ascending")
        left.plotly_chart(style(fig, f"Permutation importance · {ml.best_model_name}"),
                          use_container_width=True)
        if ml.task == "classification":
            labels = [str(v) for v in d["labels"]]
            fig = px.imshow(d["confusion_matrix"], x=labels, y=labels, text_auto=True,
                            labels=dict(x="Predicted", y="Actual"), color_continuous_scale="Teal")
            right.plotly_chart(style(fig, "Confusion matrix"), use_container_width=True)
        else:
            pv = pd.DataFrame({"actual": d["y_test"], "predicted": d["y_pred"]})
            fig = px.scatter(pv, x="actual", y="predicted", opacity=.6, color_discrete_sequence=[ACCENT])
            right.plotly_chart(style(fig, "Predicted vs actual"), use_container_width=True)

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
    q = st.text_input("Ask a question about the table", placeholder="average monthly charges by contract type")
    if q:
        with st.spinner("Writing query..."):
            res = pilot.sql_engine.ask(q)
        if res.sql:
            st.code(res.sql, language="sql")
        if res.error:
            st.error(res.error)
        else:
            st.dataframe(res.result, use_container_width=True, hide_index=True)
            fig = chart_for_result(res.result)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
    with st.expander("Run your own SQL"):
        manual = st.text_area("Query (table name is `data`)", "SELECT * FROM data LIMIT 10")
        if st.button("Run"):
            try:
                st.dataframe(pilot.sql_engine.run(manual), use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(str(e))

with tabs[6]:
    for turn in st.session_state.get("chat", []):
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("sql"):
                st.code(turn["sql"], language="sql")

    prompt = st.chat_input("e.g. what drives churn?  /  churn rate by region")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner(""):
                out = pilot.ask(prompt)
            st.markdown(f'<span class="pill">{out["route"]}</span>', unsafe_allow_html=True)
            st.markdown(out["answer"])
            sql_text = None
            if out.get("sql") is not None and out["sql"].result is not None:
                sql_text = out["sql"].sql
                st.code(sql_text, language="sql")
                st.dataframe(out["sql"].result.head(50), use_container_width=True, hide_index=True)
            if out.get("chart") is not None:
                st.plotly_chart(out["chart"], use_container_width=True)
            if out.get("sources"):
                with st.expander(f"{len(out['sources'])} sources"):
                    for s in out["sources"]:
                        st.markdown(f"**[{s['n']}]** `{s['source']}` - {s['text']}")
        st.session_state.chat += [{"role": "user", "content": prompt},
                                  {"role": "assistant", "content": out["answer"], "sql": sql_text}]
