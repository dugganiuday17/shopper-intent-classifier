"""
ShopperSense - Streamlit front-end for the five pre-trained purchase-intent
classifiers. Models are only loaded and scored here; nothing is retrained.

Run locally:  streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
SAMPLE_TEST_CSV = ROOT / "test_data.csv"
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

# Palette shared by the CSS and the matplotlib charts
INDIGO, CORAL, TEAL, AMBER, SKY, INK = "#5B4BDB", "#FF6B8B", "#13B5A6", "#F5A623", "#3A86FF", "#1F2340"
MODEL_COLORS = {
    "Logistic Regression": INDIGO,
    "Decision Tree": TEAL,
    "KNN": AMBER,
    "Naive Bayes": CORAL,
    "Support Vector Machine": SKY,
}
METRIC_ICONS = {"Accuracy": "🎯", "AUC": "📈", "Precision": "🔬", "Recall": "🧲", "F1": "⚖️", "MCC": "🧮"}
METRIC_HINTS = {
    "Accuracy": "Share of all sessions classified correctly",
    "AUC": "How well purchase probabilities rank buyers above non-buyers",
    "Precision": "Of sessions flagged as buyers, how many really bought",
    "Recall": "Of real buyers, how many were caught",
    "F1": "Harmonic mean of precision and recall",
    "MCC": "Balanced correlation of predictions vs. truth (-1 to 1)",
}

st.set_page_config(page_title="ShopperSense | Purchase-Intent Classifier", page_icon="🛒", layout="wide")

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#C9CCE0",
    "axes.labelcolor": INK,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "xtick.color": "#4A4F6A",
    "ytick.color": "#4A4F6A",
    "axes.grid": True,
    "grid.color": "#ECEEF7",
    "figure.facecolor": "white",
})

st.markdown(
    f"""
<style>
.block-container {{ padding-top: 1.6rem; padding-bottom: 2rem; }}

/* ---------- hero banner ---------- */
.hero {{
    background: linear-gradient(120deg, {INDIGO} 0%, #8E44AD 55%, {CORAL} 100%);
    border-radius: 18px; padding: 26px 32px; color: white; margin-bottom: 18px;
    box-shadow: 0 10px 30px rgba(91, 75, 219, 0.25);
}}
.hero h1 {{ color: white; font-size: 2.3rem; margin: 0; padding: 0; font-weight: 800; }}
.hero p {{ color: rgba(255,255,255,0.92); font-size: 1.02rem; margin: 6px 0 14px 0; }}
.chip {{
    display: inline-block; background: rgba(255,255,255,0.18); border: 1px solid rgba(255,255,255,0.35);
    border-radius: 999px; padding: 4px 12px; margin: 0 6px 6px 0; font-size: 0.82rem; color: white;
}}

/* ---------- stat + metric cards ---------- */
.card-row {{ display: grid; gap: 14px; margin: 6px 0 18px 0; }}
.stat-card, .metric-card {{
    background: white; border-radius: 14px; padding: 14px 18px;
    box-shadow: 0 4px 14px rgba(31, 35, 64, 0.07); border: 1px solid #ECEEF7;
}}
.stat-card {{ border-top: 4px solid var(--accent); }}
.stat-label, .metric-label {{ font-size: 0.78rem; text-transform: uppercase; letter-spacing: .06em; color: #6B7090; }}
.stat-value {{ font-size: 1.5rem; font-weight: 800; color: {INK}; margin-top: 2px;
               white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.metric-card {{ border-left: 5px solid var(--accent); }}
.metric-value {{ font-size: 1.75rem; font-weight: 800; color: var(--accent); line-height: 1.2; margin: 4px 0 8px 0; }}
.bar {{ height: 6px; background: #EEF0FA; border-radius: 99px; overflow: hidden; }}
.bar > div {{ height: 100%; background: var(--accent); border-radius: 99px; }}
.metric-hint {{ font-size: 0.72rem; color: #8A8FAA; margin-top: 8px; }}

/* ---------- section titles ---------- */
.section-title {{ font-size: 1.15rem; font-weight: 700; color: {INK}; margin: 4px 0 10px 0; }}
.section-title span {{ background: linear-gradient(90deg, {INDIGO}, {CORAL});
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}

/* ---------- bordered containers become white cards ----------
   (plain layout wrappers get the empty emotion class "st-emotion-cache-0"; only
   st.container(border=True) has real styles, so it is the one we restyle) */
[data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"]:not(.st-emotion-cache-0) {{
    background: white; border-radius: 16px !important; border: 1px solid #ECEEF7 !important;
    box-shadow: 0 4px 14px rgba(31, 35, 64, 0.06);
}}

/* ---------- tabs as pills ---------- */
.stTabs [data-baseweb="tab-list"] {{ gap: 8px; border-bottom: none; }}
.stTabs [data-baseweb="tab"] {{
    background: #EEF0FA; border-radius: 999px; padding: 8px 18px; height: auto; color: {INK}; font-weight: 600;
}}
.stTabs [aria-selected="true"] {{ background: linear-gradient(90deg, {INDIGO}, #8E44AD); color: white !important; }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none; }}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {{ background: linear-gradient(180deg, #EEF0FF 0%, #F9EEF6 100%); }}
.step {{ display: flex; align-items: center; gap: 10px; font-weight: 700; color: {INK}; margin: 10px 0 6px 0; }}
.step b {{
    background: linear-gradient(135deg, {INDIGO}, {CORAL}); color: white; border-radius: 50%;
    width: 26px; height: 26px; display: inline-flex; align-items: center; justify-content: center; font-size: .85rem;
}}

/* ---------- buttons ---------- */
.stDownloadButton button, .stButton button {{
    background: linear-gradient(90deg, {INDIGO}, #8E44AD); color: white; border: none; border-radius: 10px;
    font-weight: 600;
}}
.stDownloadButton button:hover, .stButton button:hover {{ filter: brightness(1.1); color: white; }}

/* ---------- winner banner ---------- */
.winner {{
    background: linear-gradient(120deg, {TEAL}, #2BC48A); color: white; border-radius: 14px;
    padding: 14px 20px; font-size: 1.02rem; margin: 6px 0 14px 0;
    box-shadow: 0 6px 18px rgba(19, 181, 166, 0.25);
}}
.footer {{ text-align: center; color: #8A8FAA; font-size: .8rem; margin-top: 28px; }}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- loading
@st.cache_data
def load_schema() -> dict:
    return json.loads((MODEL_DIR / "feature_schema.json").read_text())


@st.cache_data
def load_training_report() -> dict:
    return json.loads((MODEL_DIR / "metrics.json").read_text())


@st.cache_resource
def load_models(model_files: dict) -> dict:
    return {key: joblib.load(MODEL_DIR / fname) for key, fname in model_files.items()}


def to_bool_int(series: pd.Series) -> pd.Series:
    """Accept TRUE/FALSE, True/False, 1/0, yes/no and return 0/1 ints."""
    if series.dtype == bool:
        return series.astype(int)
    mapping = {"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0, "yes": 1, "no": 0}
    mapped = series.astype(str).str.strip().str.lower().map(mapping)
    if mapped.isna().any():
        bad = series[mapped.isna()].unique()[:5]
        raise ValueError(f"Column '{series.name}' has values that are not boolean-like: {list(bad)}")
    return mapped.astype(int)


def prepare_frame(raw: pd.DataFrame, schema: dict):
    """Validate the uploaded CSV and split it into X / y with training-time dtypes."""
    features = schema["numeric"] + schema["categorical"] + schema["binary"]
    target = schema["target"]
    missing = [c for c in features + [target] if c not in raw.columns]
    if missing:
        raise ValueError(f"Uploaded file is missing required column(s): {', '.join(missing)}")

    df = raw[features + [target]].copy()
    for col in schema["numeric"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[schema["numeric"]].isna().any().any():
        raise ValueError("Numeric columns contain empty or non-numeric values.")
    for col in schema["binary"] + [target]:
        df[col] = to_bool_int(df[col])
    # Integer-coded categoricals must match the int dtype the encoder saw in training
    for col in ["OperatingSystems", "Browser", "Region", "TrafficType"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64").astype(int)
    df["Month"] = df["Month"].astype(str).str.strip()
    df["VisitorType"] = df["VisitorType"].astype(str).str.strip()
    return df[features], df[target]


def score(model, X: pd.DataFrame, y: pd.Series, threshold: float):
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)
    auc = roc_auc_score(y, prob) if y.nunique() == 2 else float("nan")
    metrics = {
        "Accuracy": accuracy_score(y, pred),
        "AUC": auc,
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }
    return metrics, pred, prob


# ---------------------------------------------------------------- html helpers
def stat_cards(items):
    """items: list of (label, value, accent colour)."""
    cells = "".join(
        f'<div class="stat-card" style="--accent:{color}"><div class="stat-label">{label}</div>'
        f'<div class="stat-value" title="{value}">{value}</div></div>'
        for label, value, color in items
    )
    st.markdown(
        f'<div class="card-row" style="grid-template-columns:repeat({len(items)},1fr)">{cells}</div>',
        unsafe_allow_html=True,
    )


def metric_color(value: float) -> str:
    if np.isnan(value):
        return "#8A8FAA"
    if value >= 0.85:
        return TEAL
    if value >= 0.65:
        return INDIGO
    if value >= 0.5:
        return AMBER
    return CORAL


def metric_cards(metrics: dict):
    cells = []
    for name in METRIC_NAMES:
        value = metrics[name]
        color = metric_color(value)
        width = 0 if np.isnan(value) else max(0.0, min(value, 1.0)) * 100
        shown = "n/a" if np.isnan(value) else f"{value:.4f}"
        cells.append(
            f'<div class="metric-card" style="--accent:{color}">'
            f'<div class="metric-label">{METRIC_ICONS[name]} {name}</div>'
            f'<div class="metric-value">{shown}</div>'
            f'<div class="bar"><div style="width:{width:.1f}%"></div></div>'
            f'<div class="metric-hint">{METRIC_HINTS[name]}</div></div>'
        )
    st.markdown(
        f'<div class="card-row" style="grid-template-columns:repeat(6,1fr)">{"".join(cells)}</div>',
        unsafe_allow_html=True,
    )


def section(title: str):
    st.markdown(f'<div class="section-title"><span>{title}</span></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- plots
CM_CMAP = LinearSegmentedColormap.from_list("shopper", ["#F4F3FF", "#B3A9F5", INDIGO])
# Softer variant for tables so the dark text stays readable on every cell
TABLE_CMAP = LinearSegmentedColormap.from_list("shopper_table", ["#FFFFFF", "#E4E0FD", "#BFB6F6"])


def plot_confusion(y, pred, title: str):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    pct = cm / cm.sum() * 100
    labels = np.array([[f"{n:,}\n{p:.1f}%" for n, p in zip(row_n, row_p)] for row_n, row_p in zip(cm, pct)])
    fig, ax = plt.subplots(figsize=(4.8, 3.9))
    sns.heatmap(cm, annot=labels, fmt="", cmap=CM_CMAP, cbar=False, linewidths=3, linecolor="white", ax=ax,
                annot_kws={"fontsize": 12, "fontweight": "bold"},
                xticklabels=["No purchase", "Purchase"], yticklabels=["No purchase", "Purchase"])
    ax.grid(False)
    ax.set_xlabel("Predicted", fontweight="bold")
    ax.set_ylabel("Actual", fontweight="bold")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def plot_roc(curves: dict, highlight: str = None):
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for name, (y, prob) in curves.items():
        fpr, tpr, _ = roc_curve(y, prob)
        color = MODEL_COLORS.get(name, INDIGO)
        is_main = name == highlight or len(curves) == 1
        ax.plot(fpr, tpr, lw=2.8 if is_main else 1.5, color=color, alpha=1 if is_main else 0.8,
                label=f"{name}  (AUC {roc_auc_score(y, prob):.3f})")
        if len(curves) == 1:
            ax.fill_between(fpr, tpr, alpha=0.12, color=color)
    ax.plot([0, 1], [0, 1], ls="--", color="#A0A4BE", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right", frameon=False)
    fig.tight_layout()
    return fig


def plot_metric_bars(table: pd.DataFrame):
    long = table.reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=long, x="Metric", y="Score", hue="Model", palette=MODEL_COLORS, ax=ax,
                edgecolor="white", linewidth=1.2)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- layout
schema = load_schema()
report = load_training_report()
models = load_models(schema["model_files"])
name_to_key = {v: k for k, v in schema["display_names"].items()}

st.markdown(
    f"""
<div class="hero">
  <h1>🛒 ShopperSense</h1>
  <p>Will this browsing session end in a purchase? Five classifiers trained on the UCI
     <i>Online Shoppers Purchasing Intention</i> dataset, evaluated live on the test CSV you upload.</p>
  <span class="chip">🤖 5 pre-trained models</span>
  <span class="chip">🧾 {report['n_rows_after_dedup']:,} sessions · 17 features</span>
  <span class="chip">⚡ No retraining — load &amp; score only</span>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="step"><b>1</b> Test data</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload test CSV (features + `Revenue`)", type=["csv"])
    use_sample = st.checkbox("Use bundled test_data.csv instead", value=uploaded is None)
    st.download_button(
        "⬇️  Download sample test_data.csv",
        data=SAMPLE_TEST_CSV.read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown('<div class="step"><b>2</b> Model</div>', unsafe_allow_html=True)
    chosen_name = st.selectbox("Choose a classifier", list(name_to_key.keys()))

    st.markdown('<div class="step"><b>3</b> Decision threshold</div>', unsafe_allow_html=True)
    threshold = st.slider(
        "P(purchase) needed to predict a buyer", 0.05, 0.95, 0.50, 0.05,
        help="0.50 reproduces the README numbers. Lower it to catch more buyers at the cost of precision.",
    )
    params = report["models"][name_to_key[chosen_name]]["best_params"]
    with st.expander("🔧 Tuned hyper-parameters", expanded=False):
        st.json(params)

if uploaded is not None and not use_sample:
    source_label = uploaded.name
    raw_df = pd.read_csv(uploaded)
elif use_sample:
    source_label = "test_data.csv"
    raw_df = pd.read_csv(SAMPLE_TEST_CSV)
else:
    st.info("👈 Upload a test CSV from the sidebar, or tick **Use bundled test_data.csv**.")
    st.stop()

try:
    X, y = prepare_frame(raw_df, schema)
except ValueError as err:
    st.error(f"Could not use this file: {err}")
    st.stop()

stat_cards([
    ("📄 Data source", source_label, INDIGO),
    ("🧮 Rows scored", f"{len(X):,}", SKY),
    ("🛍️ Actual purchase rate", f"{y.mean() * 100:.1f}%", CORAL),
    ("🤖 Active model", chosen_name, MODEL_COLORS[chosen_name]),
])

tab_eval, tab_compare, tab_pred, tab_about = st.tabs(
    ["📊 Selected model", "🏁 Compare all models", "🔎 Predictions", "ℹ️ About"]
)

model = models[name_to_key[chosen_name]]
metrics, pred, prob = score(model, X, y, threshold)

with tab_eval:
    section(f"{chosen_name} · evaluation metrics @ threshold {threshold:.2f}")
    metric_cards(metrics)

    left, right = st.columns(2, gap="medium")
    with left:
        with st.container(border=True):
            section("Confusion matrix")
            st.pyplot(plot_confusion(y, pred, f"{chosen_name} @ threshold {threshold:.2f}"))
    with right:
        with st.container(border=True):
            section("Classification report")
            rep = classification_report(
                y, pred, labels=[0, 1], target_names=["No purchase (0)", "Purchase (1)"],
                output_dict=True, zero_division=0,
            )
            rep_df = pd.DataFrame(rep).T
            if "accuracy" in rep_df.index:  # sklearn repeats accuracy in every column; show it like the text report
                rep_df.loc["accuracy", ["precision", "recall"]] = np.nan
                rep_df.loc["accuracy", "support"] = len(y)
            st.dataframe(
                rep_df.style.format("{:.4f}", na_rep="").format("{:,.0f}", subset=["support"]).background_gradient(
                    cmap=TABLE_CMAP, subset=["precision", "recall", "f1-score"], vmin=0, vmax=1,
                    text_color_threshold=0,
                ),
                use_container_width=True,
            )
    if y.nunique() == 2:
        with st.container(border=True):
            section("ROC curve")
            st.pyplot(plot_roc({chosen_name: (y, prob)}, highlight=chosen_name))

with tab_compare:
    rows, curves = {}, {}
    for name, key in name_to_key.items():
        m, _, p = score(models[key], X, y, threshold)
        rows[name] = m
        curves[name] = (y, p)
    table = pd.DataFrame(rows).T[METRIC_NAMES]
    table.index.name = "Model"

    best_by_mcc = table["MCC"].idxmax()
    st.markdown(
        f'<div class="winner">🏆 Best MCC on this data: <b>{best_by_mcc}</b> '
        f'({table.loc[best_by_mcc, "MCC"]:.4f}) · best F1: <b>{table["F1"].idxmax()}</b> '
        f'({table["F1"].max():.4f})</div>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        section("All five models on the same uploaded data")
        st.dataframe(
            table.style.format("{:.4f}")
            .background_gradient(cmap=TABLE_CMAP, axis=0, text_color_threshold=0)
            .highlight_max(axis=0, props="background-color:#C8F2EA; color:#0B6B62; font-weight:800;"),
            use_container_width=True,
        )
    with st.container(border=True):
        section("Metric comparison")
        st.pyplot(plot_metric_bars(table))
    if y.nunique() == 2:
        with st.container(border=True):
            section("ROC curves (selected model highlighted)")
            st.pyplot(plot_roc(curves, highlight=chosen_name))

with tab_pred:
    out = raw_df.loc[X.index].copy()
    out["P(purchase)"] = prob.round(4)
    out["Predicted"] = pred
    out["Correct"] = np.where(pred == y.values, "✅", "❌")
    n_wrong = int((out["Correct"] == "❌").sum())
    stat_cards([
        ("✅ Correct", f"{len(out) - n_wrong:,}", TEAL),
        ("❌ Misclassified", f"{n_wrong:,}", CORAL),
        ("🛍️ Predicted buyers", f"{int(pred.sum()):,}", INDIGO),
    ])
    with st.container(border=True):
        section(f"Row-level predictions · {chosen_name}")
        only_errors = st.toggle("Show misclassified rows only")
        view = out[out["Correct"] == "❌"] if only_errors else out
        st.dataframe(
            view,
            use_container_width=True,
            height=420,
            column_config={
                "P(purchase)": st.column_config.ProgressColumn(
                    "P(purchase)", min_value=0.0, max_value=1.0, format="%.3f"
                ),
            },
        )
        st.download_button(
            "⬇️  Download predictions CSV",
            data=out.drop(columns="Correct").to_csv(index=False).encode(),
            file_name=f"predictions_{name_to_key[chosen_name]}.csv",
            mime="text/csv",
        )

with tab_about:
    with st.container(border=True):
        section("Dataset & training setup")
        st.markdown(
            f"""
- **Source:** {report['dataset']} — 12,330 sessions, 17 input features, binary target `Revenue`.
- **Cleaning:** duplicate rows dropped → {report['n_rows_after_dedup']:,} rows.
- **Split:** stratified 80/20 (train {report['n_train']:,} / test {report['n_test']:,}), `random_state={report['random_state']}`.
- **Preprocessing (inside each saved pipeline):** `log1p` + standard scaling for the 10 numeric
  features, one-hot encoding for Month / OS / Browser / Region / TrafficType / VisitorType,
  Weekend passed through as 0/1.
- **Tuning:** 5-fold stratified `GridSearchCV`, optimising F1 (≈15.6% positives).
"""
        )
    with st.container(border=True):
        section("Hold-out metrics recorded at training time (threshold 0.50)")
        ref = pd.DataFrame(
            {v["display_name"]: v["test_metrics"] for v in report["models"].values()}
        ).T[METRIC_NAMES]
        st.dataframe(ref.style.format("{:.4f}").background_gradient(cmap=TABLE_CMAP, axis=0, text_color_threshold=0),
                     use_container_width=True)

st.markdown('<div class="footer">ShopperSense · ML Assignment 2 · M.Tech AIML</div>', unsafe_allow_html=True)
