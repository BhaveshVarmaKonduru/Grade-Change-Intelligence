import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from models.recommender import recommend_actions
from explainability.shap_explainer import SHAPExplainer
from simulator.simulator import Simulator
from streamlit_autorefresh import st_autorefresh

from utils.logging_helper import log_recommendations, update_recommendations_status, get_acceptance_metrics
from utils.correlation import discover_correlations

# -----------------------------
# Cached Model & Explainers
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

@st.cache_data
def get_cached_correlations():
    return discover_correlations()

@st.cache_resource
def load_ml_model():
    try:
        return joblib.load(BASE_DIR / "models" / "model.pkl")
    except Exception as e:
        st.error(f"Failed to load machine learning model: {e}")
        return None

@st.cache_resource
def load_shap_explainer(_model):
    if _model is None:
        return None
    try:
        return SHAPExplainer(_model)
    except Exception as e:
        st.error(f"Failed to initialize SHAP explainer: {e}")
        return None

@st.cache_data
def load_grades():
    try:
        return pd.read_csv(BASE_DIR / "data" / "grades.csv")
    except Exception as e:
        st.error(f"Failed to load grades configuration: {e}")
        return None

model = load_ml_model()
explainer = load_shap_explainer(model)
grades_df = load_grades()

if model is None or grades_df is None:
    st.error("Critical components failed to load. Please verify your models and data files.")
    st.stop()

HORIZON = 60

# -----------------------------
# Session State
# -----------------------------
if "simulator" not in st.session_state:
    st.session_state.simulator = Simulator(grades_df)

if "running" not in st.session_state:
    st.session_state.running = False
if "recommendation_applied" not in st.session_state:
    st.session_state.recommendation_applied = False
if "last_logged_time" not in st.session_state:
    st.session_state.last_logged_time = -1
if "latest_recommendation_ids" not in st.session_state:
    st.session_state.latest_recommendation_ids = []

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Grade Change Intelligence",
    page_icon="📄",
    layout="wide"
)

# Refresh every second while simulation is running
if st.session_state.running:
    st_autorefresh(interval=1000, key="simulation_refresh")

st.title("📄 Grade Change Intelligence")
st.subheader("AI-Powered Paper Machine Grade Transition Advisor")

# -----------------------------
# Grade Selection & Metrics Sidebar
# -----------------------------
st.sidebar.header("Grade Transition")

grades = grades_df["grade_code"].tolist()

current_grade = st.sidebar.selectbox(
    "Current Grade",
    grades,
    index=0
)

target_grade = st.sidebar.selectbox(
    "Target Grade",
    grades,
    index=1
)

if st.sidebar.button("▶ Start Grade Change"):
    try:
        st.session_state.simulator.start_transition(
            current_grade,
            target_grade
        )
        st.session_state.running = True
        st.session_state.recommendation_applied = False
        st.session_state.last_logged_time = -1
        st.session_state.latest_recommendation_ids = []
    except Exception as e:
        st.error(f"Failed to start grade change transition: {e}")

st.sidebar.markdown("---")
st.sidebar.subheader("📈 AI Performance Metrics")
try:
    metrics = get_acceptance_metrics()
    st.sidebar.write(f"**Acceptance Rate**: {metrics['acceptance_rate']:.1%}")
    st.sidebar.write(f"Total Shown: {metrics['total_shown']}")
    st.sidebar.write(f"Accepted: {metrics['accepted']} | Rejected: {metrics['rejected']}")
except Exception as e:
    st.sidebar.error(f"Metrics error: {e}")

# -----------------------------
# Run One Simulation Step
# -----------------------------
if st.session_state.running:
    try:
        st.session_state.simulator.step()
        if not st.session_state.simulator.is_running():
            st.session_state.running = False
    except Exception as e:
        st.error(f"Error during simulation step: {e}")
        st.session_state.running = False

# -----------------------------
# Current Machine State
# -----------------------------
try:
    state = st.session_state.simulator.get_machine_state()
except Exception as e:
    st.error(f"Failed to read machine state: {e}")
    state = None

if state is None:
    st.info("Select two grades and click 'Start Grade Change'.")
    st.stop()

# Filter/prepare sample matching model training features
feature_cols = [
    "speed",
    "stock_flow",
    "steam",
    "filler",
    "speed_rate",
    "stock_flow_rate",
    "steam_rate",
    "filler_rate",
]
sample_df = pd.DataFrame([state])
for col in feature_cols:
    if col not in sample_df.columns:
        sample_df[col] = 0.0
sample = sample_df[feature_cols]

# -----------------------------
# Simulation Status Header
# -----------------------------
status = st.session_state.simulator.get_status()

st.subheader("⚙ Simulation Status")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Status",
        "🟢 Running" if status["running"] else "🔴 Stopped"
    )

with col2:
    st.metric(
        "Time",
        f'{status["time"]} s'
    )

with col3:
    st.metric(
        "Transition",
        f'{current_grade} → {target_grade}'
    )

# -----------------------------
# Current Process Values (Telemetry)
# -----------------------------
st.subheader("📊 Live Machine Telemetry")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Speed (m/min)", f'{state["speed"]:.1f}')
c2.metric("Stock Flow (L/min)", f'{state["stock_flow"]:.1f}')
c3.metric("Steam (bar)", f'{state["steam"]:.2f}')
c4.metric("Filler Flow (kg/min)", f'{state["filler"]:.1f}')

c1, c2, c3, c4 = st.columns(4)
c1.metric("Basis Weight (GSM)", f'{state["basis_weight"]:.1f}')
c2.metric("Moisture (%)", f'{state["moisture"]:.2f}')
c3.metric("Ash (%)", f'{state["ash"]:.1f}')
c4.metric("Caliper (µm)", f'{state["caliper"]:.1f}')

# -----------------------------
# Live Model Prediction
# -----------------------------
current_risk = 0.0
prediction = 0
try:
    prediction = model.predict(sample)[0]
    current_risk = float(model.predict_proba(sample)[0][1])
except Exception as e:
    st.error(f"Prediction error: {e}")

# -----------------------------
# Fast Live Dynamic Trajectory Optimizer
# -----------------------------
res = None
try:
    res = recommend_actions(
        model,
        st.session_state.simulator,
        horizon=HORIZON,
        alpha=0.7
    )
except Exception as e:
    st.error(f"Optimizer failed: {e}")

if res is not None:
    best_score = res["best_score"]
    best_settings = res["best_settings"]
    actions = res["recommendations"]
    baseline_traj = res["baseline_trajectory"]
    best_traj = res["best_trajectory"]

    baseline_score = float(np.mean(baseline_traj))
    improvement = max(0.0, baseline_score - best_score)
else:
    best_score = 1.0
    best_settings = state
    actions = []
    baseline_traj = [1.0] * HORIZON
    best_traj = [1.0] * HORIZON
    baseline_score = 1.0
    improvement = 0.0

# Log shown recommendations once per simulation step
if actions and improvement > 0.01:
    sim_time = status["time"]
    transition_str = f"{current_grade} -> {target_grade}"
    if st.session_state.last_logged_time != sim_time:
        try:
            st.session_state.latest_recommendation_ids = log_recommendations(
                transition=transition_str,
                sim_time=sim_time,
                actions_list=actions
            )
            st.session_state.last_logged_time = sim_time
        except Exception as e:
            st.error(f"Failed to log recommendations to database: {e}")

# -----------------------------
# Risk Metrics Overview
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Current Trajectory Risk",
        f"{baseline_score:.1%}",
        help="Average forecasted risk over the next 60s baseline trajectory without AI interventions."
    )

with col2:
    st.metric(
        "AI Optimised Trajectory Risk",
        f"{best_score:.1%}",
        delta=f"-{improvement:.1%}" if improvement > 0.001 else "Optimal",
        help="Average forecasted risk over 60s horizon if AI recommended setpoints are applied."
    )

st.subheader("⚠ Instantaneous Off-Spec Risk")
st.progress(float(current_risk))

if prediction:
    st.error(f"⚠ OFF-SPEC PREDICTED (Instantaneous Risk: {current_risk:.1%})")
else:
    st.success(f"✅ IN-SPEC PREDICTED (Instantaneous Risk: {current_risk:.1%})")

# -----------------------------
# Phase 3: Baseline vs AI Trajectory Comparison
# -----------------------------
st.subheader("📈 60-Second Forecasted Risk Trajectory Comparison")

chart_df = pd.DataFrame({
    "Time Step (s)": list(range(1, len(baseline_traj) + 1)),
    "Baseline Trajectory (Without AI)": [r * 100 for r in baseline_traj],
    "AI Optimised Trajectory (With AI)": [r * 100 for r in best_traj],
}).set_index("Time Step (s)")

st.line_chart(chart_df, color=["#e74c3c", "#2ecc71"])

# -----------------------------
# Phase 2 & 4: AI Recommendations & SHAP Explainability
# -----------------------------
st.subheader("🤖 AI Recommended Setpoint Adjustments")

if actions and improvement > 0.01:
    st.warning(
        f"""
        **Sustained Risk Reduction Available**: The AI optimizer projects a **{improvement:.1%}** reduction in average off-spec risk over the next 60 seconds.
        """
    )

    st.markdown("### Proposed Setpoint Changes")

    col_set1, col_set2 = st.columns(2)

    for i, act in enumerate(actions):
        parameter = act["parameter"]
        curr_val = act["curr_val"]
        prop_val = act["prop_val"]
        diff = act["diff"]
        source = act["source"]

        target_col = col_set1 if i % 2 == 0 else col_set2
        with target_col:
            st.info(
                f"**{parameter.replace('_', ' ').title()}**: {curr_val:.2f} → **{prop_val:.2f}** ({'+' if diff > 0 else ''}{diff:.2f})  \n*Source: {source}*"
            )

    st.divider()

    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        if st.button("✅ Accept Recommendation", disabled=st.session_state.recommendation_applied):
            try:
                st.session_state.simulator.apply_setpoints(best_settings)
                st.session_state.recommendation_applied = True
                update_recommendations_status(st.session_state.latest_recommendation_ids, "Accepted")
                st.success("AI recommendations applied successfully. Machine setpoints updated.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to apply setpoint recommendations: {e}")

    with col_btn2:
        if st.button("❌ Reject Recommendation"):
            try:
                update_recommendations_status(st.session_state.latest_recommendation_ids, "Rejected")
                st.info("Recommendation rejected. Continuing with baseline trajectory.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to record recommendation rejection: {e}")

else:
    st.success("Current machine trajectory is operating at optimal risk bounds for target recipe.")

# -----------------------------
# SHAP Explainability Accordion
# -----------------------------
if "last_shap_time" not in st.session_state:
    st.session_state.last_shap_time = -5
if "shap_fig" not in st.session_state:
    st.session_state.shap_fig = None
if "shap_top_factors" not in st.session_state:
    st.session_state.shap_top_factors = None

with st.expander("🔍 AI Decision Reasoning & SHAP Explainability", expanded=True):
    st.markdown("#### Feature Risk Contributions (SHAP Analysis)")
    st.caption("SHAP (SHapley Additive exPlanations) identifies the physical drivers contributing most to off-spec risk.")

    current_time = status["time"] if status["running"] else 0
    # Update every 3 seconds or if not running
    if current_time - st.session_state.last_shap_time >= 3 or not st.session_state.running:
        try:
            shap_res = explainer.explain(state)
            st.session_state.shap_top_factors = shap_res["top_factors"]
            st.session_state.shap_fig = explainer.generate_chart(state)
            st.session_state.last_shap_time = current_time
        except Exception as e:
            st.error(f"SHAP explainer failed to generate explanations: {e}")

    if st.session_state.shap_fig:
        col_shap1, col_shap2 = st.columns([1.2, 1.0])
        with col_shap1:
            st.pyplot(st.session_state.shap_fig)
        with col_shap2:
            st.markdown("**Top Off-Spec Risk Drivers:**")
            for feat, val in st.session_state.shap_top_factors[:4]:
                feat_name = feat.replace("_", " ").title()
                direction = "🔴 Increases Risk" if val > 0 else "🟢 Reduces Risk"
                st.write(f"- **{feat_name}**: `{val:+.3f}` ({direction})")

# -----------------------------
# Correlation Discovery Section
# -----------------------------
with st.expander("📊 Process Variable Correlations with Off-Spec Outcomes", expanded=False):
    st.markdown("#### Correlation & Mutual Information Discovery")
    st.caption("Derived from historical process logs to identify key factors leading to off-spec paper.")
    
    try:
        corr_findings = get_cached_correlations()
        if corr_findings:
            corr_df = pd.DataFrame(corr_findings)
            corr_df.columns = ["Process Variable", "Pearson Correlation", "Mutual Information (MI)"]
            st.dataframe(corr_df.style.format({
                "Pearson Correlation": "{:+.3f}",
                "Mutual Information (MI)": "{:.3f}"
            }))
            
            top_finding = corr_df.iloc[0]
            st.info(f"💡 **Top Finding**: `{top_finding['Process Variable']}` has the strongest correlation/influence ({top_finding['Pearson Correlation']:+.3f} Pearson, {top_finding['Mutual Information (MI)']:.3f} MI) with off-spec outcomes.")
        else:
            st.info("No historical data found to run correlation analysis.")
    except Exception as e:
        st.error(f"Error performing correlation analysis: {e}")
