import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

FEATURE_NAMES = [
    "speed",
    "stock_flow",
    "steam",
    "filler",
    "speed_rate",
    "stock_flow_rate",
    "steam_rate",
    "filler_rate",
]

DISPLAY_NAMES = {
    "speed": "Machine Speed",
    "stock_flow": "Stock Flow",
    "steam": "Steam Pressure",
    "filler": "Filler Flow",
    "speed_rate": "Speed Adj Rate",
    "stock_flow_rate": "Stock Flow Adj Rate",
    "steam_rate": "Steam Pressure Adj Rate",
    "filler_rate": "Filler Flow Adj Rate",
}


class SHAPExplainer:
    """
    Computes SHAP feature importance for off-spec risk predictions.
    """

    def __init__(self, model):
        self.model = model
        self.explainer = shap.TreeExplainer(model)

    def explain(self, state):
        """
        Calculates SHAP values for a single machine state dictionary or DataFrame row.
        """
        if isinstance(state, dict):
            state_copy = state.copy()
            for col in FEATURE_NAMES:
                if col not in state_copy:
                    state_copy[col] = 0.0
            df = pd.DataFrame([state_copy])[FEATURE_NAMES]
        else:
            df = pd.DataFrame(state)
            for col in FEATURE_NAMES:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[FEATURE_NAMES]

        shap_values = self.explainer.shap_values(df)

        # Handle output format differences across SHAP versions
        if isinstance(shap_values, list):
            sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        elif len(shap_values.shape) == 2:
            sv = shap_values[0]
        else:
            sv = shap_values[0]

        contributions = {}
        for feat, val in zip(FEATURE_NAMES, sv):
            contributions[feat] = float(val)

        base_value = float(
            self.explainer.expected_value[1]
            if isinstance(self.explainer.expected_value, (list, np.ndarray))
            else self.explainer.expected_value
        )

        # Sort features by highest risk-increasing contribution
        sorted_factors = sorted(
            contributions.items(), key=lambda x: abs(x[1]), reverse=True
        )

        return {
            "contributions": contributions,
            "base_value": base_value,
            "top_factors": sorted_factors,
        }

    def generate_chart(self, state):
        """
        Creates a clean horizontal bar chart of SHAP values for Streamlit display.
        """
        explanation = self.explain(state)
        contribs = explanation["contributions"]

        feats = [DISPLAY_NAMES[f] for f in FEATURE_NAMES]
        vals = [contribs[f] for f in FEATURE_NAMES]

        # Sort by value for plot
        sorted_pairs = sorted(zip(vals, feats), key=lambda x: x[0])
        sorted_vals, sorted_feats = zip(*sorted_pairs)

        fig, ax = plt.subplots(figsize=(8, 3.8))
        colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in sorted_vals]

        bars = ax.barh(sorted_feats, sorted_vals, color=colors, height=0.55)
        ax.axvline(0, color="#7f8c8d", linestyle="--", linewidth=1)

        ax.set_xlabel("SHAP Risk Contribution (Log-Odds Impact)", fontsize=10, fontweight="bold")
        ax.set_title("Top Off-Spec Risk Drivers (SHAP Feature Importance)", fontsize=11, fontweight="bold", pad=12)
        ax.grid(axis="x", linestyle=":", alpha=0.6)

        plt.tight_layout()
        return fig
