import joblib
from explainability.shap_explainer import SHAPExplainer

def test_shap():
    model = joblib.load("models/model.pkl")
    explainer = SHAPExplainer(model)

    sample = {
        "speed": 950,
        "stock_flow": 555,
        "steam": 4.6,
        "filler": 100,
        "basis_weight": 89.8,
        "moisture": 5.0,
        "ash": 19.2,
        "caliper": 94.4,
    }

    res = explainer.explain(sample)
    print("SHAP explanation test successful!")
    print("Top factors (feature, SHAP value):")
    for feat, val in res["top_factors"][:4]:
        direction = "INCREASES risk" if val > 0 else "DECREASES risk"
        print(f" - {feat}: {val:+.4f} ({direction})")

if __name__ == "__main__":
    test_shap()
