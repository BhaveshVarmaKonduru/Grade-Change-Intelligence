from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "model.pkl")


def predict_off_spec(data):
    """
    data: dictionary containing process variables
    """

    df = pd.DataFrame([data])

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

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    df = df[feature_cols]

    probability = model.predict_proba(df)[0][1]
    prediction = model.predict(df)[0]

    return prediction, probability


if __name__ == "__main__":

    sample = {
        "speed": 950,
        "stock_flow": 600,
        "steam": 4.6,
        "filler": 100,
        "basis_weight": 99.7,
        "moisture": 5.0,
        "ash": 20.1,
        "caliper": 96.0,
    }

    prediction, probability = predict_off_spec(sample)

    from recommender import recommend_actions

    actions = recommend_actions(sample, probability)

    print(f"\nPrediction : {'OFF-SPEC' if prediction else 'IN-SPEC'}")
    print(f"Probability: {probability:.2%}")

    print("\nRecommendations:")

    for action in actions:
        print("-", action)
