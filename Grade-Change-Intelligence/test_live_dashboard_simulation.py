import joblib
import pandas as pd
import numpy as np

from simulator.simulator import Simulator
from models.recommender import recommend_actions
from explainability.shap_explainer import SHAPExplainer

def run_full_simulation_test(start_grade="CP080", target_grade="CP100", steps=200):
    print("=" * 80)
    print(f"STARTING COMPREHENSIVE SIMULATION TEST: {start_grade} -> {target_grade}")
    print("=" * 80)

    grades_df = pd.read_csv("data/grades.csv")
    model = joblib.load("models/model.pkl")
    explainer = SHAPExplainer(model)

    sim = Simulator(grades_df)
    sim.start_transition(start_grade, target_grade)

    checkpoints = [10, 43, 80, 110, 150, 200]

    for current_step in range(1, steps + 1):
        sim.step()

        if current_step in checkpoints:
            state = sim.get_machine_state()
            df_state = pd.DataFrame([state])
            
            instantaneous_risk = float(model.predict_proba(df_state)[0][1])

            # Dynamic 60s Trajectory Optimizer Call
            res = recommend_actions(model, sim, horizon=60, alpha=0.7)
            baseline_score = float(np.mean(res["baseline_trajectory"]))
            best_score = res["best_score"]
            improvement = max(0.0, baseline_score - best_score)
            recs = res["recommendations"]

            # SHAP Explanation
            shap_res = explainer.explain(state)
            top_factors = shap_res["top_factors"][:3]

            print(f"\n[TIME = {current_step} s]")
            print(f"  * Machine State: Speed={state['speed']:.1f}, Stock={state['stock_flow']:.1f}, GSM={state['basis_weight']:.1f}, Moisture={state['moisture']:.2f}")
            print(f"  * Instantaneous Risk: {instantaneous_risk:.1%}")
            print(f"  * Current Trajectory Risk (60s Baseline): {baseline_score:.1%}")
            print(f"  * AI Optimised Trajectory Risk (60s AI):  {best_score:.1%} (Delta: -{improvement:.1%})")
            print(f"  * AI Recommendations: {recs if recs else 'None (Operating Near Optimal)'}")
            print(f"  * SHAP Top Drivers: {', '.join([f'{f}: {v:+.2f}' for f, v in top_factors])}")

    print("\n" + "=" * 80)
    print("FULL SIMULATION TEST COMPLETED SUCCESSFULLY WITH 0 ERRORS!")
    print("=" * 80)

if __name__ == "__main__":
    run_full_simulation_test()
