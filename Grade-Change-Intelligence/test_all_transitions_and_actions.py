import joblib
import pandas as pd
import numpy as np

from simulator.simulator import Simulator
from models.recommender import recommend_actions
from explainability.shap_explainer import SHAPExplainer

def test_transition_with_and_without_ai(start_grade, target_grade):
    print("\n" + "=" * 70)
    print(f"TESTING TRANSITION: {start_grade} -> {target_grade}")
    print("=" * 70)

    grades_df = pd.read_csv("data/grades.csv")
    model = joblib.load("models/model.pkl")
    explainer = SHAPExplainer(model)

    # 1. Baseline Simulation (No AI Intervention)
    sim_base = Simulator(grades_df)
    sim_base.start_transition(start_grade, target_grade)
    
    # 2. AI Simulation (Accept AI Recommendation at t=30s)
    sim_ai = Simulator(grades_df)
    sim_ai.start_transition(start_grade, target_grade)

    for t in range(1, 151):
        sim_base.step()
        sim_ai.step()

        if t == 30:
            res_ai = recommend_actions(model, sim_ai, horizon=60, alpha=0.7)
            if res_ai["recommendations"]:
                print(f"[t={t}s] AI Recommendation Generated: {res_ai['recommendations']}")
                print(f"[t={t}s] Applying AI Setpoints: {res_ai['best_settings']}")
                sim_ai.apply_setpoints(res_ai["best_settings"])
            else:
                print(f"[t={t}s] Baseline already near optimal.")

        if t in [30, 60, 100, 150]:
            st_base = sim_base.get_machine_state()
            st_ai = sim_ai.get_machine_state()

            risk_base = float(model.predict_proba(pd.DataFrame([st_base]))[0][1])
            risk_ai = float(model.predict_proba(pd.DataFrame([st_ai]))[0][1])

            res_base_traj = recommend_actions(model, sim_base, horizon=60, alpha=0.7)
            res_ai_traj = recommend_actions(model, sim_ai, horizon=60, alpha=0.7)

            base_traj_risk = float(np.mean(res_base_traj["baseline_trajectory"]))
            ai_traj_risk = float(np.mean(res_ai_traj["baseline_trajectory"]))

            print(f"  [t={t:3d}s] Baseline GSM: {st_base['basis_weight']:.1f} | Instant Risk: {risk_base:6.1%} | 60s Traj Risk: {base_traj_risk:6.1%}")
            print(f"  [t={t:3d}s] AI-Path  GSM: {st_ai['basis_weight']:.1f} | Instant Risk: {risk_ai:6.1%} | 60s Traj Risk: {ai_traj_risk:6.1%}")

            # Verify SHAP generation works without exceptions
            shap_base = explainer.explain(st_base)
            top_base = shap_base["top_factors"][0]
            print(f"         SHAP Top Driver (Baseline): {top_base[0]} ({top_base[1]:+.2f})")

def main():
    transitions_to_test = [
        ("CP080", "CP100"),
        ("CP080", "CB250"),
        ("KR120", "KR150"),
        ("NB045", "CP120"),
    ]

    for start, target in transitions_to_test:
        test_transition_with_and_without_ai(start, target)

    print("\n" + "=" * 70)
    print("ALL TRANSITIONS TESTED SUCCESSFULLY! 0 ERRORS FOUND.")
    print("=" * 70)

if __name__ == "__main__":
    main()
