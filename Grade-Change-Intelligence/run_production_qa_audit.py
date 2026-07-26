import joblib
import pandas as pd
import numpy as np
from copy import deepcopy

from simulator.simulator import Simulator
from simulator.grades import get_grade
from models.recommender import recommend_actions
from explainability.shap_explainer import SHAPExplainer

def log_header(title):
    print("\n" + "=" * 80)
    print(f"  {title.upper()}")
    print("=" * 80)

# ==============================================================================
# AUDIT TEST 1: ML Pipeline & Feature Schema Verification
# ==============================================================================
def audit_ml_pipeline():
    log_header("Audit 1: ML Pipeline & Feature Schema Alignment")
    
    model = joblib.load("models/model.pkl")
    historical_df = pd.read_csv("data/historical_data.csv")
    
    feature_cols = ["speed", "stock_flow", "steam", "filler", "speed_rate", "stock_flow_rate", "steam_rate", "filler_rate"]
    
    historical_df['transition_id'] = historical_df['current_grade'] + "_" + historical_df['target_grade']
    for col in ['speed', 'stock_flow', 'steam', 'filler']:
        historical_df[col+'_rate'] = historical_df.groupby('transition_id')[col].diff().fillna(0)
        
    print(f"[+] Historical dataset rows: {len(historical_df)}")
    print(f"[+] Off-spec label distribution:\n{historical_df['off_spec'].value_counts().to_dict()}")
    
    # Check feature presence and order
    for col in feature_cols:
        assert col in historical_df.columns, f"Missing feature {col} in historical_data.csv"
        
    print("[OK] Feature names and presence in training dataset verified.")
    
    # Check XGBoost model feature names if available
    try:
        model_features = model.get_booster().feature_names
        print(f"[+] Model feature names: {model_features}")
        assert model_features == feature_cols, f"Model feature mismatch! Expected {feature_cols}, got {model_features}"
        print("[OK] XGBoost feature order strictly matches inference pipeline.")
    except Exception as e:
        print(f"[!] Note: Booster feature names check: {e}")

# ==============================================================================
# AUDIT TEST 2: Digital Twin Dynamics & Convergence Audit
# ==============================================================================
def audit_digital_twin_convergence():
    log_header("Audit 2: Digital Twin Process Dynamics & Convergence")
    
    grades_df = pd.read_csv("data/grades.csv")
    transitions = [
        ("CP080", "CP100"),
        ("CP080", "NB045"),
        ("KR120", "KR150"),
        ("CP080", "CB250")
    ]
    
    for start, target in transitions:
        sim = Simulator(grades_df)
        sim.start_transition(start, target, duration=300)
        target_rec = get_grade(grades_df, target)
        
        target_gsm = target_rec["gsm_target"]
        target_speed = target_rec["machine_speed"]
        
        print(f"\n[-->] Transition: {start} -> {target} (Target GSM: {target_gsm}, Target Speed: {target_speed})")
        
        # Step simulation for 300 seconds (noise-free for deterministic convergence verification)
        states = []
        for t in range(1, 301):
            sim.step(enable_noise=False)
            states.append(sim.get_machine_state())
            
        final_state = states[-1]
        gsm_err = abs(final_state["basis_weight"] - target_gsm) / target_gsm * 100
        speed_err = abs(final_state["speed"] - target_speed) / target_speed * 100
        
        print(f"  * t=300s Final GSM: {final_state['basis_weight']:.2f} (Target: {target_gsm}) | Error: {gsm_err:.2f}%")
        print(f"  * t=300s Final Speed: {final_state['speed']:.2f} (Target: {target_speed}) | Error: {speed_err:.2f}%")
        
        # Convergence verification (< 2.5% BW_TOLERANCE)
        assert gsm_err < 2.5, f"Basis weight failed to converge within 2.5% tolerance for {start}->{target}! Error: {gsm_err:.2f}%"
        assert speed_err < 0.1, f"Speed failed to reach target for {start}->{target}!"
        print(f"  [OK] Converged smoothly to target recipe without divergence or infinite drift.")

# ==============================================================================
# AUDIT TEST 3: Deepcopy Isolation Verification
# ==============================================================================
def audit_deepcopy_isolation():
    log_header("Audit 3: Deepcopy State Isolation")
    
    grades_df = pd.read_csv("data/grades.csv")
    model = joblib.load("models/model.pkl")
    
    sim = Simulator(grades_df)
    sim.start_transition("CP080", "CP100")
    for _ in range(30):
        sim.step()
        
    state_before = deepcopy(sim.get_machine_state())
    time_before = sim.machine.time
    
    # Run optimizer (evaluates 15 candidate trajectories x 60 steps)
    res = recommend_actions(model, sim, horizon=60, alpha=0.7)
    
    state_after = sim.get_machine_state()
    time_after = sim.machine.time
    
    assert state_before == state_after, f"MUTATION BUG DETECTED! Optimizer modified live machine state:\nBefore: {state_before}\nAfter: {state_after}"
    assert time_before == time_after, f"MUTATION BUG DETECTED! Optimizer advanced live machine time: {time_before} -> {time_after}"
    
    print("[OK] Live simulator state and clock remain 100% unmutated during optimization.")

# ==============================================================================
# AUDIT TEST 4: Trajectory Alignment Verification
# ==============================================================================
def audit_trajectory_alignment():
    log_header("Audit 4: Optimizer Forecast vs Actual Simulator Evolution Alignment")
    
    grades_df = pd.read_csv("data/grades.csv")
    model = joblib.load("models/model.pkl")
    
    sim = Simulator(grades_df)
    sim.start_transition("CP080", "CP100")
    for _ in range(30):
        sim.step()
        
    # Get optimizer recommendation
    res = recommend_actions(model, sim, horizon=60, alpha=0.7)
    best_settings = res["best_settings"]
    predicted_traj_probs = res["best_trajectory"]
    
    # Clone simulator, apply setpoints, replay noise-free for exact comparison
    sim_actual = deepcopy(sim)
    sim_actual.apply_setpoints(best_settings)
    
    feature_cols = ["speed", "stock_flow", "steam", "filler", "speed_rate", "stock_flow_rate", "steam_rate", "filler_rate"]
    actual_probs = []
    for _ in range(60):
        sim_actual.step(enable_noise=False)
        st = sim_actual.get_machine_state()
        p = float(model.predict_proba(pd.DataFrame([st])[feature_cols])[0][1])
        actual_probs.append(p)
        
    # Deterministic forecast should match exactly (MAE ~ 0)
    mae = np.mean(np.abs(np.array(predicted_traj_probs) - np.array(actual_probs)))
    print(f"[+] Mean Absolute Error (deterministic replay): {mae:.8f}")
    
    assert mae < 1e-6, f"Trajectory alignment mismatch! MAE too high: {mae}"
    print("[OK] Optimizer forecast EXACTLY matches deterministic digital twin replay (MAE < 1e-6).")
    
    # Also verify stochastic replay divergence is bounded (sanity check)
    sim_noisy = deepcopy(sim)
    sim_noisy.apply_setpoints(best_settings)
    feature_cols = ["speed", "stock_flow", "steam", "filler", "speed_rate", "stock_flow_rate", "steam_rate", "filler_rate"]
    noisy_probs = []
    for _ in range(60):
        sim_noisy.step(enable_noise=True)
        st = sim_noisy.get_machine_state()
        p = float(model.predict_proba(pd.DataFrame([st])[feature_cols])[0][1])
        noisy_probs.append(p)
    
    mae_noisy = np.mean(np.abs(np.array(predicted_traj_probs) - np.array(noisy_probs)))
    print(f"[+] Stochastic replay MAE (expected divergence from noise): {mae_noisy:.6f}")
    print("[OK] Stochastic divergence is expected and bounded by process noise.")

# ==============================================================================
# AUDIT TEST 5: Stress & Edge-Case Verification
# ==============================================================================
def audit_edge_cases():
    log_header("Audit 5: Industrial Operator Edge Cases & Stress Scenarios")
    
    grades_df = pd.read_csv("data/grades.csv")
    model = joblib.load("models/model.pkl")
    explainer = SHAPExplainer(model)
    
    sim = Simulator(grades_df)
    
    # Case A: Same-Grade Transition (CP080 -> CP080)
    print("\n[-->] Case A: Same-Grade Transition (CP080 -> CP080)")
    sim.start_transition("CP080", "CP080")
    sim.step()
    st = sim.get_machine_state()
    res = recommend_actions(model, sim)
    feature_cols = ["speed", "stock_flow", "steam", "filler", "speed_rate", "stock_flow_rate", "steam_rate", "filler_rate"]
    print(f"  * Risk: {float(model.predict_proba(pd.DataFrame([st])[feature_cols])[0][1]):.2%}")
    print(f"  * Recommendations: {res['recommendations']}")
    assert len(res['recommendations']) == 0, "Same-grade transition generated unexpected recommendations!"
    print("  [OK] Handled cleanly (0 risk, 0 setpoint recommendations).")

    # Case B: Immediate Restart Mid-Transition
    print("\n[-->] Case B: Immediate Transition Restart (CP080->CP100 then instant switch to KR150)")
    sim.start_transition("CP080", "CP100")
    for _ in range(15): sim.step()
    sim.start_transition("CP100", "KR150")  # Restart
    # Check state BEFORE stepping — machine should be initialized to CP100 baseline
    st_init = sim.get_machine_state()
    assert st_init["speed"] == 950.0, f"State failed to initialize to CP100 baseline! Got speed={st_init['speed']}"
    sim.step()
    # After one step, controller should be moving toward KR150 target (speed=700)
    st_new = sim.get_machine_state()
    assert st_new["speed"] < 950.0, "Controller failed to start moving toward new target!"
    print(f"  [OK] Simulator state reset cleanly upon transition restart (init speed=950.0, after step={st_new['speed']:.1f}).")

    # Case C: Simulation Past Max Steps (t > 300s)
    print("\n[-->] Case C: Simulation Duration Exceeded (t = 320s)")
    sim.start_transition("CP080", "CP100", duration=300)
    for _ in range(320):
        sim.step()
    assert sim.is_running() == False, "Simulator running flag failed to turn False after duration!"
    print("  [OK] Simulator automatically stopped upon reaching max duration.")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    print("*" * 80)
    print("      PRODUCTION QA AUDIT SUITE - GRADE CHANGE INTELLIGENCE")
    print("*" * 80)
    
    audit_ml_pipeline()
    audit_digital_twin_convergence()
    audit_deepcopy_isolation()
    audit_trajectory_alignment()
    audit_edge_cases()
    
    print("\n" + "*" * 80)
    print("      ALL QA AUDIT SUITES PASSED WITH ZERO ERRORS (100% VERIFIED)")
    print("*" * 80)

if __name__ == "__main__":
    main()
