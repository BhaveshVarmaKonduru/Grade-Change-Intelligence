from copy import deepcopy
import pandas as pd
import numpy as np


def _simulate_states(simulator, settings_override, horizon=60):
    """
    Clones simulator, applies setpoints, steps simulation for `horizon` seconds,
    and returns a list of state dicts [S_1, ..., S_horizon].
    """
    trial_sim = deepcopy(simulator)
    if settings_override:
        trial_sim.apply_setpoints(settings_override)

    states = []
    for _ in range(horizon):
        if not trial_sim.is_running():
            break
        trial_sim.step(enable_noise=False)
        states.append(trial_sim.get_machine_state())

    return states


def compute_trajectory_score(probabilities, alpha=0.7):
    """
    Weighted combination of Mean Risk and Max Risk over trajectory:
    Score = alpha * mean(risks) + (1 - alpha) * max(risks)
    """
    if not probabilities:
        return 1.0
    mean_risk = float(np.mean(probabilities))
    max_risk = float(np.max(probabilities))
    return alpha * mean_risk + (1.0 - alpha) * max_risk


def recommend_actions(model, simulator, horizon=60, alpha=0.7):
    """
    Ultra-fast multi-horizon, joint multi-variable simulation-based optimizer.
    Uses batch model inference and targeted search for real-time Streamlit performance.
    """
    current = simulator.get_machine_state()

    # Load bounds from grades.csv
    try:
        grades_df = pd.read_csv("data/grades.csv")
        bounds = {
            "speed": (float(grades_df["machine_speed"].min()), float(grades_df["machine_speed"].max())),
            "stock_flow": (float(grades_df["stock_flow"].min()), float(grades_df["stock_flow"].max())),
            "steam": (float(grades_df["steam_pressure"].min()), float(grades_df["steam_pressure"].max())),
            "filler": (float(grades_df["filler_flow"].min()), float(grades_df["filler_flow"].max())),
        }
    except Exception:
        bounds = {
            "speed": (520.0, 1200.0),
            "stock_flow": (260.0, 1350.0),
            "steam": (3.5, 7.2),
            "filler": (22.0, 180.0),
        }

    def clamp_combo(combo):
        if combo is None:
            return None
        return {
            "speed": max(bounds["speed"][0], min(combo["speed"], bounds["speed"][1])),
            "stock_flow": max(bounds["stock_flow"][0], min(combo["stock_flow"], bounds["stock_flow"][1])),
            "steam": max(bounds["steam"][0], min(combo["steam"], bounds["steam"][1])),
            "filler": max(bounds["filler"][0], min(combo["filler"], bounds["filler"][1])),
        }

    # 1. Define high-impact targeted candidate setpoint vectors
    raw_combos = [
        None,  # Baseline (no AI change)
        # Single parameter adjustments
        {"speed": current["speed"] - 20, "stock_flow": current["stock_flow"], "steam": current["steam"], "filler": current["filler"]},
        {"speed": current["speed"] + 20, "stock_flow": current["stock_flow"], "steam": current["steam"], "filler": current["filler"]},
        {"speed": current["speed"], "stock_flow": current["stock_flow"] - 15, "steam": current["steam"], "filler": current["filler"]},
        {"speed": current["speed"], "stock_flow": current["stock_flow"] + 15, "steam": current["steam"], "filler": current["filler"]},
        {"speed": current["speed"], "stock_flow": current["stock_flow"], "steam": current["steam"] - 0.3, "filler": current["filler"]},
        {"speed": current["speed"], "stock_flow": current["stock_flow"], "steam": current["steam"] + 0.3, "filler": current["filler"]},
        {"speed": current["speed"], "stock_flow": current["stock_flow"], "steam": current["steam"], "filler": current["filler"] - 2},
        {"speed": current["speed"], "stock_flow": current["stock_flow"], "steam": current["steam"], "filler": current["filler"] + 2},
        # Dual parameter coupled adjustments
        {"speed": current["speed"] - 20, "stock_flow": current["stock_flow"] + 15, "steam": current["steam"], "filler": current["filler"]},
        {"speed": current["speed"] + 20, "stock_flow": current["stock_flow"] - 15, "steam": current["steam"], "filler": current["filler"]},
        {"speed": current["speed"], "stock_flow": current["stock_flow"] - 15, "steam": current["steam"] - 0.3, "filler": current["filler"]},
        {"speed": current["speed"], "stock_flow": current["stock_flow"] + 15, "steam": current["steam"] + 0.3, "filler": current["filler"]},
        {"speed": current["speed"], "stock_flow": current["stock_flow"], "steam": current["steam"] - 0.3, "filler": current["filler"] - 2},
        {"speed": current["speed"], "stock_flow": current["stock_flow"], "steam": current["steam"] + 0.3, "filler": current["filler"] + 2},
    ]

    candidate_combos = [clamp_combo(c) for c in raw_combos]

    # 2. Simulate trajectories for all candidates
    all_trajectories_states = []
    for candidate in candidate_combos:
        states = _simulate_states(simulator, candidate, horizon=horizon)
        all_trajectories_states.append(states)

    # 3. Perform BATCH model inference in a single call
    all_states_flat = []
    for states in all_trajectories_states:
        all_states_flat.extend(states)

    if not all_states_flat:
        return {
            "best_score": 1.0,
            "best_settings": current,
            "recommendations": [],
            "baseline_trajectory": [1.0] * horizon,
            "best_trajectory": [1.0] * horizon,
        }

    df_batch = pd.DataFrame(all_states_flat)
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
        if col not in df_batch.columns:
            df_batch[col] = 0.0
    df_batch = df_batch[feature_cols]
    all_probs_flat = model.predict_proba(df_batch)[:, 1]

    # 4. Unflatten predictions into per-candidate trajectory probability lists
    trajectories_probs = []
    idx = 0
    for states in all_trajectories_states:
        n_steps = len(states)
        probs = all_probs_flat[idx : idx + n_steps].tolist()
        trajectories_probs.append(probs)
        idx += n_steps

    # Baseline probability trajectory
    baseline_probs = trajectories_probs[0]
    best_score = compute_trajectory_score(baseline_probs, alpha=alpha)
    best_settings = current.copy()
    best_probs = baseline_probs

    # If baseline risk is already very low, don't recommend any changes.
    # This prevents spurious micro-optimizations on same-grade or converged transitions.
    LOW_RISK_THRESHOLD = 0.05
    if best_score < LOW_RISK_THRESHOLD:
        return {
            "best_score": best_score,
            "best_settings": best_settings,
            "recommendations": [],
            "baseline_trajectory": baseline_probs,
            "best_trajectory": best_probs,
        }

    # Evaluate candidate trajectory scores
    for i, candidate in enumerate(candidate_combos[1:], start=1):
        probs = trajectories_probs[i]
        score = compute_trajectory_score(probs, alpha=alpha)

        if score < best_score:
            best_score = score
            best_settings = candidate.copy()
            best_probs = probs

    # 5. Build recommendations list
    recommendations = []
    controllable_keys = ["speed", "stock_flow", "steam", "filler"]
    for key in controllable_keys:
        diff = round(best_settings[key] - current[key], 2)
        if abs(diff) < 1e-6:
            continue

        # Determine source tag
        min_b, max_b = bounds[key]
        if abs(best_settings[key] - min_b) < 1e-3 or abs(best_settings[key] - max_b) < 1e-3:
            source = "recipe limit"
        else:
            source = "digital-twin simulation"

        action_text = f"Increase {key.replace('_', ' ')} by {diff}" if diff > 0 else f"Decrease {key.replace('_', ' ')} by {abs(diff)}"
        recommendations.append({
            "parameter": key,
            "text": action_text,
            "diff": diff,
            "source": source,
            "curr_val": current[key],
            "prop_val": best_settings[key]
        })

    return {
        "best_score": best_score,
        "best_settings": best_settings,
        "recommendations": recommendations,
        "baseline_trajectory": baseline_probs,
        "best_trajectory": best_probs,
    }