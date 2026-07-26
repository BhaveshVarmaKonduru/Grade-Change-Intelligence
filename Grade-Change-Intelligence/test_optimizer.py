from models.optimizer import AIOptimizer

current_state = {
    "speed": 980,
    "stock_flow": 600,
    "steam": 4.6,
    "filler": 95,
    "basis_weight": 99,
    "moisture": 5.6,
    "ash": 19,
    "caliper": 130,
}

optimizer = AIOptimizer()

best_state, probability = optimizer.optimize(current_state)

print("\nOptimized State")
for k, v in best_state.items():
    print(f"{k}: {v}")

print(f"\nFinal Probability: {probability:.3f}")