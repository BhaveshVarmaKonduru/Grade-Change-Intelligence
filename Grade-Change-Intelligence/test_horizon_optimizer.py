import joblib
import pandas as pd
from simulator.simulator import Simulator
from models.recommender import recommend_actions

def test_optimizer():
    grades = pd.read_csv("data/grades.csv")
    model = joblib.load("models/model.pkl")
    
    sim = Simulator(grades)
    sim.start_transition("CP080", "CP100")
    
    for _ in range(50):
        sim.step()
        
    res = recommend_actions(model, sim, horizon=60, alpha=0.7)
    
    print("Optimization test complete!")
    print(f"Best score: {res['best_score']:.4f}")
    print(f"Recommendations: {res['recommendations']}")
    print(f"Baseline trajectory len: {len(res['baseline_trajectory'])}, start: {res['baseline_trajectory'][0]:.2%}, end: {res['baseline_trajectory'][-1]:.2%}")
    print(f"Best trajectory len: {len(res['best_trajectory'])}, start: {res['best_trajectory'][0]:.2%}, end: {res['best_trajectory'][-1]:.2%}")

if __name__ == "__main__":
    test_optimizer()
