import time
import joblib
import pandas as pd
from simulator.simulator import Simulator
from models.recommender import recommend_actions

def benchmark():
    grades = pd.read_csv("data/grades.csv")
    model = joblib.load("models/model.pkl")

    sim = Simulator(grades)
    sim.start_transition("CP080", "CP100")
    for _ in range(43):
        sim.step()

    start_time = time.time()
    res = recommend_actions(model, sim, horizon=60, alpha=0.7)
    elapsed = (time.time() - start_time) * 1000

    print(f"Optimizer execution time: {elapsed:.2f} ms")
    print(f"Recommendations: {res['recommendations']}")

if __name__ == "__main__":
    benchmark()
