from simulator.grades import load_grades
from models.simulation_evaluator import SimulationEvaluator

grades = load_grades()
recipe = grades.iloc[0].to_dict()

controls = {
    "speed": recipe["machine_speed"] - 20,
    "stock_flow": recipe["stock_flow"] + 10,
    "steam": recipe["steam_pressure"] + 0.2,
    "filler": recipe["filler_flow"] - 1,
}

sim = SimulationEvaluator()

result = sim.evaluate(recipe, controls)

print(result)