import pandas as pd


class Historian:
    """
    Stores every second of simulation data.
    """

    def __init__(self):
        self.records = []

    def log(self, machine):

        self.records.append({

            "time": machine.time,

            "current_grade": machine.current_grade,
            "target_grade": machine.target_grade,

            "speed": round(machine.speed, 2),
            "stock_flow": round(machine.stock_flow, 2),
            "steam": round(machine.steam, 2),
            "filler": round(machine.filler, 2),

            "basis_weight": round(machine.basis_weight, 2),
            "moisture": round(machine.moisture, 2),
            "ash": round(machine.ash, 2),
            "caliper": round(machine.caliper, 2),

            "off_spec": int(machine.off_spec)

        })

    def save(self, filename="data/historical_data.csv"):

        df = pd.DataFrame(self.records)

        df.to_csv(filename, index=False)

        print(f"\nDataset saved to {filename}")
        print(f"Rows generated : {len(df)}")