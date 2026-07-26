from simulator.machine import PaperMachine
from simulator.controller import Controller
from simulator.process import Process
from simulator.historian import Historian
from simulator.grades import get_grade


class Simulator:

    def __init__(self, grades_df):

        self.grades_df = grades_df

        self.controller = Controller()
        self.process = Process()
        self.historian = Historian()

        self.machine = None
        self.running = False
        self.max_steps = 0
        self.target_grade = None
        self.target_recipe = None

    # ==========================================================
    # Live Simulation API
    # ==========================================================

    def start_transition(self, start_grade, target_grade, duration=300, reset_historian=True):

        if reset_historian:
            self.historian.records = []
        self.controller = Controller()
        self.process = Process()

        start_recipe = get_grade(self.grades_df, start_grade)
        self.target_recipe = get_grade(self.grades_df, target_grade)

        self.machine = PaperMachine(start_recipe)
        self.machine.set_target_grade(self.target_recipe)

        self.target_grade = target_grade

        self.max_steps = duration

        self.running = True

    def step(self, enable_noise=True):

        if not self.running:
            return

        self.controller.update(self.machine)
        self.process.update(self.machine, enable_noise=enable_noise)
        self.historian.log(self.machine)

        if self.machine.time >= self.max_steps:

            self.machine.current_grade = self.target_grade
            self.machine.current_recipe = self.target_recipe

            self.running = False

    def is_running(self):

        return self.running

    def get_machine_state(self):

        if self.machine is None:
            return None

        speed_rate = 0.0
        stock_flow_rate = 0.0
        steam_rate = 0.0
        filler_rate = 0.0

        prev_record = None
        if len(self.historian.records) > 0:
            if self.historian.records[-1]["time"] == self.machine.time:
                if len(self.historian.records) >= 2:
                    prev_record = self.historian.records[-2]
            else:
                prev_record = self.historian.records[-1]

        if prev_record is not None:
            speed_rate = round(self.machine.speed - prev_record["speed"], 2)
            stock_flow_rate = round(self.machine.stock_flow - prev_record["stock_flow"], 2)
            steam_rate = round(self.machine.steam - prev_record["steam"], 2)
            filler_rate = round(self.machine.filler - prev_record["filler"], 2)

        return {
            "speed": round(self.machine.speed, 2),
            "stock_flow": round(self.machine.stock_flow, 2),
            "steam": round(self.machine.steam, 2),
            "filler": round(self.machine.filler, 2),
            "speed_rate": speed_rate,
            "stock_flow_rate": stock_flow_rate,
            "steam_rate": steam_rate,
            "filler_rate": filler_rate,
            "basis_weight": round(self.machine.basis_weight, 2),
            "moisture": round(self.machine.moisture, 2),
            "ash": round(self.machine.ash, 2),
            "caliper": round(self.machine.caliper, 2),
        }

    def apply_setpoints(self, settings):
        """
        Apply AI-recommended setpoints by updating the controller targets.
        """

        if self.machine is None:
            return

        try:
            speed_min = float(self.grades_df["machine_speed"].min())
            speed_max = float(self.grades_df["machine_speed"].max())
            stock_min = float(self.grades_df["stock_flow"].min())
            stock_max = float(self.grades_df["stock_flow"].max())
            steam_min = float(self.grades_df["steam_pressure"].min())
            steam_max = float(self.grades_df["steam_pressure"].max())
            filler_min = float(self.grades_df["filler_flow"].min())
            filler_max = float(self.grades_df["filler_flow"].max())
        except Exception:
            speed_min, speed_max = 520.0, 1200.0
            stock_min, stock_max = 260.0, 1350.0
            steam_min, steam_max = 3.5, 7.2
            filler_min, filler_max = 22.0, 180.0

        self.machine.target_recipe["machine_speed"] = max(speed_min, min(float(settings["speed"]), speed_max))
        self.machine.target_recipe["stock_flow"] = max(stock_min, min(float(settings["stock_flow"]), stock_max))
        self.machine.target_recipe["steam_pressure"] = max(steam_min, min(float(settings["steam"]), steam_max))
        self.machine.target_recipe["filler_flow"] = max(filler_min, min(float(settings["filler"]), filler_max))

    def get_history(self):

        return self.historian.records

    # ==========================================================
    # Offline Dataset Generation
    # ==========================================================

    def run_transition(self, start_grade, target_grade, duration=300):

        self.start_transition(start_grade, target_grade, duration, reset_historian=False)

        while self.is_running():
            self.step()

    def run_all_transitions(self):

        self.historian.records = []
        grades = list(self.grades_df["grade_code"])

        total = 0

        for start in grades:

            for target in grades:

                if start == target:
                    continue

                print(f"Running {start} -> {target}")

                self.run_transition(start, target)

                total += 1

        print(f"\nCompleted {total} transitions.")

        expected_rows = total * 300
        actual_rows = len(self.historian.records)
        assert actual_rows == expected_rows, f"Historian row count mismatch! Expected {expected_rows}, got {actual_rows}."

        self.historian.save()

    def get_status(self):
        return {
            "running": self.running,
            "time": self.machine.time if self.machine else 0,
            "current_grade": self.machine.current_grade if self.machine else None,
            "target_grade": self.machine.target_grade if self.machine else None,
            "state": self.get_machine_state()
        }