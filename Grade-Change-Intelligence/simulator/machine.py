from copy import deepcopy


class PaperMachine:
    """
    Digital Twin of the paper machine.

    Stores:
    - Manipulated variables (what the controller changes)
    - Quality variables (what the scanner measures)
    - Current recipe
    - Target recipe
    """

    def __init__(self, recipe):

        # Keep original recipes
        self.current_recipe = deepcopy(recipe)
        self.target_recipe = deepcopy(recipe)

        # Current Grade
        self.current_grade = recipe["grade_code"]
        self.target_grade = recipe["grade_code"]

        # -------------------------
        # Manipulated Variables
        # -------------------------

        self.speed = float(recipe["machine_speed"])
        self.stock_flow = float(recipe["stock_flow"])
        self.steam = float(recipe["steam_pressure"])
        self.filler = float(recipe["filler_flow"])

        # -------------------------
        # Quality Variables
        # -------------------------

        self.basis_weight = float(recipe["gsm_target"])
        self.moisture = float(recipe["moisture_target"])
        self.ash = float(recipe["ash_target"])
        self.caliper = float(recipe["caliper_target"])

        # -------------------------
        # Simulation
        # -------------------------

        self.time = 0
        self.off_spec = False

    def set_target_grade(self, recipe):
        """
        Start a grade transition.
        """

        self.target_recipe = deepcopy(recipe)
        self.target_grade = recipe["grade_code"]

    def __str__(self):

        return (
            f"Time: {self.time}s\n"
            f"Current Grade : {self.current_grade}\n"
            f"Target Grade  : {self.target_grade}\n"
            f"Speed         : {self.speed:.2f}\n"
            f"Stock Flow    : {self.stock_flow:.2f}\n"
            f"Steam         : {self.steam:.2f}\n"
            f"Filler        : {self.filler:.2f}\n"
            f"Basis Weight  : {self.basis_weight:.2f}\n"
            f"Moisture      : {self.moisture:.2f}\n"
            f"Ash           : {self.ash:.2f}\n"
            f"Caliper       : {self.caliper:.2f}\n"
            f"Off Spec      : {self.off_spec}"
        )