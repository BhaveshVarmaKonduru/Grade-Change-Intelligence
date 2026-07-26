from simulator.grades import load_grades
from simulator.simulator import Simulator


def main():

    grades = load_grades()

    simulator = Simulator(grades)

    simulator.run_all_transitions()


if __name__ == "__main__":
    main()