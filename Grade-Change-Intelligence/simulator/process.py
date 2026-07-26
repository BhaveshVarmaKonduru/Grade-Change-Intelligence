import random
from collections import deque

from config import (
    BW_GAIN,
    MOISTURE_GAIN,
    ASH_GAIN,
    CALIPER_GAIN,
    NOISE_STD,
    BW_TOLERANCE,
)


class Process:
    """
    Simulates how the paper machine responds to controller actions.
    """

    def __init__(self):
        self.bw_tau = 25.0
        self.moisture_tau = 20.0
        self.ash_tau = 35.0
        self.caliper_tau = 30.0
        self.transport_delay = 8  # seconds
        self.stock_buffer = deque()

    def first_order(self, current, target, tau):
        """
        First-order process response.
        """
        return current + (target - current) / tau

    def update(self, machine, enable_noise=True):
        """
        Update process quality variables using first-order dynamics.
        If enable_noise=False, simulates clean expected trajectory without stochastic noise.
        """
        disturbance = random.uniform(-0.5, 0.5) if enable_noise else 0.0

        recipe = machine.target_recipe

        # -------------------------
        # Transport Delay
        # -------------------------
        self.stock_buffer.append(machine.stock_flow)

        if len(self.stock_buffer) <= self.transport_delay:
            delayed_stock = machine.stock_flow
        else:
            delayed_stock = self.stock_buffer.popleft()

        # -------------------------
        # Basis Weight
        # -------------------------
        bw_target = (
            recipe["gsm_target"]
            + (delayed_stock - recipe["stock_flow"]) * BW_GAIN
            + disturbance
        )

        machine.basis_weight = self.first_order(
            machine.basis_weight,
            bw_target,
            self.bw_tau,
        )

        if enable_noise:
            machine.basis_weight += random.gauss(0, NOISE_STD)

        # -------------------------
        # Moisture
        # -------------------------
        moisture_target = (
            recipe["moisture_target"]
            - (machine.steam - recipe["steam_pressure"]) * MOISTURE_GAIN
        )

        machine.moisture = self.first_order(
            machine.moisture,
            moisture_target,
            self.moisture_tau,
        )

        if enable_noise:
            machine.moisture += random.gauss(0, NOISE_STD / 5)

        # -------------------------
        # Ash
        # -------------------------
        ash_target = (
            recipe["ash_target"]
            + (machine.filler - recipe["filler_flow"]) * ASH_GAIN
        )

        machine.ash = self.first_order(
            machine.ash,
            ash_target,
            self.ash_tau,
        )

        if enable_noise:
            machine.ash += random.gauss(0, NOISE_STD / 4)

        # -------------------------
        # Caliper
        # -------------------------
        caliper_target = (
            recipe["caliper_target"]
            + (machine.basis_weight - recipe["gsm_target"]) * CALIPER_GAIN
        )

        machine.caliper = self.first_order(
            machine.caliper,
            caliper_target,
            self.caliper_tau,
        )

        if enable_noise:
            machine.caliper += random.gauss(0, NOISE_STD)

        # -------------------------
        # Off-Spec Detection
        # -------------------------
        error_percent = (
            abs(machine.basis_weight - recipe["gsm_target"])
            / recipe["gsm_target"]
        ) * 100

        machine.off_spec = error_percent > BW_TOLERANCE

        machine.time += 1