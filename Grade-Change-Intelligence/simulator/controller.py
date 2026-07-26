from config import (
    SPEED_RATE,
    STOCK_RATE,
    STEAM_RATE,
    FILLER_RATE,
)


class Controller:
    """
    Simulates gradual controller action during a grade change with adaptive slew-rate scaling.
    """

    @staticmethod
    def _move(current, target, base_rate, transition_time=60.0):
        """
        Move current value towards target using adaptive rate scaling for large grade transitions.
        """
        delta = abs(target - current)
        if delta <= 1e-6:
            return target

        effective_rate = max(base_rate, delta / transition_time)

        if delta <= effective_rate:
            return target

        if current < target:
            return current + effective_rate

        return current - effective_rate

    def update(self, machine):
        """
        Update all manipulated variables towards target recipe setpoints.
        """
        recipe = machine.target_recipe

        machine.speed = self._move(
            machine.speed,
            recipe["machine_speed"],
            SPEED_RATE,
        )

        machine.stock_flow = self._move(
            machine.stock_flow,
            recipe["stock_flow"],
            STOCK_RATE,
        )

        machine.steam = self._move(
            machine.steam,
            recipe["steam_pressure"],
            STEAM_RATE,
        )

        machine.filler = self._move(
            machine.filler,
            recipe["filler_flow"],
            FILLER_RATE,
        )