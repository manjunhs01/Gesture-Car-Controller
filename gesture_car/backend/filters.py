"""Signal smoothing: One Euro filter (adaptive low-pass) and a plain exponential moving average."""

from __future__ import annotations

import math
from typing import Protocol


class Smoother(Protocol):
    def update(self, value: float, t: float) -> float: ...
    def reset(self) -> None: ...

    @property
    def velocity(self) -> float:
        """Estimated rate of change (units per second) of the filtered value; 0 if the filter does not estimate it."""
        ...


def _alpha(cutoff_hz: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Casiez et al. 2012. Low cutoff removes jitter at rest; the cutoff rises with speed so fast turns do not lag."""

    def __init__(self, min_cutoff: float = 1.5, beta: float = 0.05, d_cutoff: float = 1.0) -> None:
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self) -> None:
        self._x: float | None = None
        self._dx = 0.0
        self._t = 0.0

    @property
    def velocity(self) -> float:
        return self._dx

    def update(self, value: float, t: float) -> float:
        if self._x is None:
            self._x, self._t = value, t
            return value
        dt = max(t - self._t, 1e-3)
        self._t = t
        raw_dx = (value - self._x) / dt
        self._dx += _alpha(self.d_cutoff, dt) * (raw_dx - self._dx)
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        self._x += _alpha(cutoff, dt) * (value - self._x)
        return self._x


class EmaFilter:
    velocity = 0.0  # no rate estimate: no prediction

    def __init__(self, alpha: float = 0.4) -> None:
        self.alpha = alpha
        self.reset()

    def reset(self) -> None:
        self._x: float | None = None

    def update(self, value: float, t: float = 0.0) -> float:
        self._x = value if self._x is None else self._x + self.alpha * (value - self._x)
        return self._x
