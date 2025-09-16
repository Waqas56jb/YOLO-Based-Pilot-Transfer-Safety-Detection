from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class Prescription:
    L_star_m: float
    rung_steps: int
    utility: float


def evaluate_utility(delta_m: np.ndarray, strike_mask: np.ndarray | None = None,
                     w: Tuple[float, float, float, float] = (1.0, 5.0, 2.0, 1.0)) -> float:
    # w1·TimeInGreen − w2·DeckStrikes − w3·PilingRisk − w4·Max|Δ|
    w1, w2, w3, w4 = w
    green = float(np.mean(np.abs(delta_m) <= 0.15))
    strikes = float(np.count_nonzero(strike_mask)) if strike_mask is not None else 0.0
    piling = float(np.mean(np.abs(delta_m) < 0.05))
    maxd = float(np.max(np.abs(delta_m))) if len(delta_m) else 0.0
    return w1 * green - w2 * strikes - w3 * piling - w4 * maxd


def prescribe(delta_m: np.ndarray, s_m: float = 0.31) -> Prescription:
    # try shifts L in [-3, 3] rungs
    best_u = -1e9
    best_L = 0.0
    best_r = 0
    for r in range(-3, 4):
        shifted = delta_m + r * s_m
        u = evaluate_utility(shifted)
        if u > best_u:
            best_u = u
            best_L = r * s_m
            best_r = r
    return Prescription(L_star_m=float(best_L), rung_steps=best_r, utility=float(best_u))


