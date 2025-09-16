from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class LadderAssessment:
    classification: str  # OK / Too short / Too long / Piling
    delta_stats_m: dict
    strikes: int
    piling_risk: float
    recommended_L_m: float | None
    recommended_rungs: int | None


def classify_ladder(delta_m: np.ndarray, strike_mask: np.ndarray | None = None,
                    s_m: float = 0.31) -> LadderAssessment:
    if len(delta_m) == 0:
        return LadderAssessment("Unknown", {}, 0, 0.0, None, None)
    abs_delta = np.abs(delta_m)
    mean = float(np.mean(delta_m))
    p95 = float(np.percentile(abs_delta, 95))
    maxv = float(np.max(abs_delta))
    strikes = int(np.count_nonzero(strike_mask)) if strike_mask is not None else 0
    piling = float(np.mean(abs_delta < 0.05))
    # thresholds per spec
    if strikes > 0:
        cls = "Too long"
    elif p95 <= 0.15:
        cls = "OK"
    elif p95 <= 0.30:
        cls = "Caution"
    else:
        cls = "Too short"
    # simple recommendation: shift ladder by nearest rung multiples to minimize |mean|
    rungs = int(round(-mean / s_m))
    Lm = float(rungs * s_m)
    return LadderAssessment(
        classification=cls,
        delta_stats_m={"mean": mean, "p95": p95, "max": maxv},
        strikes=strikes,
        piling_risk=piling,
        recommended_L_m=Lm,
        recommended_rungs=rungs,
    )


