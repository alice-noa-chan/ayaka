"""Scalar corrections by actual route, primitive, and requested budget band."""

import math
from collections import defaultdict

from .calibrate import MIN_QUESTIONS, fit_temperature


def path_key(kind, route, budget):
    band = "low" if budget <= 128 else ("medium" if budget <= 384 else "high")
    return f"{kind}/{route}/{band}"


class PathCalibration:
    def __init__(self, temperatures=None):
        self.temperatures = temperatures if temperatures is not None else {}
        if not isinstance(self.temperatures, dict) or any(
            not isinstance(k, str)
            or type(t) not in (int, float)
            or not math.isfinite(t)
            or not 0.05 <= t <= 20
            for k, t in self.temperatures.items()
        ):
            raise ValueError("path temperatures must be finite scalars in 0.05..20")

    @classmethod
    def fit(cls, rows):
        if not rows or any(r.get("split") != "calibration" for r in rows):
            raise ValueError("path calibration requires only reserved calibration rows")
        groups = defaultdict(list)
        for r in rows:
            groups[r["type"]].append(r)
            groups[path_key(r["type"], r["route"], r["budget"])].append(r)
        temps = {}
        for key, group in groups.items():
            if len(group) >= MIN_QUESTIONS:
                temps[key] = fit_temperature(
                    [[math.log(max(p, 1e-12)) for p in r["probs"]] for r in group],
                    [r["target"] for r in group],
                )
        return cls(temps)

    def apply(self, probs, kind, route, budget):
        t = self.temperatures.get(path_key(kind, route, budget), self.temperatures.get(kind, 1.0))
        logits = [math.log(max(p, 1e-12)) / t for p in probs]
        exp = [math.exp(x - max(logits)) for x in logits]
        return [x / sum(exp) for x in exp]
