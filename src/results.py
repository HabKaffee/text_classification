from dataclasses import dataclass


@dataclass
class Result:
    loss: float
    metric: float
