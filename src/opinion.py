import numpy as np
from .utils import clamp


def update_opinion(
    x: np.ndarray,
    neighbor_lists,
    evidence: np.ndarray,
    danger: np.ndarray,
    dt: float,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
) -> np.ndarray:
    """
    Opinion dynamics (continuous-time style Euler step):

      dx_i = -alpha * x_i
             + beta * sum_{j in N(i)} (x_j - x_i)
             + gamma * (2*evidence_i - 1)   # + if sees target, - if not
             - delta * danger_i

    x clipped to [-1, 1].
    """
    n = x.shape[0]
    dx = np.zeros_like(x)

    for i in range(n):
        neigh = neighbor_lists[i]
        consensus = 0.0
        if len(neigh) > 0:
            consensus = float(np.sum(x[neigh] - x[i]))

        ev_term = (2.0 * float(evidence[i]) - 1.0)  # in {-1, +1}
        dx[i] = (
            -alpha * x[i]
            + beta * consensus
            + gamma * ev_term
            - delta * float(danger[i])
        )

    x_new = x + dt * dx
    x_new = np.clip(x_new, -1.0, 1.0)
    return x_new


def decide_modes(
    x: np.ndarray,
    threshold: float,
) -> np.ndarray:

    modes = np.array(["SEARCH"] * len(x), dtype=object)
    modes[x > threshold] = "TRACK"
    return modes
