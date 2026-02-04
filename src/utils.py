import numpy as np


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def norm(x: np.ndarray, eps: float = 1e-9) -> float:
    return float(np.sqrt(np.sum(x * x) + eps))


def unit(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.sqrt(np.sum(x * x) + eps)
    return x / n


def clip_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    n = np.linalg.norm(v)
    if n <= max_norm:
        return v
    if n < 1e-9:
        return v
    return v * (max_norm / n)


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def in_bounds(p: np.ndarray, world_size: float) -> bool:
    return (0.0 <= p[0] <= world_size) and (0.0 <= p[1] <= world_size)


def softplus(z: float) -> float:
    # stable softplus
    if z > 30:
        return z
    return float(np.log1p(np.exp(z)))


def smoothstep(x: float, edge0: float, edge1: float) -> float:
    # 0..1 smooth step
    if edge0 == edge1:
        return 0.0
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return float(t * t * (3 - 2 * t))
