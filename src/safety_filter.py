import numpy as np
from .utils import norm, unit, clip_norm, smoothstep


def safety_filter_accel(
    cfg,
    agents_pos: np.ndarray,
    agents_vel: np.ndarray,
    a_nom: np.ndarray,
    obstacles,
) -> np.ndarray:
    """
    Safety filter:
      - Adds repulsive accelerations when predicted separation falls below thresholds
      - Adds obstacle repulsion
      - Adds boundary pushback
      - Then clips accel magnitude
    """
    n = agents_pos.shape[0]
    a_safe = a_nom.copy()

    # --- Pairwise collision avoidance ---
    for i in range(n):
        for j in range(i + 1, n):
            p_i, p_j = agents_pos[i], agents_pos[j]
            v_i, v_j = agents_vel[i], agents_vel[j]
            dp = p_i - p_j
            dv = v_i - v_j
            d = norm(dp)

            # Predict a short-horizon closing (1 step lookahead)
            d_pred = norm((p_i + v_i * cfg.dt) - (p_j + v_j * cfg.dt))

            # If too close (or about to be), push apart
            trigger = min(d, d_pred)
            if trigger < (cfg.d_min * 1.6):
                # strength ramps as we approach d_min
                w = smoothstep(trigger, cfg.d_min * 1.6, cfg.d_min * 0.9)  # 0..1
                dir_ij = unit(dp)
                push = (2.5 * w) * dir_ij
                a_safe[i] += push
                a_safe[j] -= push

    # --- Obstacle avoidance ---
    for i in range(n):
        p = agents_pos[i]
        for (ox, oy, r) in obstacles:
            c = np.array([ox, oy], dtype=float)
            dvec = p - c
            d = norm(dvec) - r  # distance to obstacle boundary
            if d < (cfg.d_obs * 1.8):
                w = smoothstep(d, cfg.d_obs * 1.8, cfg.d_obs * 0.8)
                a_safe[i] += (3.0 * w) * unit(dvec)

    # --- Boundary pushback ---
    # Keep agents inside [0, world_size] with a soft margin
    ws = cfg.world_size
    m = cfg.boundary_margin
    for i in range(n):
        x, y = agents_pos[i]
        push = np.zeros((2,), dtype=float)

        if x < m:
            push[0] += (m - x) * 2.0
        if x > ws - m:
            push[0] -= (x - (ws - m)) * 2.0
        if y < m:
            push[1] += (m - y) * 2.0
        if y > ws - m:
            push[1] -= (y - (ws - m)) * 2.0

        a_safe[i] += push

    # --- Clip accel ---
    for i in range(n):
        a_safe[i] = clip_norm(a_safe[i], cfg.max_accel * 1.25)

    return a_safe
