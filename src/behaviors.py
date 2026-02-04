import numpy as np
from .utils import clip_norm, norm


def pick_search_waypoints(agents, world_size: float, hold_steps: int):
    for mem in agents.mem:
        mem.search_waypoint_age += 1
        if mem.search_waypoint_age >= hold_steps:
            mem.search_waypoint = np.random.uniform(1.0, world_size - 1.0, size=(2,))
            mem.search_waypoint_age = 0


def _estimate_velocity(belief_pos: np.ndarray, belief_prev: np.ndarray, dt: float) -> np.ndarray:
    v = (belief_pos - belief_prev) / max(dt, 1e-6)
    spd = np.linalg.norm(v, axis=1) + 1e-9
    vmax = 3.0
    scale = np.minimum(1.0, vmax / spd)
    return v * scale[:, None]


def _merge_topk(a, b, k):
    m = a + b
   
    best = {}
    for s, idx in m:
        if (idx not in best) or (s > best[idx]):
            best[idx] = s
    items = [(best[idx], idx) for idx in best]
    items.sort(key=lambda x: (x[0], -x[1]), reverse=True)  # high score first; stable-ish
    return items[:k]


def distributed_topk(scores: np.ndarray, neighbor_lists, k: int, rounds: int = 4):
    n = scores.shape[0]
    state = [[(float(scores[i]), int(i))] for i in range(n)]

    for _ in range(rounds):
        new_state = [lst[:] for lst in state]
        for i in range(n):
            merged = state[i][:]
            for j in neighbor_lists[i]:
                merged = _merge_topk(merged, state[j], k)
            new_state[i] = merged
        state = new_state

    # each agent's view of elected ids
    elected = [set(idx for _, idx in state[i]) for i in range(n)]
    return elected, state


def behavior_accel_commands(
    cfg,
    agents,
    neighbor_lists,
    danger: np.ndarray,
    seen: np.ndarray,
    meas: np.ndarray,
    belief_pos: np.ndarray,
    belief_prev: np.ndarray,
    belief_conf: np.ndarray,
    step_k: int,
):

    n = agents.n
    a_cmd = np.zeros((n, 2), dtype=float)

    pick_search_waypoints(agents, cfg.world_size, cfg.search_waypoint_hold)

    v_est = _estimate_velocity(belief_pos, belief_prev, cfg.dt)

    # Engagement decision is local (distributed) based on each agent's belief confidence.
    # Agents can engage at different times depending on info flow.
    engaged = belief_conf >= cfg.engage_confidence

    roles = np.array(["SCOUT"] * n, dtype=object)
    modes = np.array(["SEARCH"] * n, dtype=object)

    # Governance: distributed election of minimum roles
    # Tracker score: high confidence + close to belief
    dist_to_b = np.linalg.norm(agents.pos - belief_pos, axis=1)
    tracker_score = (belief_conf + 1e-6) / (dist_to_b + 0.35)

    # Interceptor score: ahead along estimated velocity direction (cut-off)
    vhat = v_est.copy()
    vnorm = np.linalg.norm(vhat, axis=1) + 1e-9
    vhat = vhat / vnorm[:, None]
    rel = agents.pos - belief_pos
    ahead = np.sum(rel * vhat, axis=1)  # projection
    interceptor_score = (belief_conf + 1e-6) * ahead

    # Only allow role elections for agents that are engaged (otherwise stay SCOUT)
    tracker_score_eff = tracker_score * engaged.astype(float)
    interceptor_score_eff = interceptor_score * engaged.astype(float)

    # distributed elections (everyone reaches same winners if connected enough)
    elected_trackers, _ = distributed_topk(tracker_score_eff, neighbor_lists, k=cfg.min_trackers, rounds=4)
    elected_interceptors, _ = distributed_topk(interceptor_score_eff, neighbor_lists, k=cfg.min_interceptors, rounds=4)

    # Each agent uses its own elected sets (distributed viewpoint).
    for i in range(n):
        if not engaged[i]:
            roles[i] = "SCOUT"
            modes[i] = "SEARCH"
            continue

        if i in elected_trackers[i]:
            roles[i] = "TRACKER"
        elif i in elected_interceptors[i]:
            roles[i] = "INTERCEPTOR"
        else:
            roles[i] = "SCOUT"

        # Mode: INTERCEPT if close enough, else PURSUIT
        if norm(belief_pos[i] - agents.pos[i]) <= cfg.intercept_switch_radius:
            modes[i] = "INTERCEPT"
        else:
            modes[i] = "PURSUIT"

    # --- Control policies per role ---
    for i in range(n):
        if modes[i] == "SEARCH":
            goal = agents.mem[i].search_waypoint
            k_gain = cfg.search_gain

        else:
            if roles[i] == "TRACKER":
                # if I see target, use my measurement for snappier pursuit
                goal = meas[i] if seen[i] > 0.5 else belief_pos[i]
                k_gain = cfg.pursuit_gain if modes[i] == "PURSUIT" else cfg.intercept_gain

            elif roles[i] == "INTERCEPTOR":
                v = v_est[i]
                spd = norm(v)
                if spd < 0.25:
                    goal = belief_pos[i]
                else:
                    tau = 1.5
                    goal = belief_pos[i] + v * tau
                k_gain = cfg.pursuit_gain if modes[i] == "PURSUIT" else cfg.intercept_gain

            else:
                # SCOUT but engaged: keep searching but bias toward belief region to help reacquisition
                goal = 0.75 * agents.mem[i].search_waypoint + 0.25 * belief_pos[i]
                k_gain = 0.9 * cfg.search_gain

        # PD-like accel toward goal
        pos_err = goal - agents.pos[i]
        a = k_gain * pos_err - 0.6 * agents.vel[i]

        # damp aggressiveness under danger 
        a *= (1.0 / (1.0 + 1.5 * float(danger[i])))

        a_cmd[i] = clip_norm(a, cfg.max_accel)

    return a_cmd, modes.tolist(), roles.tolist(), v_est
