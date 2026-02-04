from dataclasses import dataclass
import numpy as np

from .agents import SwarmAgents
from .opinion import update_opinion
from .behaviors import behavior_accel_commands
from .safety_filter import safety_filter_accel
from .utils import clip_norm


@dataclass
class SimConfig:
    n_agents: int = 10
    world_size: float = 20.0
    dt: float = 0.08
    steps: int = 500
    max_speed: float = 1.8
    max_accel: float = 2.2

    r_comm: float = 6.0
    r_detect: float = 5.5
    detect_noise: float = 0.15

    # opinion dynamics (still useful for "cognition pressure")
    opinion_alpha: float = 0.9
    opinion_beta: float = 1.2
    opinion_gamma: float = 1.6
    opinion_delta: float = 1.0

    # safety constraints
    d_min: float = 0.9
    d_obs: float = 1.0
    boundary_margin: float = 0.8
    obstacles: list = None

    # target motion
    target_speed: float = 1.6
    target_turn_noise: float = 0.35

    # behavior params
    search_waypoint_hold: int = 55
    cover_repulsion_gain: float = 1.2
    cover_center_gain: float = 0.15
    pursuit_gain: float = 2.2
    intercept_gain: float = 4.0
    search_gain: float = 1.2
    intercept_switch_radius: float = 4.0

    # belief / distributed governance
    belief_decay: float = 0.06
    belief_consensus_rounds: int = 2
    engage_confidence: float = 0.35

    # wolf-pack governance minima
    min_interceptors: int = 2
    min_trackers: int = 2

    # mission success (does NOT freeze target)
    intercept_radius: float = 1.2


class SwarmSim:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        if self.cfg.obstacles is None:
            self.cfg.obstacles = []

        self.agents = SwarmAgents(cfg.n_agents, cfg.world_size)

        self.target_pos = np.array([cfg.world_size * 0.75, cfg.world_size * 0.75], dtype=float)
        self.target_vel = np.array([cfg.target_speed, 0.0], dtype=float)

        self.k = 0
        self.intercepted = False
        self.intercept_step = None

        # VEKTOR-S: per-agent distributed belief
        self.belief_pos = np.zeros((cfg.n_agents, 2), dtype=float)
        self.belief_prev = np.zeros((cfg.n_agents, 2), dtype=float)
        self.belief_conf = np.zeros((cfg.n_agents,), dtype=float)

    def reset(self):
        self.k = 0
        self.intercepted = False
        self.intercept_step = None

        ws = self.cfg.world_size

        # Swarm starts bottom-left (pack start)
        self.agents.reset_random(margin=1.0)
        self.agents.pos[:] = np.random.uniform(0.6, 2.2, size=(self.cfg.n_agents, 2))
        self.agents.vel[:] = 0.0

        # Target starts top-right (opposite corner)
        self.target_pos = np.array([ws - 1.2, ws - 1.2], dtype=float)
        th = np.random.uniform(0, 2 * np.pi)
        self.target_vel = self.cfg.target_speed * np.array([np.cos(th), np.sin(th)], dtype=float)

        # Belief init: aim toward target corner but low confidence
        init_belief = np.array([ws - 2.0, ws - 2.0], dtype=float)
        self.belief_pos[:] = init_belief[None, :]
        self.belief_prev[:] = init_belief[None, :]
        self.belief_conf[:] = 0.08

    def _step_target(self):
        cfg = self.cfg
        th_noise = np.random.normal(scale=cfg.target_turn_noise)
        c, s = np.cos(th_noise), np.sin(th_noise)
        R = np.array([[c, -s], [s, c]], dtype=float)

        v = R @ self.target_vel
        v = clip_norm(v, cfg.target_speed)
        p = self.target_pos + v * cfg.dt

        ws = cfg.world_size
        if p[0] < 0.6 or p[0] > ws - 0.6:
            v[0] *= -1
        if p[1] < 0.6 or p[1] > ws - 0.6:
            v[1] *= -1

        self.target_vel = v
        self.target_pos = np.clip(p, 0.2, ws - 0.2)

    def _distributed_belief_update(self, seen: np.ndarray, meas: np.ndarray, neighbor_lists):
        cfg = self.cfg
        n = cfg.n_agents

        self.belief_prev[:] = self.belief_pos

        # local measurement update
        for i in range(n):
            if seen[i] > 0.5:
                self.belief_pos[i] = 0.25 * self.belief_pos[i] + 0.75 * meas[i]
                self.belief_conf[i] = min(1.0, self.belief_conf[i] + 0.55)
            else:
                self.belief_conf[i] = max(0.0, self.belief_conf[i] - cfg.belief_decay * cfg.dt)

        # local consensus rounds (weighted by confidence)
        for _ in range(cfg.belief_consensus_rounds):
            new_pos = self.belief_pos.copy()
            new_conf = self.belief_conf.copy()

            for i in range(n):
                neigh = neighbor_lists[i]
                if len(neigh) == 0:
                    continue

                idx = np.concatenate([np.array([i]), neigh])
                confs = self.belief_conf[idx]
                w = confs + 1e-6
                w = w / np.sum(w)
                fused_pos = np.sum(self.belief_pos[idx] * w[:, None], axis=0)
                fused_conf = float(np.clip(np.mean(confs), 0.0, 1.0))

                new_pos[i] = fused_pos
                new_conf[i] = fused_conf

            self.belief_pos[:] = new_pos
            self.belief_conf[:] = new_conf

    def step(self):
        cfg = self.cfg

        # Target ALWAYS moves (even after intercept flag)
        self._step_target()

        detections, neighbor_lists, danger = self.agents.sense(
            target_pos=self.target_pos,
            r_detect=cfg.r_detect,
            r_comm=cfg.r_comm,
            detect_noise=cfg.detect_noise,
            obstacles=cfg.obstacles,
            step_k=self.k,
        )

        # seen + measurement
        seen = np.zeros((cfg.n_agents,), dtype=float)
        meas = np.zeros((cfg.n_agents, 2), dtype=float)
        for i, (s, m) in enumerate(detections):
            if s and m is not None:
                seen[i] = 1.0
                meas[i] = m
            else:
                meas[i] = self.agents.mem[i].last_seen_target

        # distributed belief fusion
        self._distributed_belief_update(seen, meas, neighbor_lists)

        # cognitive pressure via opinion dynamics 
        self.agents.opinion = update_opinion(
            x=self.agents.opinion,
            neighbor_lists=neighbor_lists,
            evidence=seen,
            danger=danger,
            dt=cfg.dt,
            alpha=cfg.opinion_alpha,
            beta=cfg.opinion_beta,
            gamma=cfg.opinion_gamma,
            delta=cfg.opinion_delta,
        )

        # behaviour with distributed governance roles
        a_nom, modes, roles, v_est = behavior_accel_commands(
            cfg=cfg,
            agents=self.agents,
            neighbor_lists=neighbor_lists,
            danger=danger,
            seen=seen,
            meas=meas,
            belief_pos=self.belief_pos,
            belief_prev=self.belief_prev,
            belief_conf=self.belief_conf,
            step_k=self.k,
        )
        self.agents.mode = np.array(modes, dtype=object)

        # safety layer
        a_safe = safety_filter_accel(
            cfg=cfg,
            agents_pos=self.agents.pos,
            agents_vel=self.agents.vel,
            a_nom=a_nom,
            obstacles=cfg.obstacles
        )

        # integrate
        self.agents.step_dynamics(
            accel_cmd=a_safe,
            dt=cfg.dt,
            max_speed=cfg.max_speed
        )
        self.agents.pos = np.clip(self.agents.pos, 0.0, cfg.world_size)

        # metrics
        min_dist = self._min_inter_agent_dist()
        dists_to_target = np.linalg.norm(self.agents.pos - self.target_pos[None, :], axis=1)
        avg_t_dist = float(np.mean(dists_to_target))
        min_t = float(np.min(dists_to_target))
        seen_frac = float(np.mean(seen))
        engaged_frac = float(np.mean(np.array(modes) != "SEARCH"))

        if (not self.intercepted) and (min_t <= cfg.intercept_radius):
            self.intercepted = True
            self.intercept_step = self.k

        self.k += 1

        return {
            "pos": self.agents.pos.copy(),
            "vel": self.agents.vel.copy(),
            "danger": danger.copy(),
            "target_pos": self.target_pos.copy(),
            "min_dist": float(min_dist),
            "avg_target_dist": avg_t_dist,
            "min_target_dist": min_t,
            "seen_frac": seen_frac,
            "engaged_frac": engaged_frac,

            
            "seen": seen.copy(),
            "meas": meas.copy(),
            "belief_pos": self.belief_pos.copy(),
            "belief_conf": self.belief_conf.copy(),
            "mode": np.array(modes, dtype=object),
            "role": np.array(roles, dtype=object),
            "v_est": v_est.copy(),

            "intercepted": bool(self.intercepted),
            "intercept_step": self.intercept_step if self.intercept_step is not None else -1,
        }

    def _min_inter_agent_dist(self) -> float:
        p = self.agents.pos
        n = p.shape[0]
        if n < 2:
            return 1e9
        dp = p[:, None, :] - p[None, :, :]
        dist = np.sqrt(np.sum(dp * dp, axis=2) + 1e-9)
        dist[dist < 1e-6] = 1e9
        return float(np.min(dist))

    def run(self):
        history = []
        for _ in range(self.cfg.steps):
            history.append(self.step())
        return history
