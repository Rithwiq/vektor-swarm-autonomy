from dataclasses import dataclass
import numpy as np

from .utils import clip_norm, norm


@dataclass
class AgentMemory:
    last_seen_target: np.ndarray  # shape (2,)
    last_seen_step: int
    search_waypoint: np.ndarray   # shape (2,)
    search_waypoint_age: int


class SwarmAgents:
    def __init__(self, n: int, world_size: float):
        self.n = n
        self.world_size = world_size

        self.pos = np.zeros((n, 2), dtype=float)
        self.vel = np.zeros((n, 2), dtype=float)

        self.opinion = np.zeros((n,), dtype=float)     # x_i in [-1, 1]
        self.mode = np.array(["SEARCH"] * n, dtype=object)

        self.mem = []
        for _ in range(n):
            self.mem.append(
                AgentMemory(
                    last_seen_target=np.array([world_size / 2, world_size / 2], dtype=float),
                    last_seen_step=-9999,
                    search_waypoint=np.array([world_size / 2, world_size / 2], dtype=float),
                    search_waypoint_age=10_000,
                )
            )

    def reset_random(self, margin: float = 1.0):
        ws = self.world_size
        self.pos[:, 0] = np.random.uniform(margin, ws - margin, size=(self.n,))
        self.pos[:, 1] = np.random.uniform(margin, ws - margin, size=(self.n,))
        self.vel[:, :] = 0.0
        self.opinion[:] = np.random.uniform(-0.2, 0.2, size=(self.n,))
        self.mode[:] = "SEARCH"
        for i in range(self.n):
            self.mem[i].last_seen_target = np.array([ws / 2, ws / 2], dtype=float)
            self.mem[i].last_seen_step = -9999
            self.mem[i].search_waypoint = np.array([ws / 2, ws / 2], dtype=float)
            self.mem[i].search_waypoint_age = 10_000

    def sense(
        self,
        target_pos: np.ndarray,
        r_detect: float,
        r_comm: float,
        detect_noise: float,
        obstacles,
        step_k: int,
    ):
       
        n = self.n
        detections = []
        neighbor_lists = []
        danger = np.zeros((n,), dtype=float)

        # Pairwise distances (n,n)
        dp = self.pos[:, None, :] - self.pos[None, :, :]
        dist = np.sqrt(np.sum(dp * dp, axis=2) + 1e-9)

        for i in range(n):
            # neighbors within comm radius (excluding self)
            neigh = np.where((dist[i] < r_comm) & (np.arange(n) != i))[0]
            neighbor_lists.append(neigh)

            # target detection
            d_t = norm(target_pos - self.pos[i])
            if d_t <= r_detect:
                meas = target_pos + np.random.normal(scale=detect_noise, size=(2,))
                detections.append((True, meas))
                # update memory
                self.mem[i].last_seen_target = meas.copy()
                self.mem[i].last_seen_step = step_k
            else:
                detections.append((False, None))

            # danger: based on min neighbour distance and min obstacle distance
            min_nn = np.min(dist[i][dist[i] > 1e-6]) if n > 1 else 1e9

            min_obs = 1e9
            for (ox, oy, r) in obstacles:
                d = norm(self.pos[i] - np.array([ox, oy])) - r
                min_obs = min(min_obs, d)

            # turn distances into a normalized "danger" in [0,1]
            # close = high danger
            danger_i = 0.0
            # within 1.0m neighbor
            if min_nn < 2.0:
                danger_i += max(0.0, (2.0 - min_nn) / 2.0)
            if min_obs < 2.0:
                danger_i += max(0.0, (2.0 - min_obs) / 2.0)

            danger[i] = np.clip(danger_i, 0.0, 1.5)

        return detections, neighbor_lists, danger

    def step_dynamics(self, accel_cmd: np.ndarray, dt: float, max_speed: float):
        """
        Point-mass velocity dynamics:
          v <- v + a*dt
          p <- p + v*dt
        """
        self.vel = self.vel + accel_cmd * dt
        # speed limit
        for i in range(self.n):
            self.vel[i] = clip_norm(self.vel[i], max_speed)
        self.pos = self.pos + self.vel * dt
