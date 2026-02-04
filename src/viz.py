import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D


def render_gif(cfg, history, out_path: str, every_n: int = 2):
    frames = history[::every_n]
    n = cfg.n_agents
    ws = cfg.world_size

    fig, ax = plt.subplots(figsize=(7.4, 7.4))
    ax.set_xlim(0, ws)
    ax.set_ylim(0, ws)
    ax.set_aspect("equal", "box")
    ax.set_title("VEKTOR-S (Swarm): Distributed Swarm | Search -> Pursuit -> Intercept (Safety-Filtered)")
    plt.subplots_adjust(bottom=0.28)

    # Obstacles
    for (ox, oy, r) in cfg.obstacles:
        ax.add_patch(plt.Circle((ox, oy), r, fill=False, linewidth=2))

    # Boundary
    ax.plot([0, ws, ws, 0, 0], [0, 0, ws, ws, 0], linewidth=2)

    # Target + intercept ring
    target_scat = ax.scatter([], [], s=110, marker="x")
    intercept_circle = plt.Circle((0, 0), cfg.intercept_radius, fill=False, linewidth=2, alpha=0.35)
    ax.add_patch(intercept_circle)

    # Sensor rings
    sensor_circles = []
    for _ in range(n):
        c = plt.Circle((0, 0), cfg.r_detect, fill=False, linewidth=1, alpha=0.07)
        ax.add_patch(c)
        sensor_circles.append(c)

    # Perception lines
    sense_lines = [ax.plot([], [], "--", linewidth=1, alpha=0.22)[0] for _ in range(n)]

    # Belief points 
    belief_scat = ax.scatter([], [], s=18, alpha=0.30)

    # Role scatters (marker by agent role)
    scat_scout = ax.scatter([], [], s=55, marker="s")       # SCOUT = square
    scat_tracker = ax.scatter([], [], s=70, marker="o")     # TRACKER = circle
    scat_interceptor = ax.scatter([], [], s=85, marker="^") # INTERCEPTOR = triangle

    # Bottom HUD
    hud1 = ax.text(0.02, -0.08, "", transform=ax.transAxes)
    hud2 = ax.text(0.02, -0.13, "", transform=ax.transAxes, fontsize=12)

    # Legend 
    legend_elems = [
        Line2D([0], [0], marker="s", linestyle="None", label="SCOUT (search)", markersize=8),
        Line2D([0], [0], marker="o", linestyle="None", label="TRACKER (pursue)", markersize=8),
        Line2D([0], [0], marker="^", linestyle="None", label="INTERCEPTOR (cut-off)", markersize=8),
        Line2D([0], [0], color="deepskyblue", lw=3, label="SEARCH mode"),
        Line2D([0], [0], color="orange", lw=3, label="PURSUIT mode"),
        Line2D([0], [0], color="red", lw=3, label="INTERCEPT mode"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", bbox_to_anchor=(1.0, -0.02), frameon=False)

    def mode_color(mode: str) -> str:
        if mode == "INTERCEPT":
            return "red"
        if mode == "PURSUIT":
            return "orange"
        return "deepskyblue"

    def init():
        target_scat.set_offsets(np.zeros((1, 2)))
        intercept_circle.center = (-10, -10)

        belief_scat.set_offsets(np.zeros((n, 2)))

        scat_scout.set_offsets(np.zeros((0, 2)))
        scat_tracker.set_offsets(np.zeros((0, 2)))
        scat_interceptor.set_offsets(np.zeros((0, 2)))

        for i in range(n):
            sensor_circles[i].center = (-10, -10)
            sense_lines[i].set_data([], [])

        hud1.set_text("")
        hud2.set_text("")
        return [
            target_scat, intercept_circle,
            belief_scat, scat_scout, scat_tracker, scat_interceptor,
            hud1, hud2, *sensor_circles, *sense_lines
        ]

    def update(frame_idx):
        f = frames[frame_idx]
        pos = f["pos"]
        tpos = f["target_pos"]
        seen = f.get("seen", np.zeros((n,), dtype=float))
        belief = f.get("belief_pos", pos.copy())
        conf = f.get("belief_conf", np.zeros((n,), dtype=float))
        modes = f.get("mode", np.array(["SEARCH"] * n, dtype=object))
        roles = f.get("role", np.array(["SCOUT"] * n, dtype=object))
        mind = float(f.get("min_dist", 0.0))
        min_t = float(f.get("min_target_dist", 999.0))
        intercepted = bool(f.get("intercepted", False))

        # target + intercept ring
        target_scat.set_offsets(tpos[None, :])
        intercept_circle.center = (tpos[0], tpos[1])

        # belief points
        belief_scat.set_offsets(belief)

        # sensor rings
        for i in range(n):
            sensor_circles[i].center = (pos[i, 0], pos[i, 1])

        # perception lines
        for i, line in enumerate(sense_lines):
            if seen[i] > 0.5:
                line.set_data([pos[i, 0], tpos[0]], [pos[i, 1], tpos[1]])
            else:
                line.set_data([], [])

        # split by role
        idx_s = np.where(roles == "SCOUT")[0]
        idx_t = np.where(roles == "TRACKER")[0]
        idx_i = np.where(roles == "INTERCEPTOR")[0]

        scat_scout.set_offsets(pos[idx_s] if len(idx_s) else np.zeros((0, 2)))
        scat_tracker.set_offsets(pos[idx_t] if len(idx_t) else np.zeros((0, 2)))
        scat_interceptor.set_offsets(pos[idx_i] if len(idx_i) else np.zeros((0, 2)))

        scat_scout.set_color([mode_color(modes[j]) for j in idx_s])
        scat_tracker.set_color([mode_color(modes[j]) for j in idx_t])
        scat_interceptor.set_color([mode_color(modes[j]) for j in idx_i])

        # HUD
        tsec = frame_idx * every_n * cfg.dt
        seen_pct = 100.0 * float(np.mean(seen))
        conf_avg = float(np.mean(conf))
        c_search = int(np.sum(modes == "SEARCH"))
        c_pursuit = int(np.sum(modes == "PURSUIT"))
        c_intercept = int(np.sum(modes == "INTERCEPT"))

        r_s = int(np.sum(roles == "SCOUT"))
        r_t = int(np.sum(roles == "TRACKER"))
        r_i = int(np.sum(roles == "INTERCEPTOR"))

        hud1.set_text(
            f"t={tsec:5.1f}s | min_dist={mind:4.2f} (d_min={cfg.d_min}) | "
            f"min_target={min_t:4.2f} (r_int={cfg.intercept_radius}) | "
            f"SEEN={seen_pct:.0f}% | belief_conf(avg)={conf_avg:.2f}"
        )

        if intercepted:
            hud2.set_text("INTERCEPT ACHIEVED (target continues moving - swarm achieved contact)")
        else:
            hud2.set_text(
                f"Modes: SEARCH={c_search} PURSUIT={c_pursuit} INTERCEPT={c_intercept}   |   "
                f"Agent roles: SCOUT={r_s} TRACKER={r_t} INTERCEPTOR={r_i}   "
                f"(min TRACKER={cfg.min_trackers}, min INTERCEPTOR={cfg.min_interceptors})"
            )

        return [
            target_scat, intercept_circle,
            belief_scat, scat_scout, scat_tracker, scat_interceptor,
            hud1, hud2, *sensor_circles, *sense_lines
        ]

    ani = FuncAnimation(fig, update, frames=len(frames), init_func=init, blit=True, interval=50)
    ani.save(out_path, writer=PillowWriter(fps=18))
    plt.close(fig)


def plot_metrics(cfg, history, out_path: str):
    import numpy as np
    import matplotlib.pyplot as plt

    min_d = np.array([h["min_dist"] for h in history], dtype=float)
    avg_td = np.array([h["avg_target_dist"] for h in history], dtype=float)
    min_td = np.array([h.get("min_target_dist", np.nan) for h in history], dtype=float)
    seen_frac = np.array([h["seen_frac"] for h in history], dtype=float)
    engaged_frac = np.array([h.get("engaged_frac", 0.0) for h in history], dtype=float)

    t = np.arange(len(history)) * cfg.dt
    margin = min_d - cfg.d_min

    fig = plt.figure(figsize=(9, 7))
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    ax1.plot(t, min_d, label="min inter-agent distance")
    ax1.axhline(cfg.d_min, linestyle="--", label="d_min")
    ax1.set_ylabel("Distance (m)")
    ax1.set_title("VEKTOR-S: Safety + Task Metrics")
    ax1.legend()

    ax2.plot(t, margin, label="safety margin (min_dist - d_min)")
    ax2.axhline(0.0, linestyle="--", label="zero margin")
    ax2.set_ylabel("Margin (m)")
    ax2.legend()

    ax3.plot(t, avg_td, label="avg distance to target")
    ax3.plot(t, min_td, label="min distance to target")
    scale = float(np.nanmax(avg_td)) if np.nanmax(avg_td) > 1e-6 else 1.0
    ax3.plot(t, engaged_frac * scale, label="engaged fraction (scaled)")
    ax3.plot(t, seen_frac * scale, label="seen fraction (scaled)")
    ax3.axhline(cfg.intercept_radius, linestyle="--", label="intercept radius")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Distance / scaled fractions")
    ax3.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
