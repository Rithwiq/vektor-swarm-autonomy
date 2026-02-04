from pathlib import Path

from src.sim import SwarmSim, SimConfig
from src.viz import render_gif, plot_metrics
from src.utils import set_seed


def main():
    # ---------- Paths ----------
    root = Path(__file__).parent
    results_dir = root / "results"
    docs_dir = root / "docs"
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Reproducibility ----------
    set_seed(7)

    # ---------- Config (VEKTOR-S) ----------
    cfg = SimConfig(
        n_agents=10,
        world_size=20.0,
        dt=0.08,
        steps=500,              

        # Dynamics 
        max_speed=2.2,
        max_accel=2.8,

        # Comms + sensing 
        r_comm=7.0,
        r_detect=6.0,
        detect_noise=0.15,

        # Opinion dynamics 
        opinion_alpha=0.9,
        opinion_beta=1.2,
        opinion_gamma=1.6,
        opinion_delta=1.0,

        # Safety constraints
        d_min=0.9,
        d_obs=1.0,
        boundary_margin=0.8,
        obstacles=[
            (6.5, 10.0, 1.4),
            (13.0, 7.0, 1.2),
            (10.0, 14.0, 1.0),
        ],

        # Moving target 
        target_speed=1.8,
        target_turn_noise=0.35,

        # Search behaviour
        search_waypoint_hold=55,
        search_gain=1.2,

        # Pursuit / intercept behaviour 
        pursuit_gain=2.0,
        intercept_gain=4.6,
        intercept_switch_radius=5.0,

        # Belief / engagement 
        engage_confidence=0.28,
        belief_decay=0.045,
        belief_consensus_rounds=3,

        # Swarm minimum roles
        min_trackers=2,
        min_interceptors=3,

        # “contact achieved” radius (metric + HUD)
        intercept_radius=1.2,
    )

    sim = SwarmSim(cfg)
    sim.reset()

    print("Running simulation...")
    history = sim.run()

    gif_path = results_dir / "demo.gif"
    metrics_path = results_dir / "metrics.png"

    print(f"Rendering GIF -> {gif_path}")
    render_gif(cfg, history, gif_path.as_posix(), every_n=2)

    print(f"Saving metrics plot -> {metrics_path}")
    plot_metrics(cfg, history, metrics_path.as_posix())

    print("\nDone.")
    print(f"- GIF:     {gif_path}")
    print(f"- Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
