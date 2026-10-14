#!/usr/bin/env python3
"""Direct coupled-ODE reference for the regularized Landau waveform experiment.

At every Runge--Kutta stage this script recomputes the exact finite-feature
Tikhonov score from the *current* particles.  It therefore integrates

    dot X = F(X; S_lambda(mu_X))

directly, rather than by an outer frozen-score waveform iteration.  The script
compares this monolithic reference with the converged waveform stored by
``waveform_rff_experiment.py`` and performs an independent time-grid refinement.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from waveform_rff_experiment import (
    Config,
    FourierScore,
    initial_particles,
    invariant_diagnostics,
    make_model,
    mean_particle_distance,
    rhs,
)


Array = np.ndarray
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CoupledSolve:
    path: Array
    max_score_normal_residual: float


def coupled_rhs(
    particles: Array, model: FourierScore, ridge_lambda: float
) -> tuple[Array, float]:
    coefficients, info = model.minimize_hyvarinen(particles, ridge_lambda)
    velocity = rhs(particles, coefficients, model)
    return velocity, float(info["normal_equation_relative_residual"])


def solve_monolithic(
    particles0: Array,
    model: FourierScore,
    ridge_lambda: float,
    final_time: float,
    n_steps: int,
) -> CoupledSolve:
    """Classical RK4 with an exact current-state score solve at all four stages."""
    dt = final_time / n_steps
    path = np.empty((n_steps + 1,) + particles0.shape, dtype=float)
    path[0] = particles0
    max_residual = 0.0
    for n in range(n_steps):
        x = path[n]
        k1, r1 = coupled_rhs(x, model, ridge_lambda)
        k2, r2 = coupled_rhs(x + 0.5 * dt * k1, model, ridge_lambda)
        k3, r3 = coupled_rhs(x + 0.5 * dt * k2, model, ridge_lambda)
        k4, r4 = coupled_rhs(x + dt * k3, model, ridge_lambda)
        path[n + 1] = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        max_residual = max(max_residual, r1, r2, r3, r4)
    return CoupledSolve(path=path, max_score_normal_residual=max_residual)


def write_rows(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()

    summary_path = output / "run_summary.json"
    waveform_path = output / "raw_waveforms.npz"
    if not summary_path.exists() or not waveform_path.exists():
        raise FileNotFoundError(
            "run waveform_rff_experiment.py before the monolithic reference"
        )

    metadata = json.loads(summary_path.read_text(encoding="utf-8"))
    config = Config(**metadata["config"])
    particles0 = initial_particles(config)
    model = make_model(config)
    stored = np.load(waveform_path)
    waveform = np.asarray(stored["waveforms"])[-1]
    times = np.asarray(stored["times"])

    step_counts = [config.n_steps // 2, config.n_steps, 2 * config.n_steps]
    if config.n_steps % 2:
        raise ValueError("the baseline step count must be even")
    solves: dict[int, CoupledSolve] = {}
    for steps in step_counts:
        print(f"monolithic coupled solve: n_steps={steps}", flush=True)
        solves[steps] = solve_monolithic(
            particles0, model, config.ridge_lambda, config.final_time, steps
        )

    baseline = solves[config.n_steps]
    waveform_error = mean_particle_distance(waveform, baseline.path)
    error_rows = [
        {
            "time": float(t),
            "waveform_to_monolithic_mean_particle_error": float(error),
        }
        for t, error in zip(times, waveform_error)
    ]
    write_rows(output / "monolithic_reference_by_time.csv", error_rows)

    finest_steps = step_counts[-1]
    finest = solves[finest_steps].path
    refinement_rows: list[dict[str, float | int]] = []
    for steps in step_counts:
        solve = solves[steps]
        stride = finest_steps // steps
        error_to_finest = mean_particle_distance(solve.path, finest[::stride])
        invariants = invariant_diagnostics(solve.path)
        refinement_rows.append(
            {
                "n_steps": steps,
                "dt": config.final_time / steps,
                "path_error_to_finest_sup": (
                    0.0 if steps == finest_steps else float(np.max(error_to_finest))
                ),
                "momentum_drift_sup": float(invariants["momentum_drift_sup"]),
                "energy_drift_sup": float(invariants["energy_drift_sup"]),
                "max_score_normal_residual": solve.max_score_normal_residual,
            }
        )
    write_rows(output / "monolithic_refinement.csv", refinement_rows)

    coarse_error = float(refinement_rows[0]["path_error_to_finest_sup"])
    base_error = float(refinement_rows[1]["path_error_to_finest_sup"])
    observed_order = (
        math.log(coarse_error / base_error, 2.0)
        if coarse_error > 0.0 and base_error > 0.0
        else float("nan")
    )
    inv = invariant_diagnostics(baseline.path)
    summary = {
        "baseline_steps": config.n_steps,
        "waveform_to_monolithic_path_error_sup": float(np.max(waveform_error)),
        "waveform_to_monolithic_terminal_error": float(waveform_error[-1]),
        "monolithic_max_score_normal_residual": baseline.max_score_normal_residual,
        "monolithic_momentum_drift_sup": float(inv["momentum_drift_sup"]),
        "monolithic_energy_drift_sup": float(inv["energy_drift_sup"]),
        "refinement_observed_order": observed_order,
        "checks": {
            "waveform_reference_error_below_2e-6": bool(
                np.max(waveform_error) < 2.0e-6
            ),
            "monolithic_score_residual_below_1e-10": bool(
                baseline.max_score_normal_residual < 1.0e-10
            ),
            "monolithic_refinement_order_above_3p5": bool(observed_order > 3.5),
        },
    }
    summary["passed"] = all(summary["checks"].values())
    (output / "monolithic_reference_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    axes[0].semilogy(
        times,
        np.maximum(waveform_error, 1.0e-18),
        label="converged waveform vs monolithic",
    )
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("mean-particle path error")
    axes[0].set_title("Independent coupled-ODE reference")
    axes[0].legend(fontsize=8)

    dt = np.asarray([float(row["dt"]) for row in refinement_rows[:-1]])
    errors = np.asarray(
        [float(row["path_error_to_finest_sup"]) for row in refinement_rows[:-1]]
    )
    axes[1].loglog(
        dt,
        errors,
        "o-",
        label=f"effective order vs finest {observed_order:.2f}",
    )
    axes[1].invert_xaxis()
    axes[1].set_xlabel("dt")
    axes[1].set_ylabel("monolithic path error to finest")
    axes[1].set_title("Monolithic time refinement")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "monolithic_reference.png", dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    return 0 if (summary["passed"] or not args.strict) else 1


if __name__ == "__main__":
    raise SystemExit(main())
