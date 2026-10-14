#!/usr/bin/env python3
"""Reproducible 2D alternating score--system waveform experiment.

The experiment is a finite-dimensional special case of the score-space setting
in the accompanying paper. A fixed random Fourier feature (RFF) space is used for the
score.  At every time node the Tikhonov-regularized empirical Hyvarinen
functional is minimized by solving its normal equations exactly.  With that
entire score waveform frozen, the *new* particle positions are advanced by RK4.

Only NumPy and Matplotlib are required.  All random choices are made once from
explicit seeds; the same initial particles and features are used by every outer
iterate and by every optional time-grid refinement run.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import time
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


Array = np.ndarray
ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs"


@dataclass(frozen=True)
class Config:
    seed: int = 20260815
    n_particles: int = 48
    n_features: int = 40
    frequency_scale: float = 0.85
    ridge_lambda: float = 0.08
    final_time: float = 1.5
    n_steps: int = 150
    max_outer: int = 16
    stop_tolerance: float = 1.0e-11
    min_outer: int = 6


class FourierScore:
    """Vector-valued direct product of one scalar paired-sine/cosine RFF space."""

    def __init__(self, frequencies: Array):
        if frequencies.ndim != 2 or frequencies.shape[1] != 2:
            raise ValueError("frequencies must have shape (q, 2)")
        self.frequencies = np.asarray(frequencies, dtype=float)
        self.n_frequencies = frequencies.shape[0]
        self.n_features = 2 * self.n_frequencies
        self.scale = 1.0 / math.sqrt(self.n_frequencies)

    def features_and_derivatives(self, x: Array) -> tuple[Array, Array]:
        """Return phi(x) and d_a phi(x), shaped (n,m) and (n,m,2)."""
        phase = np.asarray(x) @ self.frequencies.T
        cos_phase = np.cos(phase)
        sin_phase = np.sin(phase)
        phi = self.scale * np.concatenate((cos_phase, sin_phase), axis=1)

        n = x.shape[0]
        dphi = np.empty((n, self.n_features, 2), dtype=float)
        for a in range(2):
            omega_a = self.frequencies[:, a][None, :]
            dphi[:, : self.n_frequencies, a] = (
                -self.scale * sin_phase * omega_a
            )
            dphi[:, self.n_frequencies :, a] = (
                self.scale * cos_phase * omega_a
            )
        return phi, dphi

    def score(self, x: Array, coefficients: Array) -> Array:
        """Evaluate s(x)=Theta phi(x); coefficients have shape (2,m)."""
        phase = np.asarray(x) @ self.frequencies.T
        phi = self.scale * np.concatenate((np.cos(phase), np.sin(phase)), axis=1)
        return phi @ coefficients.T

    def minimize_hyvarinen(
        self, particles: Array, ridge_lambda: float
    ) -> tuple[Array, dict[str, float]]:
        """Solve the empirical ridge-Hyvarinen problem exactly.

        For each output coordinate a, the normal equation is

            (N^{-1} Phi^T Phi + lambda I) theta_a
                = -N^{-1} sum_i partial_a phi(x_i).
        """
        phi, dphi = self.features_and_derivatives(particles)
        n = particles.shape[0]
        gram = (phi.T @ phi) / n + ridge_lambda * np.eye(self.n_features)
        div_linear = dphi.mean(axis=0)  # (m,2), one RHS per output coordinate
        coefficient_transpose = np.linalg.solve(gram, -div_linear)
        coefficients = coefficient_transpose.T

        score_values = phi @ coefficient_transpose
        divergence = np.einsum("ima,am->i", dphi, coefficients)
        objective = float(
            np.mean(np.sum(score_values * score_values, axis=1) + 2.0 * divergence)
            + ridge_lambda * np.sum(coefficients * coefficients)
        )
        normal_residual = gram @ coefficient_transpose + div_linear
        relative_residual = float(
            np.linalg.norm(normal_residual)
            / max(1.0, np.linalg.norm(div_linear))
        )
        diagnostics = {
            "objective": objective,
            "coefficient_norm": float(np.linalg.norm(coefficients)),
            "normal_equation_relative_residual": relative_residual,
            "normal_matrix_condition": float(np.linalg.cond(gram)),
        }
        return coefficients, diagnostics


def collision_force(particles: Array, score_values: Array) -> Array:
    """Landau-type force for A(z)=(|z|^2 I-z z^T)/(1+|z|^2)."""
    z = particles[:, None, :] - particles[None, :, :]
    ds = score_values[:, None, :] - score_values[None, :, :]
    radius_sq = np.sum(z * z, axis=2)
    radial_dot = np.sum(z * ds, axis=2)
    a_times_ds = (
        radius_sq[:, :, None] * ds - z * radial_dot[:, :, None]
    ) / (1.0 + radius_sq[:, :, None])
    return -np.mean(a_times_ds, axis=1)


def rhs(particles: Array, coefficients: Array, model: FourierScore) -> Array:
    return collision_force(particles, model.score(particles, coefficients))


def solve_frozen_system(
    initial_particles: Array,
    coefficient_path: Array,
    final_time: float,
    model: FourierScore,
) -> Array:
    """RK4 solve using linear interpolation of the frozen score coefficients."""
    n_steps = coefficient_path.shape[0] - 1
    dt = final_time / n_steps
    path = np.empty((n_steps + 1,) + initial_particles.shape, dtype=float)
    path[0] = initial_particles
    for n in range(n_steps):
        theta_left = coefficient_path[n]
        theta_right = coefficient_path[n + 1]
        theta_mid = 0.5 * (theta_left + theta_right)
        x = path[n]
        k1 = rhs(x, theta_left, model)
        k2 = rhs(x + 0.5 * dt * k1, theta_mid, model)
        k3 = rhs(x + 0.5 * dt * k2, theta_mid, model)
        k4 = rhs(x + dt * k3, theta_right, model)
        path[n + 1] = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return path


def initial_particles(config: Config) -> Array:
    """A deterministic non-equilibrium two-component cloud.

    Independent component streams make the N=24,48,96 clouds nested within
    each mixture component, which removes an avoidable source of variation in
    the particle-number sweep.
    """
    n_left = config.n_particles // 2
    n_right = config.n_particles - n_left
    left_noise = np.random.default_rng(config.seed + 101).standard_normal((n_left, 2))
    right_noise = np.random.default_rng(config.seed + 202).standard_normal((n_right, 2))
    left = np.array([-1.10, 0.45]) + left_noise @ np.array(
        [[0.42, 0.12], [-0.04, 0.24]]
    ).T
    right = np.array([0.90, -0.35]) + right_noise @ np.array(
        [[0.28, -0.08], [0.14, 0.46]]
    ).T
    return np.concatenate((left, right), axis=0)


def make_model(config: Config) -> FourierScore:
    if config.n_features % 2:
        raise ValueError("n_features must be even (paired cosine/sine features)")
    rng = np.random.default_rng(config.seed + 1)
    frequencies = config.frequency_scale * rng.standard_normal(
        (config.n_features // 2, 2)
    )
    return FourierScore(frequencies)


def score_waveform(
    path: Array, model: FourierScore, ridge_lambda: float
) -> tuple[Array, list[dict[str, float]]]:
    coefficients = np.empty((path.shape[0], 2, model.n_features), dtype=float)
    diagnostics: list[dict[str, float]] = []
    for n, particles in enumerate(path):
        coefficients[n], info = model.minimize_hyvarinen(particles, ridge_lambda)
        diagnostics.append(info)
    return coefficients, diagnostics


def penalized_feature_gram(
    particles: Array, model: FourierScore, ridge_lambda: float
) -> Array:
    phi, _ = model.features_and_derivatives(particles)
    return (phi.T @ phi) / particles.shape[0] + ridge_lambda * np.eye(
        model.n_features
    )


def hyvarinen_objective(
    particles: Array,
    model: FourierScore,
    coefficients: Array,
    ridge_lambda: float,
) -> float:
    phi, dphi = model.features_and_derivatives(particles)
    score_values = phi @ coefficients.T
    divergence = np.einsum("ima,am->i", dphi, coefficients)
    return float(
        np.mean(np.sum(score_values * score_values, axis=1) + 2.0 * divergence)
        + ridge_lambda * np.sum(coefficients * coefficients)
    )


def exact_quadratic_score_gap(
    particles: Array,
    model: FourierScore,
    coefficient_perturbation: Array,
    ridge_lambda: float,
) -> float:
    """Return sum_a delta_a^T(Phi^T Phi/N+lambda I)delta_a."""
    penalized_gram = penalized_feature_gram(particles, model, ridge_lambda)
    return float(
        np.einsum(
            "am,mn,an->", coefficient_perturbation, penalized_gram, coefficient_perturbation
        )
    )


def mean_particle_distance(path_a: Array, path_b: Array) -> Array:
    return np.mean(np.linalg.norm(path_a - path_b, axis=2), axis=1)


def running_max(values: Array) -> Array:
    return np.maximum.accumulate(values)


def cumulative_trapezoid(values: Array, dt: float) -> Array:
    out = np.zeros_like(values)
    out[1:] = np.cumsum(0.5 * dt * (values[:-1] + values[1:]))
    return out


def invariant_diagnostics(path: Array) -> dict[str, float | Array]:
    momentum = np.mean(path, axis=1)
    energy = np.mean(np.sum(path * path, axis=2), axis=1)
    momentum_drift = np.linalg.norm(momentum - momentum[0], axis=1)
    energy_drift = np.abs(energy - energy[0])
    return {
        "momentum": momentum,
        "energy": energy,
        "momentum_drift": momentum_drift,
        "energy_drift": energy_drift,
        "momentum_drift_sup": float(np.max(momentum_drift)),
        "energy_drift_sup": float(np.max(energy_drift)),
    }


def midpoint_residual(
    new_path: Array,
    frozen_coefficients: Array,
    final_time: float,
    model: FourierScore,
) -> tuple[float, float]:
    """A transparent O(dt^2) chord-at-midpoint residual diagnostic."""
    dt = final_time / (new_path.shape[0] - 1)
    derivatives = np.diff(new_path, axis=0) / dt
    residual_values = []
    for n in range(new_path.shape[0] - 1):
        x_mid = 0.5 * (new_path[n] + new_path[n + 1])
        theta_mid = 0.5 * (frozen_coefficients[n] + frozen_coefficients[n + 1])
        residual = derivatives[n] - rhs(x_mid, theta_mid, model)
        residual_values.append(np.mean(np.linalg.norm(residual, axis=1)))
    residual_array = np.asarray(residual_values)
    return float(np.max(residual_array)), float(dt * np.sum(residual_array))


def summarize_score(diagnostics: list[dict[str, float]]) -> dict[str, float]:
    def values(key: str) -> Array:
        return np.asarray([entry[key] for entry in diagnostics])

    return {
        "score_objective_min": float(np.min(values("objective"))),
        "score_objective_mean": float(np.mean(values("objective"))),
        "score_objective_max": float(np.max(values("objective"))),
        "score_coefficient_norm_max": float(np.max(values("coefficient_norm"))),
        "score_normal_residual_max": float(
            np.max(values("normal_equation_relative_residual"))
        ),
        "score_condition_max": float(np.max(values("normal_matrix_condition"))),
    }


def run_iteration(
    config: Config,
    particles0: Array,
    model: FourierScore,
    verbose: bool = True,
    coefficient_perturbation: Array | None = None,
) -> dict[str, object]:
    n_times = config.n_steps + 1
    current = np.repeat(particles0[None, :, :], n_times, axis=0)
    waveforms = [current]
    coefficient_paths: list[Array] = []
    score_summaries: list[dict[str, float]] = []
    increment_curves: list[Array] = []
    system_residuals: list[tuple[float, float]] = []

    for outer in range(config.max_outer):
        exact_coefficients, score_info = score_waveform(
            current, model, config.ridge_lambda
        )
        coefficients = exact_coefficients
        if coefficient_perturbation is not None:
            coefficients = exact_coefficients + coefficient_perturbation[None, :, :]
        new_path = solve_frozen_system(
            particles0, coefficients, config.final_time, model
        )
        increment_curve = mean_particle_distance(new_path, current)
        residual = midpoint_residual(
            new_path, coefficients, config.final_time, model
        )

        coefficient_paths.append(coefficients)
        score_summaries.append(summarize_score(score_info))
        increment_curves.append(increment_curve)
        system_residuals.append(residual)
        waveforms.append(new_path)
        increment_sup = float(np.max(increment_curve))
        if verbose:
            print(
                f"outer {outer:02d} -> {outer + 1:02d}: "
                f"sup increment={increment_sup:.6e}, "
                f"energy drift={invariant_diagnostics(new_path)['energy_drift_sup']:.3e}, "
                f"score normal residual={score_summaries[-1]['score_normal_residual_max']:.3e}",
                flush=True,
            )
        current = new_path
        if outer + 1 >= config.min_outer and increment_sup <= config.stop_tolerance:
            break

    # Apply the discrete waveform map once more at the reported reference path.
    final_exact_coefficients, final_score_info = score_waveform(
        current, model, config.ridge_lambda
    )
    final_coefficients = final_exact_coefficients
    if coefficient_perturbation is not None:
        final_coefficients = (
            final_exact_coefficients + coefficient_perturbation[None, :, :]
        )
    mapped_final = solve_frozen_system(
        particles0, final_coefficients, config.final_time, model
    )
    fixed_point_curve = mean_particle_distance(mapped_final, current)
    fixed_point_residual = midpoint_residual(
        current, final_coefficients, config.final_time, model
    )

    return {
        "waveforms": np.asarray(waveforms),
        "coefficient_paths": np.asarray(coefficient_paths),
        "score_summaries": score_summaries,
        "increment_curves": np.asarray(increment_curves),
        "system_residuals": np.asarray(system_residuals),
        "final_exact_coefficients": final_exact_coefficients,
        "final_coefficients": final_coefficients,
        "final_score_summary": summarize_score(final_score_info),
        "mapped_final": mapped_final,
        "fixed_point_curve": fixed_point_curve,
        "fixed_point_defect_sup": float(np.max(fixed_point_curve)),
        "fixed_point_midpoint_residual_sup": fixed_point_residual[0],
        "fixed_point_midpoint_residual_l1": fixed_point_residual[1],
    }


def write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def factorial_envelope(increments: Array, final_time: float) -> tuple[float, Array]:
    """Smallest empirical C fitting D_k <= D_0 (CT)^k/(k+1)! above noise."""
    d0 = float(increments[0])
    noise_floor = max(1.0e-14, 1.0e-12 * d0)
    candidates = []
    for k in range(1, len(increments)):
        if increments[k] > noise_floor and d0 > 0.0:
            candidate = (
                (increments[k] / d0) * math.factorial(k + 1)
            ) ** (1.0 / k) / final_time
            candidates.append(candidate)
    c_fit = max(candidates, default=0.0)
    envelope = np.asarray(
        [
            d0 * (c_fit * final_time) ** k / math.factorial(k + 1)
            for k in range(len(increments))
        ]
    )
    return c_fit, envelope


def save_results(
    config: Config,
    particles0: Array,
    model: FourierScore,
    result: dict[str, object],
    output: Path,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    waveforms = np.asarray(result["waveforms"])
    increment_curves = np.asarray(result["increment_curves"])
    increments = np.max(increment_curves, axis=1)
    system_residuals = np.asarray(result["system_residuals"])
    times = np.linspace(0.0, config.final_time, config.n_steps + 1)
    reference = waveforms[-1]
    tail_errors = np.asarray(
        [np.max(mean_particle_distance(path, reference)) for path in waveforms]
    )
    c_fit, envelope = factorial_envelope(increments, config.final_time)

    increment_rows = []
    dt = config.final_time / config.n_steps
    for k, curve in enumerate(increment_curves):
        previous_ratio = ""
        if k >= 1 and increments[k - 1] > 0.0:
            previous_ratio = (k + 1) * increments[k] / increments[k - 1]
        volterra_constant = ""
        if k >= 1:
            denominator = cumulative_trapezoid(
                running_max(increment_curves[k - 1]), dt
            )
            numerator = running_max(curve)
            mask = denominator > max(1.0e-14, 1.0e-12 * increments[0])
            if np.any(mask) and increments[k] > 1.0e-14:
                volterra_constant = float(np.max(numerator[mask] / denominator[mask]))
        increment_rows.append(
            {
                "increment_index_k": k,
                "D_k_sup": float(increments[k]),
                "D_k_terminal": float(curve[-1]),
                "D_k_time_l1": float(np.trapezoid(curve, times)),
                "tail_from_Xk_to_reference": float(tail_errors[k]),
                "factorial_envelope": float(envelope[k]),
                "scaled_successive_ratio_(k+1)Dk/Dkm1": previous_ratio,
                "empirical_volterra_constant": volterra_constant,
                "midpoint_system_residual_sup": float(system_residuals[k, 0]),
                "midpoint_system_residual_l1": float(system_residuals[k, 1]),
            }
        )
    write_csv(output / "outer_convergence.csv", increment_rows[0].keys(), increment_rows)

    invariant_rows = []
    outer_rows = []
    score_summaries = list(result["score_summaries"])
    score_summaries.append(dict(result["final_score_summary"]))
    for k, path in enumerate(waveforms):
        inv = invariant_diagnostics(path)
        score_info = score_summaries[k]
        outer_rows.append(
            {
                "waveform_index": k,
                "tail_to_reference_sup": float(tail_errors[k]),
                "momentum_drift_sup": inv["momentum_drift_sup"],
                "energy_drift_sup": inv["energy_drift_sup"],
                **score_info,
            }
        )
        for n, t in enumerate(times):
            invariant_rows.append(
                {
                    "waveform_index": k,
                    "time": float(t),
                    "momentum_x": float(inv["momentum"][n, 0]),
                    "momentum_y": float(inv["momentum"][n, 1]),
                    "momentum_drift": float(inv["momentum_drift"][n]),
                    "energy": float(inv["energy"][n]),
                    "energy_drift": float(inv["energy_drift"][n]),
                }
            )
    write_csv(output / "outer_summary.csv", outer_rows[0].keys(), outer_rows)
    write_csv(output / "invariants_by_time.csv", invariant_rows[0].keys(), invariant_rows)

    final_rows = []
    for i in range(config.n_particles):
        final_rows.append(
            {
                "particle": i,
                "initial_x": float(particles0[i, 0]),
                "initial_y": float(particles0[i, 1]),
                "final_x": float(reference[-1, i, 0]),
                "final_y": float(reference[-1, i, 1]),
            }
        )
    write_csv(output / "final_particles.csv", final_rows[0].keys(), final_rows)

    np.savez_compressed(
        output / "raw_waveforms.npz",
        times=times,
        waveforms=waveforms,
        increment_curves=increment_curves,
        coefficient_paths=np.asarray(result["coefficient_paths"]),
        final_coefficients=np.asarray(result["final_coefficients"]),
        mapped_final=np.asarray(result["mapped_final"]),
        frequencies=model.frequencies,
        initial_particles=particles0,
    )

    max_momentum_drift = max(float(row["momentum_drift_sup"]) for row in outer_rows)
    max_energy_drift = max(float(row["energy_drift_sup"]) for row in outer_rows)
    max_score_residual = max(
        float(row["score_normal_residual_max"]) for row in outer_rows
    )
    reduction = float(increments[-1] / increments[0])
    validation = {
        "all_values_finite": bool(
            np.all(np.isfinite(waveforms)) and np.all(np.isfinite(increments))
        ),
        "outer_increment_reduction": reduction,
        "fixed_point_defect_sup": float(result["fixed_point_defect_sup"]),
        "max_momentum_drift": max_momentum_drift,
        "max_energy_drift": max_energy_drift,
        "max_score_normal_equation_relative_residual": max_score_residual,
        "factorial_compatible_C_fit": c_fit,
        "checks": {
            "increment_reduced_by_1e6": bool(reduction < 1.0e-6),
            "fixed_point_defect_below_1e-9": bool(
                float(result["fixed_point_defect_sup"]) < 1.0e-9
            ),
            "momentum_drift_below_1e-11": bool(max_momentum_drift < 1.0e-11),
            "energy_drift_below_1e-7": bool(max_energy_drift < 1.0e-7),
            "score_normal_residual_below_1e-10": bool(max_score_residual < 1.0e-10),
        },
    }
    validation["passed"] = bool(
        validation["all_values_finite"] and all(validation["checks"].values())
    )
    metadata = {
        "config": asdict(config),
        "actual_outer_steps": len(increments),
        "collision_matrix": "A(z)=(|z|^2 I-z z^T)/(1+|z|^2)",
        "score_model": "paired sine/cosine random Fourier features",
        "score_solver": "exact Tikhonov empirical Hyvarinen normal equations",
        "system_solver": "classical RK4 with linearly interpolated score coefficients",
        "validation": validation,
    }
    (output / "run_summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    make_plots(config, times, waveforms, increments, tail_errors, envelope, outer_rows, output)
    return metadata


def positive_for_log(values: Array) -> Array:
    return np.maximum(np.asarray(values, dtype=float), 1.0e-18)


def make_plots(
    config: Config,
    times: Array,
    waveforms: Array,
    increments: Array,
    tail_errors: Array,
    envelope: Array,
    outer_rows: list[dict],
    output: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    k = np.arange(len(increments))
    axes[0].semilogy(k, positive_for_log(increments), "o-", label=r"$\sup_t D_k(t)$")
    axes[0].semilogy(k, positive_for_log(envelope), "--", label="fitted factorial envelope")
    axes[0].semilogy(
        np.arange(len(tail_errors)),
        positive_for_log(tail_errors),
        "s-",
        label="tail to last waveform",
    )
    axes[0].set_xlabel("outer waveform index")
    axes[0].set_ylabel("mean-particle path error")
    axes[0].set_title("Outer convergence")
    axes[0].legend(fontsize=8)

    if len(increments) > 1:
        scaled = np.asarray(
            [(j + 1) * increments[j] / increments[j - 1] for j in range(1, len(increments))]
        )
        axes[1].plot(np.arange(1, len(increments)), scaled, "o-")
    axes[1].set_xlabel(r"increment index $k$")
    axes[1].set_ylabel(r"$(k+1)D_k^*/D_{k-1}^*$")
    axes[1].set_title("Factorial-decay diagnostic")
    fig.tight_layout()
    fig.savefig(output / "outer_convergence.png", dpi=180)
    plt.close(fig)

    momentum_drift = positive_for_log(
        [row["momentum_drift_sup"] for row in outer_rows]
    )
    energy_drift = positive_for_log([row["energy_drift_sup"] for row in outer_rows])
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    axes[0].semilogy(np.arange(len(outer_rows)), momentum_drift, "o-", label="momentum")
    axes[0].semilogy(np.arange(len(outer_rows)), energy_drift, "s-", label="energy")
    axes[0].set_xlabel("waveform index")
    axes[0].set_ylabel("maximum absolute drift")
    axes[0].set_title("Discrete conservation")
    axes[0].legend()

    final_inv = invariant_diagnostics(waveforms[-1])
    axes[1].plot(times, final_inv["momentum"][:, 0], label="momentum x")
    axes[1].plot(times, final_inv["momentum"][:, 1], label="momentum y")
    axes[1].plot(times, final_inv["energy"], label="kinetic energy")
    axes[1].set_xlabel("time")
    axes[1].set_title("Final waveform invariants")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "invariants.png", dpi=180)
    plt.close(fig)

    initial = waveforms[0, 0]
    terminal = waveforms[-1, -1]
    all_points = np.concatenate((initial, terminal), axis=0)
    low = np.min(all_points, axis=0) - 0.25
    high = np.max(all_points, axis=0) + 0.25
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), sharex=True, sharey=True)
    axes[0].scatter(initial[:, 0], initial[:, 1], s=25, alpha=0.8)
    axes[0].set_title("Initial particles")
    axes[1].scatter(terminal[:, 0], terminal[:, 1], s=25, alpha=0.8)
    axes[1].set_title(f"Terminal particles, t={config.final_time:g}")
    for ax in axes:
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("v1")
    axes[0].set_ylabel("v2")
    fig.tight_layout()
    fig.savefig(output / "particle_clouds.png", dpi=180)
    plt.close(fig)

    objectives = [row["score_objective_mean"] for row in outer_rows]
    coefficient_norms = [row["score_coefficient_norm_max"] for row in outer_rows]
    normal_residuals = positive_for_log(
        [row["score_normal_residual_max"] for row in outer_rows]
    )
    conditions = [row["score_condition_max"] for row in outer_rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    axes[0].plot(objectives, "o-", label="mean minimum objective")
    axes[0].plot(coefficient_norms, "s-", label="max coefficient norm")
    axes[0].set_xlabel("score waveform index")
    axes[0].set_title("Regularized score step")
    axes[0].legend(fontsize=8)
    axes[1].semilogy(normal_residuals, "o-", label="normal-equation residual")
    axes[1].semilogy(conditions, "s-", label="normal-matrix condition")
    axes[1].set_xlabel("score waveform index")
    axes[1].set_title("Score-solve diagnostics")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "score_diagnostics.png", dpi=180)
    plt.close(fig)


def run_refinement(
    config: Config,
    particles0: Array,
    model: FourierScore,
    baseline: dict[str, object],
    output: Path,
) -> None:
    """Repeat on nested time grids and compare converged discrete waveforms."""
    if config.n_steps % 2:
        raise ValueError("--refinement requires an even baseline n_steps")
    step_counts = [config.n_steps // 2, config.n_steps, 2 * config.n_steps]
    results: dict[int, dict[str, object]] = {config.n_steps: baseline}
    for steps in (step_counts[0], step_counts[2]):
        print(f"\nrefinement run: n_steps={steps}", flush=True)
        results[steps] = run_iteration(
            replace(config, n_steps=steps), particles0, model, verbose=False
        )

    fine_steps = step_counts[-1]
    fine_path = np.asarray(results[fine_steps]["waveforms"])[-1]
    rows = []
    for steps in step_counts:
        path = np.asarray(results[steps]["waveforms"])[-1]
        stride = fine_steps // steps
        error = mean_particle_distance(path, fine_path[::stride])
        inv = invariant_diagnostics(path)
        rows.append(
            {
                "n_steps": steps,
                "dt": config.final_time / steps,
                "outer_steps": np.asarray(results[steps]["increment_curves"]).shape[0],
                "path_error_to_finest_sup": float(np.max(error)) if steps != fine_steps else 0.0,
                "fixed_point_defect_sup": float(results[steps]["fixed_point_defect_sup"]),
                "momentum_drift_sup": inv["momentum_drift_sup"],
                "energy_drift_sup": inv["energy_drift_sup"],
                "midpoint_residual_sup": float(
                    results[steps]["fixed_point_midpoint_residual_sup"]
                ),
            }
        )
    write_csv(output / "time_refinement.csv", rows[0].keys(), rows)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    dt = np.asarray([row["dt"] for row in rows])
    residual = np.asarray([row["midpoint_residual_sup"] for row in rows])
    energy = positive_for_log([row["energy_drift_sup"] for row in rows])
    axes[0].loglog(dt, positive_for_log(residual), "o-", label="midpoint residual")
    axes[0].loglog(dt, energy, "s-", label="energy drift")
    axes[0].invert_xaxis()
    axes[0].set_xlabel("dt")
    axes[0].set_ylabel("diagnostic")
    axes[0].set_title("Time-grid refinement")
    axes[0].legend(fontsize=8)
    errors = positive_for_log([row["path_error_to_finest_sup"] for row in rows[:-1]])
    axes[1].loglog(dt[:-1], errors, "o-")
    axes[1].invert_xaxis()
    axes[1].set_xlabel("dt")
    axes[1].set_ylabel("path error to finest")
    axes[1].set_title("Converged waveform refinement")
    fig.tight_layout()
    fig.savefig(output / "time_refinement.png", dpi=180)
    plt.close(fig)


def run_particle_number_sweep(
    config: Config,
    baseline: dict[str, object],
    output: Path,
) -> dict[str, object]:
    """Compare outer increments for nested N=24,48,96 particle clouds."""
    particle_counts = [24, 48, 96]
    if config.n_particles not in particle_counts:
        particle_counts.append(config.n_particles)
        particle_counts.sort()
    results: dict[int, dict[str, object]] = {}
    rows = []
    summary_rows = []
    for n_particles in particle_counts:
        sweep_config = replace(config, n_particles=n_particles)
        particles_n = initial_particles(sweep_config)
        model_n = make_model(sweep_config)  # identical features for every N
        if n_particles == config.n_particles:
            result_n = baseline
        else:
            print(f"\nparticle-number sweep: N={n_particles}", flush=True)
            result_n = run_iteration(
                sweep_config, particles_n, model_n, verbose=False
            )
        results[n_particles] = result_n
        increments = np.max(np.asarray(result_n["increment_curves"]), axis=1)
        c_fit, _ = factorial_envelope(increments, config.final_time)
        summary_rows.append(
            {
                "n_particles": n_particles,
                "outer_steps": len(increments),
                "initial_increment": float(increments[0]),
                "last_increment": float(increments[-1]),
                "fixed_point_defect_sup": float(result_n["fixed_point_defect_sup"]),
                "factorial_compatible_C_fit": c_fit,
            }
        )
        for k, increment in enumerate(increments):
            rows.append(
                {
                    "n_particles": n_particles,
                    "increment_index_k": k,
                    "D_k_sup": float(increment),
                    "D_k_over_D0": float(increment / increments[0]),
                    "scaled_successive_ratio_(k+1)Dk/Dkm1": (
                        ""
                        if k == 0
                        else float((k + 1) * increment / increments[k - 1])
                    ),
                }
            )
    write_csv(output / "particle_number_sweep.csv", rows[0].keys(), rows)
    write_csv(
        output / "particle_number_sweep_summary.csv",
        summary_rows[0].keys(),
        summary_rows,
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    for n_particles in particle_counts:
        increments = np.max(
            np.asarray(results[n_particles]["increment_curves"]), axis=1
        )
        k = np.arange(len(increments))
        axes[0].semilogy(k, positive_for_log(increments), "o-", label=f"N={n_particles}")
        axes[1].semilogy(
            k,
            positive_for_log(increments / increments[0]),
            "o-",
            label=f"N={n_particles}",
        )
    axes[0].set_xlabel("increment index k")
    axes[0].set_ylabel(r"$\sup_t D_k(t)$")
    axes[0].set_title("Raw outer increments")
    axes[1].set_xlabel("increment index k")
    axes[1].set_ylabel(r"$\sup_t D_k(t)/\sup_tD_0(t)$")
    axes[1].set_title("N-normalized outer convergence")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "particle_number_sweep.png", dpi=180)
    plt.close(fig)
    return {
        "particle_counts": particle_counts,
        "summary": summary_rows,
    }


def run_inexact_score_sweep(
    config: Config,
    particles0: Array,
    model: FourierScore,
    exact_baseline: dict[str, object],
    output: Path,
) -> dict[str, object]:
    """Measure the persistent-error floor from fixed score perturbations."""
    amplitudes = [0.0, 0.01, 0.02, 0.04, 0.08, 0.16]
    rng = np.random.default_rng(config.seed + 303)
    direction = rng.standard_normal((2, model.n_features))
    direction /= np.linalg.norm(direction)
    exact_reference = np.asarray(exact_baseline["waveforms"])[-1]
    times = np.linspace(0.0, config.final_time, config.n_steps + 1)

    rows = []
    time_rows = []
    for amplitude in amplitudes:
        perturbation = amplitude * direction
        if amplitude == 0.0:
            result = exact_baseline
        else:
            print(f"\ninexact-score sweep: coefficient amplitude={amplitude:g}", flush=True)
            result = run_iteration(
                config,
                particles0,
                model,
                verbose=False,
                coefficient_perturbation=perturbation,
            )
        biased_path = np.asarray(result["waveforms"])[-1]
        exact_coefficients = np.asarray(result["final_exact_coefficients"])
        gaps = np.empty(config.n_steps + 1)
        identity_errors = np.empty(config.n_steps + 1)
        for n, particles in enumerate(biased_path):
            gaps[n] = exact_quadratic_score_gap(
                particles, model, perturbation, config.ridge_lambda
            )
            objective_exact = hyvarinen_objective(
                particles, model, exact_coefficients[n], config.ridge_lambda
            )
            objective_perturbed = hyvarinen_objective(
                particles,
                model,
                exact_coefficients[n] + perturbation,
                config.ridge_lambda,
            )
            identity_errors[n] = abs(
                (objective_perturbed - objective_exact) - gaps[n]
            )
            time_rows.append(
                {
                    "coefficient_amplitude": amplitude,
                    "time": float(times[n]),
                    "exact_quadratic_objective_gap": float(gaps[n]),
                    "sqrt_gap": float(math.sqrt(max(gaps[n], 0.0))),
                    "objective_gap_identity_error": float(identity_errors[n]),
                }
            )
        integrated_sqrt_gap = float(
            np.trapezoid(np.sqrt(np.maximum(gaps, 0.0)), times)
        )
        integrated_gap = float(np.trapezoid(gaps, times))
        path_error = float(
            np.max(mean_particle_distance(biased_path, exact_reference))
        )
        increments = np.max(np.asarray(result["increment_curves"]), axis=1)
        rows.append(
            {
                "coefficient_amplitude": amplitude,
                "integrated_exact_gap": integrated_gap,
                "integrated_sqrt_gap": integrated_sqrt_gap,
                "biased_fixed_point_path_error_sup": path_error,
                "error_over_integrated_sqrt_gap": (
                    "" if integrated_sqrt_gap == 0.0 else path_error / integrated_sqrt_gap
                ),
                "last_outer_increment": float(increments[-1]),
                "biased_fixed_point_defect_sup": float(result["fixed_point_defect_sup"]),
                "max_objective_gap_identity_error": float(np.max(identity_errors)),
            }
        )
    write_csv(output / "inexact_score_floor.csv", rows[0].keys(), rows)
    write_csv(
        output / "inexact_score_gap_by_time.csv", time_rows[0].keys(), time_rows
    )

    x = np.asarray([float(row["integrated_sqrt_gap"]) for row in rows[1:]])
    y = np.asarray(
        [float(row["biased_fixed_point_path_error_sup"]) for row in rows[1:]]
    )
    slope_origin = float(np.dot(x, y) / np.dot(x, x))
    fitted = slope_origin * x
    ss_residual = float(np.sum((y - fitted) ** 2))
    ss_total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_residual / ss_total if ss_total > 0.0 else 1.0
    log_slope, log_intercept = np.polyfit(np.log(x), np.log(y), 1)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    axes[0].plot(x, y, "o", label="biased fixed points")
    x_line = np.linspace(0.0, 1.05 * np.max(x), 100)
    axes[0].plot(x_line, slope_origin * x_line, "--", label="fit through origin")
    axes[0].set_xlabel(r"$\int_0^T \sqrt{\varepsilon(t)}\,dt$")
    axes[0].set_ylabel("fixed-point path error")
    axes[0].set_title("Persistent score-error floor")
    axes[0].legend(fontsize=8)
    axes[1].loglog(x, y, "o-", label="experiment")
    reference_line = y[0] * (x / x[0])
    axes[1].loglog(x, reference_line, "--", label="slope one")
    axes[1].set_xlabel(r"integrated $\sqrt{\mathrm{gap}}$")
    axes[1].set_ylabel("fixed-point path error")
    axes[1].set_title(f"log-log slope = {log_slope:.3f}")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "inexact_score_floor.png", dpi=180)
    plt.close(fig)
    return {
        "amplitudes": amplitudes,
        "linear_fit_through_origin": slope_origin,
        "linear_fit_R_squared": r_squared,
        "log_log_slope": float(log_slope),
        "log_log_intercept": float(log_intercept),
        "max_objective_gap_identity_error": max(
            float(row["max_objective_gap_identity_error"]) for row in rows
        ),
    }


def run_theorem_sweeps(
    config: Config,
    particles0: Array,
    model: FourierScore,
    baseline: dict[str, object],
    output: Path,
) -> dict[str, object]:
    particle_summary = run_particle_number_sweep(config, baseline, output)
    inexact_summary = run_inexact_score_sweep(
        config, particles0, model, baseline, output
    )
    summary = {
        "particle_number_sweep": particle_summary,
        "inexact_score_sweep": inexact_summary,
    }
    (output / "theorem_sweeps_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--particles", type=int, default=Config.n_particles)
    p.add_argument("--features", type=int, default=Config.n_features)
    p.add_argument("--frequency-scale", type=float, default=Config.frequency_scale)
    p.add_argument("--ridge", type=float, default=Config.ridge_lambda)
    p.add_argument("--final-time", type=float, default=Config.final_time)
    p.add_argument("--steps", type=int, default=Config.n_steps)
    p.add_argument("--outer", type=int, default=Config.max_outer)
    p.add_argument("--stop-tolerance", type=float, default=Config.stop_tolerance)
    p.add_argument("--refinement", action="store_true")
    p.add_argument(
        "--theorem-sweeps",
        action="store_true",
        help="run N-independence and persistent inexact-score tests",
    )
    p.add_argument("--strict", action="store_true", help="return nonzero if validation fails")
    return p


def main() -> int:
    args = parser().parse_args()
    config = Config(
        seed=args.seed,
        n_particles=args.particles,
        n_features=args.features,
        frequency_scale=args.frequency_scale,
        ridge_lambda=args.ridge,
        final_time=args.final_time,
        n_steps=args.steps,
        max_outer=args.outer,
        stop_tolerance=args.stop_tolerance,
    )
    if config.n_particles < 2 or config.n_steps < 2 or config.max_outer < 1:
        raise ValueError("particles >=2, steps >=2, and outer >=1 are required")
    if config.ridge_lambda <= 0.0 or config.final_time <= 0.0:
        raise ValueError("ridge and final-time must be positive")

    output = args.output.expanduser().resolve()
    particles0 = initial_particles(config)
    model = make_model(config)
    started = time.perf_counter()
    print("Alternating RFF score--system experiment")
    print(json.dumps(asdict(config), indent=2))
    result = run_iteration(config, particles0, model)
    metadata = save_results(config, particles0, model, result, output)
    if args.refinement:
        run_refinement(config, particles0, model, result, output)
    if args.theorem_sweeps:
        run_theorem_sweeps(config, particles0, model, result, output)
    elapsed = time.perf_counter() - started
    print(f"\noutputs: {output}")
    print(f"elapsed: {elapsed:.2f} s")
    print(json.dumps(metadata["validation"], indent=2))
    if args.strict and not metadata["validation"]["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
