#!/usr/bin/env python3
"""Deterministic audit of the empirical ``lambda -> 0`` dichotomy.

This script deliberately studies a *static* score-minimization problem.  It
does not claim a particle-to-continuum or Coulomb limit.  Two exactly
computable examples isolate what the regularization parameter can and cannot
do.

1. In a four-dimensional trigonometric feature space whose empirical Gram
   matrix is uniformly positive definite, the ridge minimizer converges to the
   unregularized minimizer at first order in ``lambda``.
2. In the infinite-dimensional Gaussian RKHS, the empirical divergence
   representer has a nonzero component perpendicular to all kernel sections at
   the training points.  Training-point score values still converge, whereas
   the full RKHS norm, a fixed off-grid value, and the magnitude of the minimum
   objective grow like ``1/lambda``.

The Gaussian calculation uses the exact RKHS inner products of kernel and
derivative sections; it is not a finite-feature approximation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


Array = np.ndarray
ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs"


def log_log_slope(x: Array, y: Array, fit_count: int) -> float:
    """Fit the slope of the final ``fit_count`` positive log--log data."""
    if fit_count < 2 or fit_count > x.size:
        raise ValueError("fit_count must lie between 2 and the sample count")
    x_fit = np.asarray(x[-fit_count:], dtype=float)
    y_fit = np.asarray(y[-fit_count:], dtype=float)
    if np.any(x_fit <= 0.0) or np.any(y_fit <= 0.0):
        raise ValueError("log--log fit data must be positive")
    return float(np.polyfit(np.log(x_fit), np.log(y_fit), 1)[0])


def finite_feature_case(lambdas: Array) -> tuple[list[dict[str, float]], dict]:
    """Solve a coercive empirical score problem in four bounded features."""
    n_particles = 24
    base = 2.0 * np.pi * np.arange(n_particles, dtype=float) / n_particles
    # A deterministic nonuniform cloud on the circle.  The small third
    # harmonic breaks an otherwise unnecessary reflection symmetry.
    particles = base + 0.4 * np.sin(base) + 0.08 * np.sin(3.0 * base + 0.4)
    sqrt_two = np.sqrt(2.0)
    phi = sqrt_two * np.column_stack(
        (
            np.cos(particles),
            np.sin(particles),
            np.cos(2.0 * particles),
            np.sin(2.0 * particles),
        )
    )
    dphi = sqrt_two * np.column_stack(
        (
            -np.sin(particles),
            np.cos(particles),
            -2.0 * np.sin(2.0 * particles),
            2.0 * np.cos(2.0 * particles),
        )
    )

    gram = (phi.T @ phi) / n_particles
    divergence_vector = dphi.mean(axis=0)
    theta_zero = np.linalg.solve(gram, -divergence_vector)
    score_zero = phi @ theta_zero
    eigenvalues = np.linalg.eigvalsh(gram)

    rows: list[dict[str, float]] = []
    max_relative_residual = 0.0
    for ridge_lambda in lambdas:
        normal_matrix = gram + ridge_lambda * np.eye(gram.shape[0])
        theta = np.linalg.solve(normal_matrix, -divergence_vector)
        residual = normal_matrix @ theta + divergence_vector
        relative_residual = float(
            np.linalg.norm(residual) / max(1.0, np.linalg.norm(divergence_vector))
        )
        max_relative_residual = max(max_relative_residual, relative_residual)
        rows.append(
            {
                "coefficient_error": float(np.linalg.norm(theta - theta_zero)),
                "training_value_rms_error": float(
                    np.linalg.norm(phi @ theta - score_zero) / np.sqrt(n_particles)
                ),
                "coefficient_norm": float(np.linalg.norm(theta)),
                "normal_matrix_condition": float(np.linalg.cond(normal_matrix)),
            }
        )

    info = {
        "n_particles": n_particles,
        "n_features": int(phi.shape[1]),
        "gram_eigenvalues": [float(value) for value in eigenvalues],
        "gram_min_eigenvalue": float(eigenvalues[0]),
        "gram_condition": float(np.linalg.cond(gram)),
        "unregularized_coefficient_norm": float(np.linalg.norm(theta_zero)),
        "max_normal_equation_relative_residual": max_relative_residual,
    }
    return rows, info


def gaussian_kernel(x: Array, y: Array, sigma: float) -> Array:
    difference = np.asarray(x, dtype=float)[..., None] - np.asarray(y, dtype=float)
    return np.exp(-(difference * difference) / (2.0 * sigma * sigma))


def gaussian_rkhs_case(lambdas: Array) -> tuple[list[dict[str, float]], dict]:
    """Use an exact evaluation/nullspace decomposition in a Gaussian RKHS.

    With ``k_i = k(z_i, .)`` and ``d_i = partial_1 k(z_i, .)``, let

        b = N^{-1} sum_i d_i,   V = span{k_i},   b_perp = b - P_V b.

    The empirical Hessian vanishes on ``V^perp``.  Consequently the exact
    ridge minimizer has the component ``-b_perp/lambda``.
    """
    particles = np.array([-1.4, -0.6, 0.1, 0.9, 1.5], dtype=float)
    sigma = 0.55
    n_particles = particles.size
    difference = particles[:, None] - particles[None, :]
    kernel_gram = np.exp(
        -(difference * difference) / (2.0 * sigma * sigma)
    )

    # Row j, column i is d_i(z_j) = partial_{z_i} k(z_i,z_j).
    derivative_values = (
        difference / (sigma * sigma) * kernel_gram
    )
    b_values = derivative_values.mean(axis=1)

    # <d_i,d_j> = partial_{z_i} partial_{z_j} k(z_i,z_j).
    derivative_gram = (
        1.0 / (sigma * sigma)
        - difference * difference / (sigma**4)
    ) * kernel_gram
    b_norm_squared = float(derivative_gram.mean())

    projection_coefficients = np.linalg.solve(kernel_gram, b_values)
    projection_norm_squared = float(b_values @ projection_coefficients)
    hidden_norm_squared = b_norm_squared - projection_norm_squared
    if hidden_norm_squared <= 0.0:
        raise RuntimeError("Gaussian hidden component was not resolved as positive")

    projected_training_values = kernel_gram @ projection_coefficients
    null_evaluation_residual = float(
        np.max(np.abs(b_values - projected_training_values))
    )

    # Pick one probe before sweeping lambda.  It maximizes |b_perp(x)| on a
    # fixed deterministic grid and is not a training point.
    probe_grid = np.linspace(-2.5, 2.5, 4001)
    probe_kernel = gaussian_kernel(probe_grid, particles, sigma)
    b_on_grid = (
        (probe_grid[:, None] - particles[None, :])
        / (sigma * sigma)
        * probe_kernel
    ).mean(axis=1)
    hidden_on_grid = b_on_grid - probe_kernel @ projection_coefficients
    probe_index = int(np.argmax(np.abs(hidden_on_grid)))
    probe = float(probe_grid[probe_index])
    probe_kernel_vector = probe_kernel[probe_index]
    hidden_probe_value = float(hidden_on_grid[probe_index])

    # This is the lambda=0 solution only on the visible subspace V.  It is also
    # the limit of the score values at the training points.
    limiting_training_values = -n_particles * np.linalg.solve(
        kernel_gram, b_values
    )

    rows: list[dict[str, float]] = []
    max_relative_residual = 0.0
    max_objective_identity_error = 0.0
    hidden_norm = np.sqrt(hidden_norm_squared)
    for ridge_lambda in lambdas:
        training_values = np.linalg.solve(
            kernel_gram + n_particles * ridge_lambda * np.eye(n_particles),
            -n_particles * b_values,
        )
        visible_coefficients = np.linalg.solve(kernel_gram, training_values)
        visible_norm_squared = float(training_values @ visible_coefficients)
        hidden_score_norm = hidden_norm / ridge_lambda
        full_score_norm = np.sqrt(
            visible_norm_squared + hidden_norm_squared / ridge_lambda**2
        )
        probe_score = float(
            probe_kernel_vector @ visible_coefficients
            - hidden_probe_value / ridge_lambda
        )

        coefficient_residual = (
            (kernel_gram / n_particles + ridge_lambda * np.eye(n_particles))
            @ visible_coefficients
            + projection_coefficients
        )
        relative_residual = float(
            np.linalg.norm(coefficient_residual)
            / max(1.0, np.linalg.norm(projection_coefficients))
        )
        max_relative_residual = max(max_relative_residual, relative_residual)

        # Direct quadratic value and the stationarity identity J(s_lambda)
        # = <b,s_lambda> must agree.
        b_inner_score = float(
            b_values @ visible_coefficients
            - hidden_norm_squared / ridge_lambda
        )
        direct_objective = float(
            np.mean(training_values * training_values)
            + 2.0 * b_inner_score
            + ridge_lambda * full_score_norm**2
        )
        max_objective_identity_error = max(
            max_objective_identity_error,
            abs(direct_objective - b_inner_score),
        )

        rows.append(
            {
                "on_grid_rms_error": float(
                    np.linalg.norm(training_values - limiting_training_values)
                    / np.sqrt(n_particles)
                ),
                "visible_h_norm": float(np.sqrt(visible_norm_squared)),
                "hidden_h_norm": float(hidden_score_norm),
                "full_h_norm": float(full_score_norm),
                "probe_abs_score": abs(probe_score),
                "minimum_objective": direct_objective,
                "normal_matrix_condition": float(
                    np.linalg.cond(
                        kernel_gram
                        + n_particles * ridge_lambda * np.eye(n_particles)
                    )
                ),
            }
        )

    info = {
        "n_particles": int(n_particles),
        "sigma": sigma,
        "training_points": [float(value) for value in particles],
        "kernel_gram_condition": float(np.linalg.cond(kernel_gram)),
        "divergence_representer_h_norm": float(np.sqrt(b_norm_squared)),
        "visible_component_h_norm": float(np.sqrt(projection_norm_squared)),
        "hidden_component_h_norm": float(hidden_norm),
        "hidden_energy_fraction": float(hidden_norm_squared / b_norm_squared),
        "hidden_component_training_evaluation_residual": null_evaluation_residual,
        "off_grid_probe": probe,
        "hidden_component_at_probe": hidden_probe_value,
        "max_normal_equation_relative_residual": max_relative_residual,
        "max_objective_stationarity_identity_error": max_objective_identity_error,
    }
    return rows, info


def write_csv(
    path: Path,
    lambdas: Array,
    finite_rows: list[dict[str, float]],
    gaussian_rows: list[dict[str, float]],
) -> None:
    fieldnames = ["lambda"]
    fieldnames.extend(f"finite_{key}" for key in finite_rows[0])
    fieldnames.extend(f"gaussian_{key}" for key in gaussian_rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for ridge_lambda, finite, gaussian in zip(
            lambdas, finite_rows, gaussian_rows
        ):
            row: dict[str, float] = {"lambda": float(ridge_lambda)}
            row.update({f"finite_{key}": value for key, value in finite.items()})
            row.update(
                {f"gaussian_{key}": value for key, value in gaussian.items()}
            )
            writer.writerow(row)


def make_plot(
    path: Path,
    lambdas: Array,
    finite_rows: list[dict[str, float]],
    gaussian_rows: list[dict[str, float]],
    finite_info: dict,
    gaussian_info: dict,
) -> None:
    finite_coefficient_error = np.array(
        [row["coefficient_error"] for row in finite_rows]
    )
    finite_training_error = np.array(
        [row["training_value_rms_error"] for row in finite_rows]
    )
    gaussian_on_grid_error = np.array(
        [row["on_grid_rms_error"] for row in gaussian_rows]
    )
    gaussian_hidden_norm = np.array(
        [row["hidden_h_norm"] for row in gaussian_rows]
    )
    gaussian_full_norm = np.array([row["full_h_norm"] for row in gaussian_rows])
    gaussian_probe = np.array([row["probe_abs_score"] for row in gaussian_rows])
    gaussian_objective = -np.array(
        [row["minimum_objective"] for row in gaussian_rows]
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), constrained_layout=True)
    ax = axes[0, 0]
    ax.loglog(lambdas, finite_coefficient_error, "o-", label=r"$\|\theta_\lambda-\theta_0\|$")
    ax.loglog(lambdas, finite_training_error, "s-", label="training-value RMS")
    reference = finite_coefficient_error[-1] * lambdas / lambdas[-1]
    ax.loglog(lambdas, reference, "k--", linewidth=1.1, label=r"reference $\lambda$")
    ax.set_title("Coercive finite-feature problem")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("error")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    ax.text(
        0.04,
        0.06,
        rf"$\lambda_{{\min}}(G)={finite_info['gram_min_eigenvalue']:.3f}$",
        transform=ax.transAxes,
        fontsize=9,
    )

    ax = axes[0, 1]
    ax.loglog(
        lambdas,
        gaussian_full_norm,
        "o-",
        label=r"full $\|S_\lambda\|_{\mathcal{H}}$",
    )
    ax.loglog(lambdas, gaussian_hidden_norm, "s--", label="hidden component")
    ax.loglog(
        lambdas,
        gaussian_hidden_norm[-1] * lambdas[-1] / lambdas,
        "k:",
        linewidth=1.3,
        label=r"reference $\lambda^{-1}$",
    )
    ax.set_title("Gaussian RKHS norm blow-up")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("RKHS norm")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.loglog(lambdas, gaussian_on_grid_error, "o-", color="tab:green", label="training-point RMS error")
    ax.loglog(
        lambdas,
        gaussian_on_grid_error[-1] * lambdas / lambdas[-1],
        "k--",
        linewidth=1.1,
        label=r"reference $\lambda$",
    )
    ax.set_title("Gaussian RKHS: observed values still converge")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"error to visible $\lambda=0$ limit")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.loglog(lambdas, gaussian_probe, "o-", color="tab:red", label="absolute off-grid score")
    ax.loglog(lambdas, gaussian_objective, "s-", color="tab:purple", label=r"$-\min J_\lambda$")
    ax.loglog(
        lambdas,
        gaussian_probe[-1] * lambdas[-1] / lambdas,
        "k:",
        linewidth=1.3,
        label=r"reference $\lambda^{-1}$",
    )
    ax.set_title("Hidden instability is visible off-grid")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("magnitude")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    ax.text(
        0.04,
        0.06,
        rf"probe $x={gaussian_info['off_grid_probe']:.3f}$",
        transform=ax.transAxes,
        fontsize=9,
    )

    fig.suptitle(
        r"Static empirical score matching: coercive limit versus hidden-nullspace blow-up",
        fontsize=13,
    )
    fig.savefig(path, dpi=190)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lambda-max", type=float, default=1.0e-1)
    parser.add_argument("--lambda-min", type=float, default=1.0e-8)
    parser.add_argument("--n-lambda", type=int, default=15)
    parser.add_argument(
        "--strict", action="store_true", help="return nonzero if validation fails"
    )
    args = parser.parse_args()
    if args.lambda_max <= args.lambda_min or args.lambda_min <= 0.0:
        raise ValueError("require lambda_max > lambda_min > 0")
    if args.n_lambda < 8:
        raise ValueError("n-lambda must be at least 8")

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    lambdas = np.geomspace(args.lambda_max, args.lambda_min, args.n_lambda)
    finite_rows, finite_info = finite_feature_case(lambdas)
    gaussian_rows, gaussian_info = gaussian_rkhs_case(lambdas)

    fit_count = max(5, args.n_lambda // 2)
    finite_errors = np.array([row["coefficient_error"] for row in finite_rows])
    on_grid_errors = np.array([row["on_grid_rms_error"] for row in gaussian_rows])
    hidden_norms = np.array([row["hidden_h_norm"] for row in gaussian_rows])
    full_norms = np.array([row["full_h_norm"] for row in gaussian_rows])
    probe_values = np.array([row["probe_abs_score"] for row in gaussian_rows])
    negative_objectives = -np.array(
        [row["minimum_objective"] for row in gaussian_rows]
    )
    slopes = {
        "finite_coefficient_error": log_log_slope(
            lambdas, finite_errors, fit_count
        ),
        "gaussian_on_grid_error": log_log_slope(
            lambdas, on_grid_errors, fit_count
        ),
        "gaussian_hidden_h_norm": log_log_slope(
            lambdas, hidden_norms, fit_count
        ),
        "gaussian_full_h_norm": log_log_slope(lambdas, full_norms, fit_count),
        "gaussian_probe_abs_score": log_log_slope(
            lambdas, probe_values, fit_count
        ),
        "gaussian_negative_minimum_objective": log_log_slope(
            lambdas, negative_objectives, fit_count
        ),
    }

    checks = {
        "finite_gram_uniformly_coercive": finite_info["gram_min_eigenvalue"] > 0.5,
        "finite_gram_condition_below_3": finite_info["gram_condition"] < 3.0,
        "finite_error_has_first_order_slope": 0.98
        < slopes["finite_coefficient_error"]
        < 1.02,
        "finite_normal_equations_solved": finite_info[
            "max_normal_equation_relative_residual"
        ]
        < 1.0e-12,
        "gaussian_kernel_gram_well_conditioned": gaussian_info[
            "kernel_gram_condition"
        ]
        < 10.0,
        "gaussian_hidden_component_nontrivial": gaussian_info[
            "hidden_component_h_norm"
        ]
        > 0.1,
        "gaussian_hidden_component_vanishes_on_training_set": gaussian_info[
            "hidden_component_training_evaluation_residual"
        ]
        < 1.0e-12,
        "gaussian_training_values_have_first_order_limit": 0.98
        < slopes["gaussian_on_grid_error"]
        < 1.02,
        "gaussian_full_norm_has_inverse_first_order_slope": -1.02
        < slopes["gaussian_full_h_norm"]
        < -0.98,
        "gaussian_off_grid_score_has_inverse_first_order_slope": -1.02
        < slopes["gaussian_probe_abs_score"]
        < -0.98,
        "gaussian_objective_has_inverse_first_order_slope": -1.02
        < slopes["gaussian_negative_minimum_objective"]
        < -0.98,
        "gaussian_normal_equations_solved": gaussian_info[
            "max_normal_equation_relative_residual"
        ]
        < 1.0e-11,
        "gaussian_objective_identity_verified": gaussian_info[
            "max_objective_stationarity_identity_error"
        ]
        < 1.0e-7,
    }
    passed = all(checks.values())

    csv_path = output / "lambda_limit_sweep.csv"
    figure_path = output / "lambda_limit_sweep.png"
    summary_path = output / "lambda_limit_summary.json"
    write_csv(csv_path, lambdas, finite_rows, gaussian_rows)
    make_plot(
        figure_path,
        lambdas,
        finite_rows,
        gaussian_rows,
        finite_info,
        gaussian_info,
    )

    summary = {
        "scope": (
            "Static deterministic empirical score minimization only; this is "
            "not a Coulomb, dynamic, mean-field, or simultaneous-limit test."
        ),
        "lambda_values": [float(value) for value in lambdas],
        "asymptotic_fit_count": fit_count,
        "finite_feature": finite_info,
        "gaussian_rkhs": gaussian_info,
        "log_log_slopes": slopes,
        "smallest_lambda_results": {
            "finite_coefficient_error": float(finite_errors[-1]),
            "gaussian_on_grid_rms_error": float(on_grid_errors[-1]),
            "gaussian_full_h_norm": float(full_norms[-1]),
            "gaussian_probe_abs_score": float(probe_values[-1]),
            "gaussian_minimum_objective": float(
                gaussian_rows[-1]["minimum_objective"]
            ),
        },
        "validation": {"passed": passed, "checks": checks},
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for name, value in checks.items():
        print(f"{'PASS' if value else 'FAIL'}  {name}")
    print("\nlog--log slopes on the smallest lambda values:")
    for name, value in slopes.items():
        print(f"  {name}: {value:+.6f}")
    print(f"\noutputs: {output}")
    print(f"{'PASS' if passed else 'FAIL'}: lambda-limit audit")
    return 0 if passed or not args.strict else 1


if __name__ == "__main__":
    raise SystemExit(main())
