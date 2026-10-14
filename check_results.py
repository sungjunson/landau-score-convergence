#!/usr/bin/env python3
"""Validate generated outputs for the alternating waveform experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument(
        "--require-refinement", action="store_true", help="also require time_refinement.csv"
    )
    parser.add_argument(
        "--require-lambda-sweep",
        action="store_true",
        help="also require the independent static lambda-limit audit",
    )
    args = parser.parse_args()
    output = args.output.expanduser().resolve()

    baseline_path = output / "run_summary.json"
    theorem_path = output / "theorem_sweeps_summary.json"
    monolithic_path = output / "monolithic_reference_summary.json"
    if not baseline_path.exists() or not theorem_path.exists() or not monolithic_path.exists():
        print("FAIL: run the theorem sweeps and monolithic reference before checking")
        return 1

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    theorem = json.loads(theorem_path.read_text(encoding="utf-8"))
    monolithic = json.loads(monolithic_path.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    checks["baseline_validation"] = bool(baseline["validation"]["passed"])

    particle_summary = theorem["particle_number_sweep"]["summary"]
    particle_counts = [entry["n_particles"] for entry in particle_summary]
    constants = [entry["factorial_compatible_C_fit"] for entry in particle_summary]
    checks["particle_counts_24_48_96"] = all(
        count in particle_counts for count in (24, 48, 96)
    )
    checks["N_sweep_constants_within_25_percent"] = (
        max(constants) / min(constants) < 1.25
    )
    checks["N_sweep_defects_below_1e-9"] = all(
        entry["fixed_point_defect_sup"] < 1.0e-9 for entry in particle_summary
    )

    inexact = theorem["inexact_score_sweep"]
    checks["gap_identity_below_1e-10"] = (
        inexact["max_objective_gap_identity_error"] < 1.0e-10
    )
    checks["inexact_log_slope_near_one"] = 0.9 < inexact["log_log_slope"] < 1.1
    checks["inexact_linear_fit_R2_above_0.99"] = (
        inexact["linear_fit_R_squared"] > 0.99
    )
    checks["monolithic_reference_validation"] = bool(monolithic["passed"])
    checks["waveform_to_monolithic_below_2e-6"] = (
        monolithic["waveform_to_monolithic_path_error_sup"] < 2.0e-6
    )
    checks["monolithic_RK4_order_above_3p5"] = (
        monolithic["refinement_observed_order"] > 3.5
    )

    required = [
        "outer_convergence.csv",
        "outer_convergence.png",
        "invariants_by_time.csv",
        "invariants.png",
        "particle_number_sweep.csv",
        "particle_number_sweep.png",
        "inexact_score_floor.csv",
        "inexact_score_floor.png",
        "raw_waveforms.npz",
        "monolithic_reference_by_time.csv",
        "monolithic_refinement.csv",
        "monolithic_reference.png",
    ]
    checks["required_outputs_exist"] = all((output / name).exists() for name in required)
    if args.require_refinement:
        checks["refinement_outputs_exist"] = all(
            (output / name).exists()
            for name in ("time_refinement.csv", "time_refinement.png")
        )
        if checks["refinement_outputs_exist"]:
            with (output / "time_refinement.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                refinement = list(csv.DictReader(handle))
            refinement.sort(key=lambda row: float(row["dt"]), reverse=True)
            midpoint = [float(row["midpoint_residual_sup"]) for row in refinement]
            energy = [float(row["energy_drift_sup"]) for row in refinement]
            checks["refinement_midpoint_residual_decreases"] = all(
                right < left for left, right in zip(midpoint, midpoint[1:])
            )
            checks["refinement_energy_drift_decreases"] = all(
                right < left for left, right in zip(energy, energy[1:])
            )
    if args.require_lambda_sweep:
        lambda_summary_path = output / "lambda_limit_summary.json"
        lambda_required = (
            "lambda_limit_sweep.csv",
            "lambda_limit_sweep.png",
            "lambda_limit_summary.json",
        )
        checks["lambda_sweep_outputs_exist"] = all(
            (output / name).exists() for name in lambda_required
        )
        if lambda_summary_path.exists():
            lambda_summary = json.loads(
                lambda_summary_path.read_text(encoding="utf-8")
            )
            checks["lambda_sweep_validation"] = bool(
                lambda_summary["validation"]["passed"]
            )

    passed = all(checks.values())
    for name, value in checks.items():
        print(f"{'PASS' if value else 'FAIL'}  {name}")
    print(f"\n{'PASS' if passed else 'FAIL'}: generated experiment audit")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
