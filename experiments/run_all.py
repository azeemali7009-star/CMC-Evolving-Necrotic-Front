#!/usr/bin/env python3
"""Reproduce every numerical result, table and figure in the submission package.

The script is deliberately non-destructive: it reads only from ``data/`` and
writes only to ``results/`` and ``figures/``.
"""
from __future__ import annotations

import json
import math
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cmc_remodelling.core import (
    D_POST,
    D_PRE,
    birth_commutation_residual,
    orientation_invariance_residual,
    radii_to_shell_capacities,
    shell_capacities_to_radii,
)

warnings.filterwarnings("default", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
SEED = 20260804
EPS_RADIUS = 1.0e-6


def safe_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def load_longitudinal_data() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    raw: dict[str, pd.DataFrame] = {}
    medians: dict[str, pd.Series] = {}
    for symbol, filename in [
        ("R", "WM793b_outer.csv"),
        ("I", "WM793b_inhibited.csv"),
        ("N", "WM793b_necrotic.csv"),
    ]:
        frame = pd.read_csv(DATA / filename)[["Day", "InitialCondition", "Radius"]].dropna()
        frame = frame.astype({"Day": float, "InitialCondition": int, "Radius": float})
        raw[symbol] = frame
        medians[symbol] = frame.groupby(["InitialCondition", "Day"]).Radius.median()

    wide = (
        pd.concat(medians, axis=1)
        .reset_index()
        .dropna()
        .sort_values(["InitialCondition", "Day"])
    )
    wide.to_csv(RESULTS / "longitudinal_medians.csv", index=False)

    transitions: list[list[float]] = []
    for initial_condition, group in wide.groupby("InitialCondition"):
        states = group[["Day", "R", "I", "N"]].to_numpy(float)
        for current, future in zip(states[:-1], states[1:]):
            transitions.append(
                [initial_condition, current[0], future[0] - current[0], *current[1:], *future[1:]]
            )
    pairs = pd.DataFrame(
        transitions,
        columns=["ic", "day", "dt", "R", "I", "N", "R2", "I2", "N2"],
    )
    pairs["fI"] = pairs.I / pairs.R
    pairs["fN"] = np.divide(pairs.N, pairs.I, out=np.zeros(len(pairs)), where=pairs.I > 0)
    pairs["fI2"] = pairs.I2 / pairs.R2
    pairs["fN2"] = np.divide(pairs.N2, pairs.I2, out=np.zeros(len(pairs)), where=pairs.I2 > 0)
    pairs.to_csv(RESULTS / "transition_table.csv", index=False)
    return raw, wide, pairs


def evaluate(true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    true = np.asarray(true, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    errors = predicted - true
    rmse = np.sqrt(np.mean(errors**2, axis=0))
    mae = np.mean(np.abs(errors), axis=0)
    relative = np.mean(
        np.linalg.norm(errors, axis=1) / (np.linalg.norm(true, axis=1) + 1e-12)
    )
    violation = np.mean(
        (predicted[:, 1] > predicted[:, 0])
        | (predicted[:, 2] > predicted[:, 1])
        | (predicted < 0).any(axis=1)
    )
    return {
        "rmse_outer": float(rmse[0]),
        "rmse_inhibited": float(rmse[1]),
        "rmse_necrotic": float(rmse[2]),
        "mae_outer": float(mae[0]),
        "mae_inhibited": float(mae[1]),
        "mae_necrotic": float(mae[2]),
        "mean_relative_state_error": float(relative),
        "constraint_violation_rate": float(violation),
    }



def event_forest_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    seed: int = SEED,
    mode: str = "cmc",
) -> np.ndarray:
    """Predict one-step increments with matched ExtraTrees regressors.

    The raw-radii mode is a controlled representation ablation: it uses the
    same number of regressors, hyperparameters and random seeds as the CMC
    model, but predicts increments in (R, R_I, R_N) and does not reconstruct
    the ordered shell state.
    """
    if mode == "no_initial_condition":
        features = ["day", "dt", "R", "fI", "fN"]
    elif mode == "raw_radii":
        features = ["day", "dt", "ic", "R", "I", "N"]
    elif mode == "cmc":
        features = ["day", "dt", "ic", "R", "fI", "fN"]
    else:
        raise ValueError(f"Unknown event-forest mode: {mode}")

    x_train = train[features].to_numpy()
    x_test = test[features].to_numpy()
    kwargs = dict(n_estimators=500, max_depth=5, min_samples_leaf=2, n_jobs=1)

    if mode == "raw_radii":
        targets = [train.R2 - train.R, train.I2 - train.I, train.N2 - train.N]
        current = test[["R", "I", "N"]].to_numpy()
        increments = []
        for offset, target in enumerate(targets, start=1):
            model = ExtraTreesRegressor(**kwargs, random_state=seed + offset)
            model.fit(x_train, target)
            increments.append(model.predict(x_test))
        return current + np.column_stack(increments)

    targets = [train.R2 - train.R, train.fI2 - train.fI, train.fN2 - train.fN]
    increments = []
    for offset, target in enumerate(targets, start=1):
        model = ExtraTreesRegressor(**kwargs, random_state=seed + offset)
        model.fit(x_train, target)
        increments.append(model.predict(x_test))

    outer = np.maximum(EPS_RADIUS, test.R.to_numpy() + increments[0])
    f_inhibited = np.clip(test.fI.to_numpy() + increments[1], 0.0, 1.0)
    f_necrotic = np.clip(test.fN.to_numpy() + increments[2], 0.0, 1.0)
    inhibited = outer * f_inhibited
    necrotic = inhibited * f_necrotic
    return np.column_stack([outer, inhibited, necrotic])


def time_conditioned_mlp_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Neural comparator operating on the same ordered event channels."""
    features = ["day", "dt", "ic", "R", "fI", "fN"]
    predictions: list[np.ndarray] = []
    targets = [train.R2 - train.R, train.fI2 - train.fI, train.fN2 - train.fN]
    for target, seed in zip(targets, [31, 32, 33]):
        model = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(16,),
                activation="tanh",
                alpha=1.0,
                max_iter=10000,
                random_state=seed,
            ),
        )
        model.fit(train[features], target)
        predictions.append(model.predict(test[features]))
    outer = np.maximum(EPS_RADIUS, test.R.to_numpy() + predictions[0])
    f_inhibited = np.clip(test.fI.to_numpy() + predictions[1], 0.0, 1.0)
    f_necrotic = np.clip(test.fN.to_numpy() + predictions[2], 0.0, 1.0)
    inhibited = outer * f_inhibited
    necrotic = inhibited * f_necrotic
    return np.column_stack([outer, inhibited, necrotic])


def fit_gompertz_greenspan(
    train_states: pd.DataFrame,
    test_transitions: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit a classical shared-asymptote Gompertz--Greenspan comparator.

    The outer radius follows a shared-asymptote Gompertz law with one initial
    displacement parameter per seeding condition. The inhibited and necrotic
    ratios follow the radial structural equations used in Greenspan-type
    diffusion-limited spheroid models.
    """
    initial_conditions = sorted(train_states.InitialCondition.unique())

    def unpack_growth(theta: np.ndarray):
        carrying_radius = float(np.exp(theta[0]))
        growth_rate = float(np.exp(theta[1]))
        offsets = {
            int(ic): float(np.exp(theta[2 + i]))
            for i, ic in enumerate(initial_conditions)
        }
        return carrying_radius, growth_rate, offsets

    def growth_residual(theta: np.ndarray) -> np.ndarray:
        carrying_radius, growth_rate, offsets = unpack_growth(theta)
        prediction = np.array(
            [
                carrying_radius
                * np.exp(-offsets[int(ic)] * np.exp(-growth_rate * day))
                for ic, day in zip(train_states.InitialCondition, train_states.Day)
            ]
        )
        return (prediction - train_states.R.to_numpy()) / 20.0

    initial_theta = np.array(
        [math.log(400.0), math.log(0.1)] + [math.log(2.0)] * len(initial_conditions)
    )
    fitted_growth = least_squares(growth_residual, initial_theta, max_nfev=20000)
    carrying_radius, growth_rate, offsets = unpack_growth(fitted_growth.x)

    observed_phi = np.divide(
        train_states.I,
        train_states.R,
        out=np.zeros(len(train_states)),
        where=train_states.R > 0,
    )
    observed_eta = np.divide(
        train_states.N,
        train_states.R,
        out=np.zeros(len(train_states)),
        where=train_states.R > 0,
    )

    def real_roots(coefficients: list[float]) -> list[float]:
        roots = np.roots(coefficients)
        return sorted(float(root.real) for root in roots if abs(root.imag) < 1e-7)

    def structural_fractions(
        radii: np.ndarray, q_parameter: float, critical_radius: float
    ) -> tuple[np.ndarray, np.ndarray]:
        phi_values: list[float] = []
        eta_values: list[float] = []
        for radius in np.asarray(radii, dtype=float):
            if radius <= critical_radius:
                eta = 0.0
                phi = math.sqrt(max(0.0, 1.0 - (q_parameter * critical_radius / radius) ** 2))
            else:
                ratio_squared = (critical_radius / radius) ** 2
                eta_candidates = [
                    value
                    for value in real_roots([2.0, -3.0, 0.0, 1.0 - ratio_squared])
                    if 0.0 <= value <= 1.0
                ]
                eta = eta_candidates[0] if eta_candidates else 0.0
                phi_candidates = [
                    value
                    for value in real_roots(
                        [
                            1.0,
                            0.0,
                            q_parameter**2 * ratio_squared - (1.0 + 2.0 * eta**3),
                            2.0 * eta**3,
                        ]
                    )
                    if eta - 1e-8 <= value <= 1.0 + 1e-8
                ]
                phi = max(phi_candidates) if phi_candidates else eta
            phi_values.append(float(np.clip(phi, eta, 1.0)))
            eta_values.append(float(np.clip(eta, 0.0, 1.0)))
        return np.asarray(phi_values), np.asarray(eta_values)

    def structure_residual(theta: np.ndarray) -> np.ndarray:
        q_parameter, critical_radius = np.exp(theta)
        predicted_phi, predicted_eta = structural_fractions(
            train_states.R.to_numpy(), q_parameter, critical_radius
        )
        return np.concatenate([predicted_phi - observed_phi, predicted_eta - observed_eta])

    best_structure = None
    for q0 in [0.3, 0.6, 0.9, 1.2]:
        for rc0 in [100.0, 180.0, 260.0, 340.0]:
            candidate = least_squares(
                structure_residual,
                [math.log(q0), math.log(rc0)],
                bounds=([math.log(0.05), math.log(20.0)], [math.log(5.0), math.log(600.0)]),
                max_nfev=10000,
            )
            if best_structure is None or candidate.cost < best_structure.cost:
                best_structure = candidate
    assert best_structure is not None
    q_parameter, critical_radius = np.exp(best_structure.x)

    target_day = test_transitions.day.to_numpy() + test_transitions.dt.to_numpy()
    predicted_outer = np.array(
        [
            carrying_radius
            * np.exp(-offsets[int(ic)] * np.exp(-growth_rate * day))
            for ic, day in zip(test_transitions.ic, target_day)
        ]
    )
    predicted_phi, predicted_eta = structural_fractions(
        predicted_outer, float(q_parameter), float(critical_radius)
    )
    prediction = np.column_stack(
        [predicted_outer, predicted_outer * predicted_phi, predicted_outer * predicted_eta]
    )
    parameters = {
        "carrying_radius": carrying_radius,
        "growth_rate": growth_rate,
        "q_parameter": float(q_parameter),
        "critical_radius": float(critical_radius),
        **{f"gompertz_offset_{ic}": offsets[int(ic)] for ic in initial_conditions},
    }
    return prediction, parameters



def build_forecasts(wide: pd.DataFrame, pairs: pd.DataFrame):
    train = pairs[pairs.day + pairs.dt <= 18].copy()
    test = pairs[pairs.day + pairs.dt > 18].copy()
    true = test[["R2", "I2", "N2"]].to_numpy()
    predictions: dict[str, np.ndarray] = {
        "Persistence": test[["R", "I", "N"]].to_numpy(),
    }

    ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    ridge.fit(train[["day", "dt", "ic", "R", "I", "N"]], train[["R2", "I2", "N2"]])
    predictions["Linear ridge"] = ridge.predict(test[["day", "dt", "ic", "R", "I", "N"]])

    direct_mlp = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation="tanh",
            alpha=1.0,
            max_iter=3000,
            random_state=7,
        ),
    )
    direct_mlp.fit(train[["day", "dt", "ic", "R", "I", "N"]], train[["R2", "I2", "N2"]])
    predictions["Direct MLP"] = direct_mlp.predict(test[["day", "dt", "ic", "R", "I", "N"]])

    predictions["CMC-EventForest"] = event_forest_predict(train, test)
    predictions["CMC-TimeConditionedMLP"] = time_conditioned_mlp_predict(train, test)
    predictions["Matched raw-radii forest"] = event_forest_predict(train, test, mode="raw_radii")
    predictions["CMC without initial condition"] = event_forest_predict(train, test, mode="no_initial_condition")

    classical_prediction, classical_parameters = fit_gompertz_greenspan(
        wide[wide.Day <= 18].copy(), test
    )
    predictions["Gompertz-Greenspan"] = classical_prediction
    with (RESULTS / "gompertz_greenspan_parameters.json").open("w") as handle:
        json.dump(classical_parameters, handle, indent=2)

    rows = []
    case_rows = []
    target_days = test.day.to_numpy() + test.dt.to_numpy()
    for name, prediction in predictions.items():
        rows.append({"model": name, **evaluate(true, prediction)})
        case_relative = np.linalg.norm(prediction - true, axis=1) / (
            np.linalg.norm(true, axis=1) + 1e-12
        )
        output = pd.DataFrame({
            "ic": test.ic.to_numpy(),
            "day": test.day.to_numpy(),
            "dt": test.dt.to_numpy(),
            "target_day": target_days,
            "true_R": true[:, 0],
            "true_I": true[:, 1],
            "true_N": true[:, 2],
            "pred_R": prediction[:, 0],
            "pred_I": prediction[:, 1],
            "pred_N": prediction[:, 2],
            "case_relative_error": case_relative,
        })
        output.to_csv(RESULTS / f"predictions_{safe_slug(name)}.csv", index=False)
        for record in output.to_dict(orient="records"):
            case_rows.append({"model": name, **record})

    metrics = pd.DataFrame(rows).sort_values("mean_relative_state_error").reset_index(drop=True)
    metrics.to_csv(RESULTS / "forecast_metrics.csv", index=False)
    case_errors = pd.DataFrame(case_rows)
    case_errors.to_csv(RESULTS / "forecast_case_errors.csv", index=False)

    cmc = case_errors[case_errors.model == "CMC-EventForest"].reset_index(drop=True)
    raw = case_errors[case_errors.model == "Matched raw-radii forest"].reset_index(drop=True)
    summary = {
        "number_of_test_cases": int(len(cmc)),
        "cmc_lower_error_cases": int((cmc.case_relative_error < raw.case_relative_error).sum()),
        "matched_raw_lower_error_cases": int((raw.case_relative_error < cmc.case_relative_error).sum()),
        "equal_error_cases": int(np.isclose(cmc.case_relative_error, raw.case_relative_error).sum()),
    }
    (RESULTS / "paired_case_summary.json").write_text(json.dumps(summary, indent=2))
    return train, test, true, predictions, metrics, case_errors


def build_loio(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for held_condition in sorted(pairs.ic.unique()):
        train = pairs[pairs.ic != held_condition]
        test = pairs[pairs.ic == held_condition]
        true = test[["R2", "I2", "N2"]].to_numpy()
        for name, prediction in [
            ("CMC-EventForest", event_forest_predict(train, test)),
            ("Matched raw-radii forest", event_forest_predict(train, test, mode="raw_radii")),
        ]:
            rows.append({
                "held_initial_condition": int(held_condition),
                "model": name,
                **evaluate(true, prediction),
            })
    result = pd.DataFrame(rows)
    result.to_csv(RESULTS / "loio_metrics.csv", index=False)
    return result


def build_seed_sensitivity(train: pd.DataFrame, test: pd.DataFrame, true: np.ndarray) -> pd.DataFrame:
    rows = []
    for seed in range(SEED, SEED + 4):
        for name, prediction in [
            ("CMC-EventForest", event_forest_predict(train, test, seed=seed)),
            ("Matched raw-radii forest", event_forest_predict(train, test, seed=seed, mode="raw_radii")),
        ]:
            rows.append({"seed": seed, "model": name, **evaluate(true, prediction)})
    result = pd.DataFrame(rows)
    result.to_csv(RESULTS / "seed_sensitivity.csv", index=False)
    summary = result.groupby("model").agg(
        mean_relative_error=("mean_relative_state_error", "mean"),
        sd_relative_error=("mean_relative_state_error", "std"),
        min_relative_error=("mean_relative_state_error", "min"),
        max_relative_error=("mean_relative_state_error", "max"),
    ).reset_index()
    summary.to_csv(RESULTS / "seed_sensitivity_summary.csv", index=False)
    return result


def analyse_necrosis_onset(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for initial_condition, group in wide.groupby("InitialCondition"):
        positive = group[group.N > 5.0]
        rows.append({
            "initial_condition": int(initial_condition),
            "observed_onset_day": float(positive.Day.min()) if len(positive) else np.nan,
        })
    onset = pd.DataFrame(rows)
    onset.to_csv(RESULTS / "necrosis_onset.csv", index=False)

    sensitivity = []
    for threshold in [1.0, 2.5, 5.0, 10.0, 20.0]:
        for initial_condition, group in wide.groupby("InitialCondition"):
            positive = group[group.N > threshold]
            sensitivity.append({
                "threshold_um": threshold,
                "initial_condition": int(initial_condition),
                "observed_onset_day": float(positive.Day.min()) if len(positive) else np.nan,
            })
    pd.DataFrame(sensitivity).to_csv(RESULTS / "necrosis_onset_sensitivity.csv", index=False)
    return onset


def analyse_doxorubicin() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(DATA / "diosdi_2025_group_means.csv")
    rows = []
    measures = [
        "diameter_um",
        "perimeter_um",
        "area_um2",
        "volume_um3",
        "circularity",
        "sphericity",
    ]
    for (tumour_type, time_h), group in data.groupby(["tumour_type", "time_h"]):
        control = group[group.doxorubicin == 0].iloc[0]
        treatment = group[group.doxorubicin == 1].iloc[0]
        row: dict[str, float | str] = {"tumour_type": tumour_type, "time_h": int(time_h)}
        for measure in measures:
            row[f"{measure}_log_ratio"] = float(np.log(treatment[measure] / control[measure]))
            row[f"{measure}_percent_change"] = float(
                100.0 * (treatment[measure] / control[measure] - 1.0)
            )
        rows.append(row)
    effects = pd.DataFrame(rows)
    effects.to_csv(RESULTS / "doxorubicin_effects.csv", index=False)
    return data, effects


def write_audits(wide: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> dict[str, float | int | bool]:
    observed_radii = wide[["R", "I", "N"]].to_numpy()
    round_trip = shell_capacities_to_radii(radii_to_shell_capacities(observed_radii))
    audits: dict[str, float | int | bool] = {
        "necrotic_birth_commutation_residual": birth_commutation_residual(),
        "orientation_invariance_residual": orientation_invariance_residual(),
        "capacity_roundtrip_residual": float(np.max(np.abs(round_trip - observed_radii))),
        "nested_constraints_all_observations": bool(
            ((wide.N <= wide.I) & (wide.I <= wide.R) & (wide.N >= 0)).all()
        ),
        "pre_necrotic_incidence_rows": int(D_PRE.shape[0]),
        "post_necrotic_incidence_rows": int(D_POST.shape[0]),
        "forecast_test_cases": int(len(test)),
        "training_transitions": int(len(train)),
        "random_seed": SEED,
    }
    with (RESULTS / "algebraic_audits.json").open("w") as handle:
        json.dump(audits, handle, indent=2)
    return audits



def write_tables_and_macros(
    metrics: pd.DataFrame,
    onset: pd.DataFrame,
    effects: pd.DataFrame,
    loio: pd.DataFrame,
    raw: dict[str, pd.DataFrame],
    case_errors: pd.DataFrame,
    seed_sensitivity: pd.DataFrame,
) -> None:
    best = metrics.iloc[0]
    lookup = metrics.set_index("model")
    loio_summary = loio.groupby("model").agg(
        mean_relative_error=("mean_relative_state_error", "mean"),
        sd_relative_error=("mean_relative_state_error", "std"),
        maximum_violation_rate=("constraint_violation_rate", "max"),
    ).reset_index()
    loio_summary.to_csv(RESULTS / "loio_summary.csv", index=False)
    seed_summary = seed_sensitivity.groupby("model").agg(
        mean_relative_error=("mean_relative_state_error", "mean"),
        sd_relative_error=("mean_relative_state_error", "std"),
    ).reset_index().set_index("model")

    macros = (
        f"\\newcommand{{\\BestModel}}{{{best.model}}}\n"
        f"\\newcommand{{\\BestRelError}}{{{best.mean_relative_state_error:.3f}}}\n"
        f"\\newcommand{{\\BestOuterRMSE}}{{{best.rmse_outer:.1f}}}\n"
        f"\\newcommand{{\\BestInhibitedRMSE}}{{{best.rmse_inhibited:.1f}}}\n"
        f"\\newcommand{{\\BestNecroticRMSE}}{{{best.rmse_necrotic:.1f}}}\n"
        f"\\newcommand{{\\TimeMLPRelError}}{{{lookup.loc['CMC-TimeConditionedMLP','mean_relative_state_error']:.3f}}}\n"
        f"\\newcommand{{\\MatchedRawRelError}}{{{lookup.loc['Matched raw-radii forest','mean_relative_state_error']:.3f}}}\n"
        f"\\newcommand{{\\DirectMLPRelError}}{{{lookup.loc['Direct MLP','mean_relative_state_error']:.3f}}}\n"
        f"\\newcommand{{\\GompertzRelError}}{{{lookup.loc['Gompertz-Greenspan','mean_relative_state_error']:.3f}}}\n"
        f"\\newcommand{{\\PersistenceRelError}}{{{lookup.loc['Persistence','mean_relative_state_error']:.3f}}}\n"
        f"\\newcommand{{\\SeedCMCMean}}{{{seed_summary.loc['CMC-EventForest','mean_relative_error']:.4f}}}\n"
        f"\\newcommand{{\\SeedCMCSD}}{{{seed_summary.loc['CMC-EventForest','sd_relative_error']:.4f}}}\n"
        f"\\newcommand{{\\SeedRawMean}}{{{seed_summary.loc['Matched raw-radii forest','mean_relative_error']:.4f}}}\n"
        f"\\newcommand{{\\SeedRawSD}}{{{seed_summary.loc['Matched raw-radii forest','sd_relative_error']:.4f}}}\n"
        f"\\newcommand{{\\NumOuterObs}}{{{len(raw['R'])}}}\n"
        f"\\newcommand{{\\NumFrontObs}}{{{len(raw['I'])}}}\n"
        "\\newcommand{\\NumDoxSpheroids}{90}\n"
    )
    (RESULTS / "result_macros.tex").write_text(macros)

    lines = [r"\begin{tabular}{lrrrrr}\toprule Model & Rel. error & RMSE $R$ & RMSE $R_I$ & RMSE $R_N$ & Violation \\ \midrule"]
    for _, row in metrics.iterrows():
        lines.append(f"{row.model} & {row.mean_relative_state_error:.3f} & {row.rmse_outer:.1f} & {row.rmse_inhibited:.1f} & {row.rmse_necrotic:.1f} & {row.constraint_violation_rate:.3f} \\\\")
    lines.append(r"\bottomrule\end{tabular}")
    (RESULTS / "forecast_table.tex").write_text("\n".join(lines))

    lines = [r"\begin{tabular}{rr}\toprule Initial cells & First sampled day with $R_N>5\,\mu\mathrm{m}$ \\ \midrule"]
    for _, row in onset.iterrows():
        lines.append(f"{int(row.initial_condition)} & {row.observed_onset_day:.0f} \\\\")
    lines.append(r"\bottomrule\end{tabular}")
    (RESULTS / "onset_table.tex").write_text("\n".join(lines))

    lines = [r"\begin{tabular}{lrrr}\toprule Tumour & Time (h) & Volume change (\%) & Sphericity change (\%) \\ \midrule"]
    for _, row in effects.iterrows():
        lines.append(f"{row.tumour_type} & {int(row.time_h)} & {row.volume_um3_percent_change:.1f} & {row.sphericity_percent_change:.1f} \\\\")
    lines.append(r"\bottomrule\end{tabular}")
    (RESULTS / "dox_table.tex").write_text("\n".join(lines))

    lines = [r"\begin{tabular}{lrrr}\toprule Model & Mean rel. error & SD & Max. violation \\ \midrule"]
    for _, row in loio_summary.iterrows():
        lines.append(f"{row.model} & {row.mean_relative_error:.3f} & {row.sd_relative_error:.3f} & {row.maximum_violation_rate:.3f} \\\\")
    lines.append(r"\bottomrule\end{tabular}")
    (RESULTS / "loio_table.tex").write_text("\n".join(lines))

    cmc_cases = case_errors[case_errors.model == "CMC-EventForest"]
    lines = [r"\begin{tabular}{rrrrrrrr}\toprule Initial cells & Target day & Obs. $R$ & Pred. $R$ & Obs. $R_I$ & Pred. $R_I$ & Obs. $R_N$ & Pred. $R_N$ \\ \midrule"]
    for _, row in cmc_cases.iterrows():
        lines.append(f"{int(row.ic)} & {int(row.target_day)} & {row.true_R:.1f} & {row.pred_R:.1f} & {row.true_I:.1f} & {row.pred_I:.1f} & {row.true_N:.1f} & {row.pred_N:.1f} \\\\")
    lines.append(r"\bottomrule\end{tabular}")
    (RESULTS / "forecast_case_table.tex").write_text("\n".join(lines))

    lines = [r"\begin{tabular}{lrr}\toprule Model & Mean rel. error & Seed SD \\ \midrule"]
    for model, row in seed_summary.iterrows():
        lines.append(f"{model} & {row.mean_relative_error:.3f} & {row.sd_relative_error:.3f} \\\\")
    lines.append(r"\bottomrule\end{tabular}")
    (RESULTS / "seed_sensitivity_table.tex").write_text("\n".join(lines))


def generate_figures(
    raw: dict[str, pd.DataFrame],
    wide: pd.DataFrame,
    dox_data: pd.DataFrame,
    effects: pd.DataFrame,
    onset: pd.DataFrame,
    true: np.ndarray,
    predictions: dict[str, np.ndarray],
    metrics: pd.DataFrame,
    audits: dict[str, float | int | bool],
) -> None:
    plt.rcParams.update({"font.size": 9, "figure.dpi": 140, "savefig.bbox": "tight"})

    def save(name: str) -> None:
        plt.savefig(FIGURES / f"{name}.pdf")
        plt.savefig(FIGURES / f"{name}.png", dpi=220)
        plt.close()

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    labels = [
        "Public longitudinal\nradius observations",
        "Evolving active\ncell complex",
        "Topology-constrained\nevent learning",
        "Temporally held-out\none-step forecast",
        "Independent doxorubicin\nremodelling analysis",
    ]
    positions = np.linspace(0.08, 0.92, len(labels))
    for x, label in zip(positions, labels):
        ax.add_patch(plt.Rectangle((x - 0.08, 0.35), 0.16, 0.3, fill=False, lw=1.5))
        ax.text(x, 0.5, label, ha="center", va="center")
    for left, right in zip(positions[:-1], positions[1:]):
        ax.annotate(
            "",
            xy=(right - 0.085, 0.5),
            xytext=(left + 0.085, 0.5),
            arrowprops=dict(arrowstyle="->", lw=1.4),
        )
    ax.text(
        0.5,
        0.12,
        "Measured radii define capacities; incidence and compatible event maps define the intrinsic topology.",
        ha="center",
    )
    save("fig01_workflow")

    fig, ax = plt.subplots(figsize=(7, 4))
    names = ["Outer radius", "Inhibited radius", "Necrotic radius", "Dox spheroids"]
    values = [len(raw["R"]), len(raw["I"]), len(raw["N"]), 90]
    ax.bar(names, values)
    ax.set_yscale("log")
    ax.set_ylabel("Experimental observations")
    ax.set_title("Public experimental evidence used in this study")
    save("fig02_data_landscape")

    for index, measure, label in [
        (3, "volume_um3", r"Volume ($\mu$m$^3$)"),
        (4, "diameter_um", r"Diameter ($\mu$m)"),
        (5, "circularity", "Circularity"),
        (6, "sphericity", "Sphericity"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for tumour_type in dox_data.tumour_type.unique():
            for treatment, linestyle in [(0, "-"), (1, "--")]:
                group = dox_data[
                    (dox_data.tumour_type == tumour_type)
                    & (dox_data.doxorubicin == treatment)
                ].sort_values("time_h")
                ax.plot(
                    group.time_h,
                    group[measure],
                    marker="o",
                    linestyle=linestyle,
                    label=f"{tumour_type} {'dox' if treatment else 'control'}",
                )
        ax.set_xlabel("Time (h)")
        ax.set_ylabel(label)
        ax.legend(ncol=2, fontsize=7)
        ax.set_title("Experimental multiculture spheroid morphology")
        save(f"fig{index:02d}_{measure}")

    heatmap_columns = [
        "diameter_um_percent_change",
        "perimeter_um_percent_change",
        "area_um2_percent_change",
        "volume_um3_percent_change",
        "circularity_percent_change",
        "sphericity_percent_change",
    ]
    matrix = effects.set_index(["tumour_type", "time_h"])[heatmap_columns]
    fig, ax = plt.subplots(figsize=(9, 5))
    image = ax.imshow(matrix.values, aspect="auto", cmap="coolwarm", vmin=-60, vmax=60)
    ax.set_yticks(range(len(matrix)))
    ax.set_yticklabels([f"{tumour}, {time} h" for tumour, time in matrix.index])
    ax.set_xticks(range(len(heatmap_columns)))
    ax.set_xticklabels(
        [column.replace("_percent_change", "").replace("_", " ") for column in heatmap_columns],
        rotation=35,
        ha="right",
    )
    fig.colorbar(image, ax=ax, label="Doxorubicin change (%)")
    ax.set_title("Treatment-associated remodelling is cell-line and time dependent")
    save("fig07_dox_effect_heatmap")

    for index, symbol, label in [
        (8, "R", "Outer radius"),
        (9, "I", "Growth-inhibited radius"),
        (10, "N", "Necrotic radius"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for initial_condition, group in raw[symbol].groupby("InitialCondition"):
            median = group.groupby("Day").Radius.median()
            ax.plot(median.index, median.values, label=str(initial_condition))
        ax.set_xlabel("Day")
        ax.set_ylabel(label + r" ($\mu$m)")
        ax.legend(title="Initial cells", fontsize=7)
        ax.set_title("Longitudinal group medians")
        save(f"fig{index:02d}_{symbol}_longitudinal")


    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    for ax, symbol, label in zip(
        axes.flat[:3],
        ["R", "I", "N"],
        ["Outer radius", "Growth-inhibited radius", "Necrotic radius"],
    ):
        for initial_condition, group in raw[symbol].groupby("InitialCondition"):
            median = group.groupby("Day").Radius.median()
            ax.plot(median.index, median.values, marker="o", label=str(initial_condition))
        ax.set_xlabel("Day")
        ax.set_ylabel(label + r" ($\mu$m)")
        ax.set_title(label)
    onset_ax = axes.flat[3]
    onset_ax.scatter(onset.initial_condition, onset.observed_onset_day, s=55)
    for _, row in onset.iterrows():
        onset_ax.annotate(
            f"{int(row.observed_onset_day)}",
            (row.initial_condition, row.observed_onset_day),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
        )
    onset_ax.set_xscale("log")
    onset_ax.set_xlabel("Initial cell number")
    onset_ax.set_ylabel(r"First sampled day with median $R_N>5\,\mu$m")
    onset_ax.set_title("Operational threshold crossing")
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, title="Initial cells", loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save("fig08_front_dynamics")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_aspect("equal")
    ax.axis("off")
    for radius, label in [
        (1.0, "viable outer shell"),
        (0.68, "growth-inhibited shell"),
        (0.34, "necrotic core"),
    ]:
        ax.add_patch(plt.Circle((0, 0), radius, fill=False, lw=2))
        ax.text(0, radius - 0.08, label, ha="center", va="top")
    ax.annotate("$e_{IV}$", xy=(0.65, 0), xytext=(0.38, 0), arrowprops=dict(arrowstyle="<->"))
    ax.annotate("$e_{NI}$", xy=(0.32, 0), xytext=(0.05, 0), arrowprops=dict(arrowstyle="<->"))
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_title("Nested-shell physical complex")
    save("fig11_nested_complex")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis("off")
    ax.text(0.03, 0.80, "Pre-necrotic: D- = [-1, 1]", fontsize=13, family="monospace")
    ax.text(
        0.48,
        0.80,
        "Post-necrotic: D+ = [[-1, 1, 0], [0, -1, 1]]",
        fontsize=13,
        family="monospace",
    )
    ax.text(0.03, 0.50, "P_C = [[1, 0], [1, 0], [0, 1]]", fontsize=13, family="monospace")
    ax.text(0.57, 0.50, "P_E = [[0], [1]]", fontsize=13, family="monospace")
    ax.text(0.5, 0.26, r"$D^+P_C=P_ED^-$", fontsize=20, ha="center")
    ax.text(
        0.5,
        0.07,
        "The inhibited-viable interface persists; a necrotic-inhibited interface is born.",
        ha="center",
    )
    save("fig12_incidence_chain_map")



    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(onset.initial_condition, onset.observed_onset_day, s=60)
    for _, row in onset.iterrows():
        ax.annotate(
            f"{int(row.observed_onset_day)}",
            (row.initial_condition, row.observed_onset_day),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Initial cell number")
    ax.set_ylabel(r"First sampled day with median $R_N>5\,\mu$m")
    ax.set_title("Operational necrosis-threshold crossing")
    save("fig13_necrosis_onset")

    ordered = metrics.sort_values("mean_relative_state_error", ascending=False)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    y = np.arange(len(ordered))
    ax.hlines(y, 0.07, ordered.mean_relative_state_error, linewidth=1)
    ax.scatter(ordered.mean_relative_state_error, y, s=55)
    ax.set_yticks(y)
    ax.set_yticklabels(ordered.model)
    ax.set_xscale("log")
    ax.set_xlim(0.07, 1.0)
    ax.set_xlabel("Mean relative three-front forecast error (log scale)")
    ax.set_title("Temporally held-out one-step forecasts at days 21 and 24")
    save("fig14_forecast_comparison")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(metrics))
    width = 0.25
    for offset, column in enumerate(["rmse_outer", "rmse_inhibited", "rmse_necrotic"]):
        ax.bar(x + (offset - 1) * width, metrics[column], width, label=column.replace("rmse_", ""))
    ax.set_xticks(x)
    ax.set_xticklabels(metrics.model, rotation=34, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel(r"RMSE ($\mu$m; log scale)")
    ax.legend()
    ax.set_title("Forecast error by moving front")
    save("fig15_compartment_rmse")


    audit_rows = [
        ("Necrotic-birth chain map", r"$\|D^+P_C-P_ED^-\|_F$", float(audits["necrotic_birth_commutation_residual"])),
        ("Interface-orientation invariance", r"$\|D^TD-D_r^TD_r\|_F$", float(audits["orientation_invariance_residual"])),
        ("Radius-capacity round trip", "max. reconstruction error", float(audits["capacity_roundtrip_residual"])),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 2.8))
    ax.axis("off")
    table_text = [[name, definition, f"{value:.3e}"] for name, definition, value in audit_rows]
    table = ax.table(cellText=table_text, colLabels=["Audit", "Residual definition", "Computed residual"], cellLoc="left", colLoc="left", loc="center", colWidths=[0.39, 0.39, 0.22])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.45)
    ax.set_title("Algebraic and reconstruction audits", pad=12)
    save("fig16_constraint_audit")

    selected = metrics[
        metrics.model.isin(
            [
                "CMC-EventForest",
                "CMC-TimeConditionedMLP",
                "Matched raw-radii forest",
                "CMC without initial condition",
            ]
        )
    ].sort_values("mean_relative_state_error", ascending=False)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    positions = np.arange(len(selected))
    ax.barh(positions, selected.mean_relative_state_error)
    ax.set_yticks(positions)
    ax.set_yticklabels(selected.model)
    ax.set_xlabel("Mean relative error")
    ax.set_title("Matched representation and feature ablations")
    save("fig17_ablation")

    best_prediction = predictions["CMC-EventForest"]
    fig, ax = plt.subplots(figsize=(5.6, 5.3))
    for index, label in enumerate(["outer", "growth-inhibited", "necrotic"]):
        ax.scatter(true[:, index], best_prediction[:, index], label=label, s=45)
    lower = min(true.min(), best_prediction.min())
    upper = max(true.max(), best_prediction.max())
    ax.plot([lower, upper], [lower, upper], "--")
    ax.set_xlabel(r"Observed radius ($\mu$m)")
    ax.set_ylabel(r"Predicted radius ($\mu$m)")
    ax.legend()
    ax.set_title("Observed-predicted agreement for held-out radii")
    save("fig18_calibration")

    fig, ax = plt.subplots(figsize=(6, 5))
    for initial_condition, group in wide.groupby("InitialCondition"):
        ax.plot(
            group.I / group.R,
            np.divide(group.N, group.I, out=np.zeros(len(group)), where=group.I > 0),
            marker="o",
            label=str(initial_condition),
        )
    ax.set_xlabel(r"$R_I/R$")
    ax.set_ylabel(r"$R_N/R_I$")
    ax.legend(title="Initial cells")
    ax.set_title("Normalised shell-state trajectories")
    save("fig19_phase_portrait")


    residuals = best_prediction - true
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["outer", "inhibited", "necrotic"]
    rng = np.random.default_rng(SEED)
    for index, label in enumerate(labels):
        x = np.full(residuals.shape[0], index, dtype=float) + rng.normal(0.0, 0.035, residuals.shape[0])
        ax.scatter(x, residuals[:, index], s=34)
        ax.hlines(residuals[:, index].mean(), index - 0.18, index + 0.18, linewidth=2)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"Prediction residual ($\mu$m)")
    ax.set_title("Case-level CMC-EventForest residuals")
    save("fig20_residuals")


def bibliography_audit() -> None:
    tex_paths = [ROOT / "manuscript" / "main.tex", ROOT / "supplementary" / "supplement.tex"]
    bib_text = (ROOT / "manuscript" / "references.bib").read_text(errors="ignore")
    bibliography_keys = set(re.findall(r"@[A-Za-z]+\s*\{\s*([^,\s]+)", bib_text))
    citation_keys: set[str] = set()
    for path in tex_paths:
        text = path.read_text(errors="ignore")
        for group in re.findall(r"\\cite[a-zA-Z*]*\{([^}]+)\}", text):
            citation_keys.update(key.strip() for key in group.split(",") if key.strip())
    missing = sorted(citation_keys - bibliography_keys)
    report = pd.DataFrame(
        [
            {
                "bibliography_records": len(bibliography_keys),
                "unique_cited_keys": len(citation_keys),
                "missing_citation_keys": len(missing),
                "missing_keys": ";".join(missing),
            }
        ]
    )
    report.to_csv(RESULTS / "bibliography_audit.csv", index=False)
    if missing:
        raise RuntimeError(f"Missing bibliography keys: {missing}")


def main() -> None:
    raw, wide, pairs = load_longitudinal_data()
    train, test, true, predictions, metrics, case_errors = build_forecasts(wide, pairs)
    loio = build_loio(pairs)
    seed_sensitivity = build_seed_sensitivity(train, test, true)
    onset = analyse_necrosis_onset(wide)
    dox_data, effects = analyse_doxorubicin()
    audits = write_audits(wide, train, test)
    write_tables_and_macros(metrics, onset, effects, loio, raw, case_errors, seed_sensitivity)
    generate_figures(raw, wide, dox_data, effects, onset, true, predictions, metrics, audits)
    bibliography_audit()
    summary = {
        "best_model": metrics.iloc[0].model,
        "best_mean_relative_error": float(metrics.iloc[0].mean_relative_state_error),
        "training_transitions": len(train),
        "test_transitions": len(test),
        "figures_pdf": len(list(FIGURES.glob("fig*.pdf"))),
        "figures_png": len(list(FIGURES.glob("fig*.png"))),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
