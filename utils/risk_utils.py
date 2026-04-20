
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.data_utils import normalize_text


def _normalized_weight_series(weights: pd.Series) -> pd.Series:
    s = pd.to_numeric(weights, errors="coerce").astype(float).fillna(0.0)
    s = s[s.index.map(lambda x: normalize_text(x) != "")]
    if s.empty or s.sum() <= 0:
        return s
    return s / s.sum()


def _get_static_weights(selection_df: pd.DataFrame) -> pd.Series:
    if selection_df.empty or "ticker" not in selection_df.columns or "static_weight" not in selection_df.columns:
        return pd.Series(dtype=float)
    weights = pd.Series(
        pd.to_numeric(selection_df["static_weight"], errors="coerce").values,
        index=selection_df["ticker"].map(normalize_text).str.upper(),
        dtype=float,
    )
    return _normalized_weight_series(weights)


def _get_active_weights(active_result: dict[str, Any] | None, static_weights: pd.Series) -> pd.Series:
    if active_result is None:
        return static_weights.copy()

    weights_snapshot = active_result.get("weights_snapshot")
    if isinstance(weights_snapshot, pd.DataFrame) and not weights_snapshot.empty:
        if {"ticker", "latest_active_weight"}.issubset(weights_snapshot.columns):
            weights = pd.Series(
                pd.to_numeric(weights_snapshot["latest_active_weight"], errors="coerce").values,
                index=weights_snapshot["ticker"].map(normalize_text).str.upper(),
                dtype=float,
            )
            weights = _normalized_weight_series(weights)
            if not weights.empty:
                return weights

    weights_daily = active_result.get("weights_daily")
    if isinstance(weights_daily, pd.DataFrame) and not weights_daily.empty:
        tmp = weights_daily.copy()
        ignore_cols = {"date", "rebalance_turnover", "transaction_cost_rate", "rule_used"}
        cols = [c for c in tmp.columns if c not in ignore_cols]
        if cols:
            last = tmp.iloc[-1][cols]
            weights = pd.Series(pd.to_numeric(last, errors="coerce"), index=cols, dtype=float)
            weights = _normalized_weight_series(weights)
            if not weights.empty:
                return weights

    return static_weights.copy()


def _nearest_psd(cov: np.ndarray) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    psd = 0.5 * (psd + psd.T)
    jitter = 1e-10 * np.eye(psd.shape[0])
    return psd + jitter


def _simulate_terminal_returns(
    mean_vec: np.ndarray,
    cov: np.ndarray,
    weights: np.ndarray,
    horizon_days: int,
    n_sims: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    cov_psd = _nearest_psd(cov)
    sims = rng.multivariate_normal(mean=mean_vec, cov=cov_psd, size=(n_sims, horizon_days), method="eigh")
    port_daily = np.tensordot(sims, weights, axes=([2], [0]))
    terminal = np.prod(1.0 + port_daily, axis=1) - 1.0
    return terminal.astype(float)


def _distribution_summary(simulated_terminal_returns: np.ndarray, portfolio: str, horizon_days: int, n_sims: int) -> dict[str, Any]:
    sims = pd.Series(simulated_terminal_returns).dropna().astype(float)
    if sims.empty:
        return {
            "portfolio": portfolio,
            "horizon_days": horizon_days,
            "n_sims": n_sims,
            "mean_terminal_return": np.nan,
            "median_terminal_return": np.nan,
            "pct_05_terminal_return": np.nan,
            "pct_01_terminal_return": np.nan,
            "var_95": np.nan,
            "cvar_95": np.nan,
            "prob_loss": np.nan,
        }

    q05 = float(np.quantile(sims, 0.05))
    q01 = float(np.quantile(sims, 0.01))
    tail = sims[sims <= q05]
    cvar_95 = float(tail.mean()) if not tail.empty else np.nan
    return {
        "portfolio": portfolio,
        "horizon_days": int(horizon_days),
        "n_sims": int(n_sims),
        "mean_terminal_return": float(sims.mean()),
        "median_terminal_return": float(sims.median()),
        "pct_05_terminal_return": q05,
        "pct_01_terminal_return": q01,
        "var_95": float(-q05),
        "cvar_95": float(-cvar_95) if pd.notna(cvar_95) else np.nan,
        "prob_loss": float((sims < 0).mean()),
    }


def build_scenario_outputs(
    price_panel: pd.DataFrame,
    selection_df: pd.DataFrame,
    active_result: dict[str, Any] | None,
    constraints: dict[str, Any],
    oos_start: pd.Timestamp,
    oos_end: pd.Timestamp,
) -> dict[str, Any]:
    static_weights = _get_static_weights(selection_df)
    if static_weights.empty:
        empty_summary = pd.DataFrame(
            columns=[
                "portfolio",
                "horizon_days",
                "n_sims",
                "mean_terminal_return",
                "median_terminal_return",
                "pct_05_terminal_return",
                "pct_01_terminal_return",
                "var_95",
                "cvar_95",
                "prob_loss",
            ]
        )
        empty_stress = pd.DataFrame(
            columns=[
                "scenario",
                "scenario_date",
                "scenario_type",
                "benchmark_return",
                "equal_weight_universe_return",
                "static_portfolio_impact",
                "active_portfolio_impact",
                "notes",
            ]
        )
        return {
            "simulation_summary": empty_summary,
            "simulation_draws": pd.DataFrame(columns=["portfolio", "terminal_return"]),
            "stress_summary": empty_stress,
            "static_weights": static_weights,
            "active_weights": static_weights,
            "notes": ["Scenario layer is pending because the revised portfolio selection is not ready."],
        }

    active_weights = _get_active_weights(active_result, static_weights)
    tickers = sorted(set(static_weights.index) | set(active_weights.index))
    available = [ticker for ticker in tickers if ticker in price_panel.columns]
    if len(available) < 2:
        empty_summary = pd.DataFrame(
            [{"portfolio": "Revised Static Fund", "notes": "Not enough selected tickers have price history for simulation."}]
        )
        return {
            "simulation_summary": empty_summary,
            "simulation_draws": pd.DataFrame(columns=["portfolio", "terminal_return"]),
            "stress_summary": pd.DataFrame(),
            "static_weights": static_weights,
            "active_weights": active_weights,
            "notes": ["Not enough selected tickers have price history for the scenario layer."],
        }

    px = price_panel[available].copy().sort_index()
    returns = px.loc[(px.index >= pd.Timestamp(oos_start)) & (px.index <= pd.Timestamp(oos_end))].pct_change().dropna(how="any")
    if returns.shape[0] < 40:
        return {
            "simulation_summary": pd.DataFrame(
                [{"portfolio": "Revised Static Fund", "notes": "Not enough out-of-sample returns for the scenario layer."}]
            ),
            "simulation_draws": pd.DataFrame(columns=["portfolio", "terminal_return"]),
            "stress_summary": pd.DataFrame(),
            "static_weights": static_weights,
            "active_weights": active_weights,
            "notes": ["Not enough out-of-sample returns for the scenario layer."],
        }

    static_w = static_weights.reindex(available).fillna(0.0)
    static_w = _normalized_weight_series(static_w)
    active_w = active_weights.reindex(available).fillna(0.0)
    active_w = _normalized_weight_series(active_w)

    est_months = pd.to_numeric(constraints.get("estimation_window_months"), errors="coerce")
    est_months = 36 if pd.isna(est_months) else int(est_months)
    horizon_days = max(21, min(252, est_months * 21))
    n_sims = 2000
    seed = pd.to_numeric(constraints.get("random_seed"), errors="coerce")
    seed = 42 if pd.isna(seed) else int(seed)

    mean_vec = returns.mean().values.astype(float)
    cov = returns.cov().values.astype(float)

    static_terminal = _simulate_terminal_returns(mean_vec, cov, static_w.values, horizon_days, n_sims, seed)
    active_terminal = _simulate_terminal_returns(mean_vec, cov, active_w.values, horizon_days, n_sims, seed + 1)

    sim_summary = pd.DataFrame(
        [
            _distribution_summary(static_terminal, "Revised Static Fund", horizon_days, n_sims),
            _distribution_summary(active_terminal, "Revised Active Fund", horizon_days, n_sims),
        ]
    )
    sim_draws = pd.DataFrame(
        {
            "portfolio": np.repeat(["Revised Static Fund", "Revised Active Fund"], [len(static_terminal), len(active_terminal)]),
            "terminal_return": np.concatenate([static_terminal, active_terminal]),
        }
    )

    # Historical and synthetic stress scenarios
    benchmark_proxy = returns.mean(axis=1)
    scenario_rows: list[dict[str, Any]] = []

    def add_scenario(name: str, date_value: Any, scenario_type: str, asset_returns: pd.Series, benchmark_value: float | None, notes: str) -> None:
        asset_vec = asset_returns.reindex(available).astype(float)
        ew_ret = float(asset_vec.mean())
        static_impact = float((static_w * asset_vec).sum())
        active_impact = float((active_w * asset_vec).sum())
        scenario_rows.append(
            {
                "scenario": name,
                "scenario_date": "" if date_value is None else pd.Timestamp(date_value).strftime("%Y-%m-%d"),
                "scenario_type": scenario_type,
                "benchmark_return": np.nan if benchmark_value is None else float(benchmark_value),
                "equal_weight_universe_return": ew_ret,
                "static_portfolio_impact": static_impact,
                "active_portfolio_impact": active_impact,
                "notes": notes,
            }
        )

    covid_date = pd.Timestamp("2020-03-16")
    if covid_date in returns.index:
        row = returns.loc[covid_date]
        add_scenario(
            "COVID Crash Day",
            covid_date,
            "historical",
            row,
            float(benchmark_proxy.loc[covid_date]) if covid_date in benchmark_proxy.index else None,
            "Uses actual selected-universe daily returns on 2020-03-16.",
        )

    worst_day = benchmark_proxy.idxmin()
    add_scenario(
        "Worst OOS Day",
        worst_day,
        "historical",
        returns.loc[worst_day],
        float(benchmark_proxy.loc[worst_day]),
        "Uses the worst benchmark-proxy day in the out-of-sample return window.",
    )

    worst_21_idx = benchmark_proxy.rolling(21).mean().idxmin()
    if pd.notna(worst_21_idx):
        window = returns.loc[:worst_21_idx].tail(21)
        window_asset_ret = (1.0 + window).prod() - 1.0
        add_scenario(
            "Worst 21-Day Window",
            worst_21_idx,
            "historical_window",
            window_asset_ret,
            float((1.0 + benchmark_proxy.loc[window.index]).prod() - 1.0),
            "Uses compounded selected-universe returns over the worst 21-day benchmark-proxy window.",
        )

    parallel_10 = pd.Series(-0.10, index=available, dtype=float)
    add_scenario(
        "Parallel -10% Shock",
        None,
        "synthetic",
        parallel_10,
        -0.10,
        "Synthetic shock: every selected stock falls by 10% at the same time.",
    )

    parallel_20 = pd.Series(-0.20, index=available, dtype=float)
    add_scenario(
        "Parallel -20% Shock",
        None,
        "synthetic",
        parallel_20,
        -0.20,
        "Synthetic shock: every selected stock falls by 20% at the same time.",
    )

    stress_summary = pd.DataFrame(scenario_rows)

    return {
        "simulation_summary": sim_summary,
        "simulation_draws": sim_draws,
        "stress_summary": stress_summary,
        "static_weights": static_w,
        "active_weights": active_w,
        "notes": [
            "Monte Carlo uses multivariate-normal daily returns fitted on the out-of-sample selected-universe panel.",
            "Stress scenarios combine historical shocks and simple synthetic parallel shocks.",
        ],
    }


def plot_monte_carlo_distribution(simulation_draws: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    if simulation_draws.empty:
        ax.text(0.5, 0.5, "Scenario simulation outputs are not ready yet.", ha="center", va="center")
        ax.set_axis_off()
        return

    for portfolio, group in simulation_draws.groupby("portfolio"):
        ax.hist(group["terminal_return"].dropna(), bins=50, alpha=0.45, density=True, label=normalize_text(portfolio))
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_title("Monte Carlo Terminal Return Distribution")
    ax.set_xlabel("Terminal return")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.25)


def plot_stress_scenarios(stress_summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    if stress_summary.empty:
        ax.text(0.5, 0.5, "Stress-scenario outputs are not ready yet.", ha="center", va="center")
        ax.set_axis_off()
        return

    chart = stress_summary.copy()
    labels = chart["scenario"].map(normalize_text).tolist()
    x = np.arange(len(chart))
    width = 0.35

    static_vals = pd.to_numeric(chart.get("static_portfolio_impact", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    active_vals = pd.to_numeric(chart.get("active_portfolio_impact", pd.Series(dtype=float)), errors="coerce").fillna(0.0)

    ax.bar(x - width / 2, static_vals, width, label="Revised Static Fund")
    ax.bar(x + width / 2, active_vals, width, label="Revised Active Fund")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Portfolio Impacts Under Stress Scenarios")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Portfolio return impact")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
