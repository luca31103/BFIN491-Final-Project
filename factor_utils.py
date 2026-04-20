
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.data_utils import normalize_text

try:
    import statsmodels.api as sm
except Exception:  # pragma: no cover - defensive fallback
    sm = None


PORTFOLIO_LABELS = {
    "legacy_return": "Legacy Fund",
    "revised_static_return": "Revised Static Fund",
    "revised_active_return": "Revised Active Fund",
}


def _standardize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        return out
    for candidate in ["Date", "index"]:
        if candidate in out.columns:
            return out.rename(columns={candidate: "date"})
    return out


def build_portfolio_return_panel(
    legacy_fund_daily: pd.DataFrame,
    static_daily: pd.DataFrame | None,
    active_daily: pd.DataFrame | None,
    benchmark_daily: pd.DataFrame,
    oos_start: pd.Timestamp,
    oos_end: pd.Timestamp,
) -> pd.DataFrame:
    """Assemble a clean daily return panel for factor/regression work."""
    frames: list[pd.DataFrame] = []

    legacy = _standardize_date_column(legacy_fund_daily)
    legacy["date"] = pd.to_datetime(legacy["date"], errors="coerce")
    legacy = legacy.dropna(subset=["date"]).set_index("date").sort_index()
    if "legacy_fund_return" in legacy.columns:
        frames.append(legacy[["legacy_fund_return"]].rename(columns={"legacy_fund_return": "legacy_return"}))

    if static_daily is not None and not static_daily.empty:
        static = _standardize_date_column(static_daily)
        static["date"] = pd.to_datetime(static["date"], errors="coerce")
        static = static.dropna(subset=["date"]).set_index("date").sort_index()
        if "revised_static_return" in static.columns:
            frames.append(static[["revised_static_return"]])

    if active_daily is not None and not active_daily.empty:
        active = _standardize_date_column(active_daily)
        active["date"] = pd.to_datetime(active["date"], errors="coerce")
        active = active.dropna(subset=["date"]).set_index("date").sort_index()
        if "revised_active_return" in active.columns:
            frames.append(active[["revised_active_return"]])

    benchmark = _standardize_date_column(benchmark_daily)
    benchmark["date"] = pd.to_datetime(benchmark["date"], errors="coerce")
    benchmark = benchmark.dropna(subset=["date"]).set_index("date").sort_index()
    if "benchmark_return" in benchmark.columns:
        frames.append(benchmark[["benchmark_return"]])

    if not frames:
        return pd.DataFrame()

    panel = pd.concat(frames, axis=1, join="outer").sort_index()
    panel = panel.loc[(panel.index >= pd.Timestamp(oos_start)) & (panel.index <= pd.Timestamp(oos_end))].copy()
    return panel


def _simple_ols(y: np.ndarray, x: np.ndarray) -> dict[str, float]:
    """Fallback OLS if statsmodels is unavailable."""
    X = np.column_stack([np.ones(len(x)), x])
    beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta_hat
    resid = y - y_hat
    n = len(y)
    k = X.shape[1]
    sse = float((resid**2).sum())
    sst = float(((y - y.mean())**2).sum())
    r2 = np.nan if sst == 0 else 1.0 - sse / sst
    sigma2 = np.nan if n <= k else sse / (n - k)
    try:
        var_beta = sigma2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(var_beta))
        tvals = beta_hat / se
    except Exception:
        tvals = np.array([np.nan, np.nan])

    return {
        "alpha_daily": float(beta_hat[0]),
        "beta": float(beta_hat[1]),
        "alpha_tstat": float(tvals[0]) if np.isfinite(tvals[0]) else np.nan,
        "beta_tstat": float(tvals[1]) if np.isfinite(tvals[1]) else np.nan,
        "alpha_pvalue": np.nan,
        "beta_pvalue": np.nan,
        "r_squared": float(r2) if np.isfinite(r2) else np.nan,
    }


def run_capm_regression(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    portfolio_name: str,
    risk_free_daily: float = 0.0,
) -> dict[str, Any]:
    aligned = pd.concat(
        [
            pd.to_numeric(portfolio_returns, errors="coerce").rename("portfolio"),
            pd.to_numeric(benchmark_returns, errors="coerce").rename("benchmark"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    if aligned.shape[0] < 20:
        return {
            "portfolio": portfolio_name,
            "n_obs": int(aligned.shape[0]),
            "alpha_daily": np.nan,
            "alpha_ann": np.nan,
            "beta": np.nan,
            "alpha_tstat": np.nan,
            "alpha_pvalue": np.nan,
            "beta_tstat": np.nan,
            "beta_pvalue": np.nan,
            "r_squared": np.nan,
            "corr_to_benchmark": np.nan,
            "market_proxy": "benchmark_return",
            "risk_free_assumption": risk_free_daily,
            "notes": "Not enough aligned daily observations for CAPM regression.",
        }

    y = aligned["portfolio"].astype(float) - float(risk_free_daily)
    x = aligned["benchmark"].astype(float) - float(risk_free_daily)

    if sm is not None:
        X = sm.add_constant(x.values, has_constant="add")
        model = sm.OLS(y.values, X).fit()
        alpha_daily = float(model.params[0])
        beta = float(model.params[1]) if len(model.params) > 1 else np.nan
        alpha_tstat = float(model.tvalues[0]) if len(model.tvalues) > 0 else np.nan
        alpha_pvalue = float(model.pvalues[0]) if len(model.pvalues) > 0 else np.nan
        beta_tstat = float(model.tvalues[1]) if len(model.tvalues) > 1 else np.nan
        beta_pvalue = float(model.pvalues[1]) if len(model.pvalues) > 1 else np.nan
        r_squared = float(model.rsquared)
    else:
        fallback = _simple_ols(y.values, x.values)
        alpha_daily = fallback["alpha_daily"]
        beta = fallback["beta"]
        alpha_tstat = fallback["alpha_tstat"]
        alpha_pvalue = fallback["alpha_pvalue"]
        beta_tstat = fallback["beta_tstat"]
        beta_pvalue = fallback["beta_pvalue"]
        r_squared = fallback["r_squared"]

    alpha_ann = alpha_daily * 252.0
    corr = float(aligned["portfolio"].corr(aligned["benchmark"]))

    return {
        "portfolio": portfolio_name,
        "n_obs": int(aligned.shape[0]),
        "alpha_daily": alpha_daily,
        "alpha_ann": alpha_ann,
        "beta": beta,
        "alpha_tstat": alpha_tstat,
        "alpha_pvalue": alpha_pvalue,
        "beta_tstat": beta_tstat,
        "beta_pvalue": beta_pvalue,
        "r_squared": r_squared,
        "corr_to_benchmark": corr,
        "market_proxy": "benchmark_return",
        "risk_free_assumption": risk_free_daily,
        "notes": "Starter CAPM uses the benchmark daily return as the market proxy and assumes daily risk-free rate = 0.",
    }


def build_capm_summary(
    return_panel: pd.DataFrame,
    benchmark_col: str = "benchmark_return",
    risk_free_daily: float = 0.0,
) -> pd.DataFrame:
    if return_panel.empty or benchmark_col not in return_panel.columns:
        return pd.DataFrame(
            columns=[
                "portfolio",
                "n_obs",
                "alpha_daily",
                "alpha_ann",
                "beta",
                "alpha_tstat",
                "alpha_pvalue",
                "beta_tstat",
                "beta_pvalue",
                "r_squared",
                "corr_to_benchmark",
                "market_proxy",
                "risk_free_assumption",
                "notes",
            ]
        )

    rows: list[dict[str, Any]] = []
    for col in [c for c in return_panel.columns if c != benchmark_col]:
        label = PORTFOLIO_LABELS.get(col, normalize_text(col))
        rows.append(
            run_capm_regression(
                portfolio_returns=return_panel[col],
                benchmark_returns=return_panel[benchmark_col],
                portfolio_name=label,
                risk_free_daily=risk_free_daily,
            )
        )

    return pd.DataFrame(rows)


def build_rolling_beta_table(
    return_panel: pd.DataFrame,
    benchmark_col: str = "benchmark_return",
    window: int = 63,
) -> pd.DataFrame:
    if return_panel.empty or benchmark_col not in return_panel.columns:
        return pd.DataFrame(columns=["date", "portfolio", "rolling_beta", "window"])

    benchmark = pd.to_numeric(return_panel[benchmark_col], errors="coerce")
    rows: list[pd.DataFrame] = []

    for col in [c for c in return_panel.columns if c != benchmark_col]:
        port = pd.to_numeric(return_panel[col], errors="coerce")
        beta = port.rolling(window).cov(benchmark) / benchmark.rolling(window).var()
        tmp = pd.DataFrame(
            {
                "date": beta.index,
                "portfolio": PORTFOLIO_LABELS.get(col, normalize_text(col)),
                "rolling_beta": beta.values,
                "window": int(window),
            }
        )
        rows.append(tmp)

    if not rows:
        return pd.DataFrame(columns=["date", "portfolio", "rolling_beta", "window"])

    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.dropna(subset=["date"]).reset_index(drop=True)


def plot_factor_alpha_beta(capm_summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    valid = capm_summary.dropna(subset=["beta", "alpha_ann"]).copy() if not capm_summary.empty else pd.DataFrame()

    if valid.empty:
        ax.text(0.5, 0.5, "Factor/regression outputs are not ready yet.", ha="center", va="center")
        ax.set_axis_off()
        return

    ax.scatter(valid["beta"], valid["alpha_ann"])
    for _, row in valid.iterrows():
        ax.annotate(
            normalize_text(row.get("portfolio")),
            (row["beta"], row["alpha_ann"]),
            fontsize=8,
            xytext=(3, 3),
            textcoords="offset points",
        )
    ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("CAPM Summary: Beta vs Annualized Alpha")
    ax.set_xlabel("Benchmark beta")
    ax.set_ylabel("Annualized alpha (daily alpha × 252)")
    ax.grid(alpha=0.25)


def plot_rolling_beta(rolling_beta_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    if rolling_beta_df.empty:
        ax.text(0.5, 0.5, "Rolling-beta table is not ready yet.", ha="center", va="center")
        ax.set_axis_off()
        return

    chart_df = rolling_beta_df.copy()
    chart_df["date"] = pd.to_datetime(chart_df["date"], errors="coerce")
    chart_df = chart_df.dropna(subset=["date"])
    if chart_df.empty:
        ax.text(0.5, 0.5, "Rolling-beta dates could not be parsed.", ha="center", va="center")
        ax.set_axis_off()
        return

    pivot = chart_df.pivot(index="date", columns="portfolio", values="rolling_beta").sort_index()
    if pivot.empty:
        ax.text(0.5, 0.5, "Rolling-beta data is empty after pivoting.", ha="center", va="center")
        ax.set_axis_off()
        return

    for col in pivot.columns:
        ax.plot(pivot.index, pivot[col], label=normalize_text(col), linewidth=1.5)

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Rolling 63-Day Benchmark Beta")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling beta")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
