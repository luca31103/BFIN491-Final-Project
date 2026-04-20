
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.data_utils import (
    TRADING_DAYS_PER_YEAR,
    clean_candidate_rows,
    compute_drawdown,
    compute_performance_metrics,
    normalize_text,
)


def _first_nonblank_text(*values: Any) -> str:
    for value in values:
        text = normalize_text(value)
        if text:
            return text
    return ""


def _standardize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        return out
    for candidate in ["Date", "index"]:
        if candidate in out.columns:
            return out.rename(columns={candidate: "date"})
    return out


def _window_return(price_series: pd.Series, lookback_days: int) -> float:
    px = price_series.dropna().astype(float)
    if px.shape[0] < lookback_days + 1:
        return np.nan
    start_price = px.iloc[-(lookback_days + 1)]
    end_price = px.iloc[-1]
    if start_price == 0 or pd.isna(start_price) or pd.isna(end_price):
        return np.nan
    return float(end_price / start_price - 1.0)


def _annualized_vol(daily_returns: pd.Series) -> float:
    daily_returns = daily_returns.dropna().astype(float)
    if daily_returns.empty:
        return np.nan
    return float(daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def _safe_correlation(left: pd.Series, right: pd.Series) -> float:
    aligned = pd.concat([left, right], axis=1, join="inner").dropna()
    if aligned.shape[0] < 3:
        return np.nan
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))


def _safe_beta(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    aligned = pd.concat([asset_returns, benchmark_returns], axis=1, join="inner").dropna()
    if aligned.shape[0] < 3:
        return np.nan
    bench_var = aligned.iloc[:, 1].var(ddof=1)
    if pd.isna(bench_var) or bench_var == 0:
        return np.nan
    covariance = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
    return float(covariance / bench_var)


def build_candidate_screen(
    price_panel: pd.DataFrame,
    candidates_df: pd.DataFrame,
    decision_date: pd.Timestamp,
    benchmark_ticker: str,
    legacy_fund_daily: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a candidate research table using only information available through the decision date.
    """
    candidates = clean_candidate_rows(candidates_df)
    benchmark = normalize_text(benchmark_ticker).upper()

    empty_columns = [
        "candidate_ticker",
        "download_ticker",
        "company_name",
        "sector_theme",
        "thesis_1line",
        "screening_note",
        "add_decision",
        "selected_for_final",
        "status",
        "first_date",
        "last_date",
        "n_obs",
        "total_return_pre2020",
        "ann_return_pre2020",
        "ann_vol_pre2020",
        "sharpe_pre2020",
        "max_drawdown_pre2020",
        "return_1y_pre2020",
        "return_3m_pre2020",
        "vol_63d_pre2020",
        "current_drawdown_pre2020",
        "corr_to_legacy_pre2020",
        "corr_to_benchmark_pre2020",
        "beta_to_benchmark_pre2020",
        "notes",
    ]
    if candidates.empty:
        return pd.DataFrame(columns=empty_columns)

    legacy_series = legacy_fund_daily.copy()
    legacy_series["date"] = pd.to_datetime(legacy_series["date"], errors="coerce")
    legacy_rets = (
        legacy_series.set_index("date")["legacy_fund_return"].astype(float).loc[:decision_date].dropna()
    )

    benchmark_rets = pd.Series(dtype=float)
    if benchmark in price_panel.columns:
        benchmark_rets = (
            price_panel[benchmark].astype(float).loc[:decision_date].dropna().pct_change().dropna()
        )

    rows: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        candidate_ticker = normalize_text(row.get("candidate_ticker")).upper()
        download_ticker = _first_nonblank_text(row.get("download_ticker"), candidate_ticker).upper()

        record: dict[str, Any] = {
            "candidate_ticker": candidate_ticker,
            "download_ticker": download_ticker,
            "company_name": normalize_text(row.get("company_name")),
            "sector_theme": normalize_text(row.get("sector_theme")),
            "thesis_1line": normalize_text(row.get("thesis_1line")),
            "screening_note": normalize_text(row.get("screening_note")),
            "add_decision": normalize_text(row.get("add_decision")),
            "selected_for_final": normalize_text(row.get("selected_for_final")),
            "status": "ok",
            "first_date": "",
            "last_date": "",
            "n_obs": 0,
            "total_return_pre2020": np.nan,
            "ann_return_pre2020": np.nan,
            "ann_vol_pre2020": np.nan,
            "sharpe_pre2020": np.nan,
            "max_drawdown_pre2020": np.nan,
            "return_1y_pre2020": np.nan,
            "return_3m_pre2020": np.nan,
            "vol_63d_pre2020": np.nan,
            "current_drawdown_pre2020": np.nan,
            "corr_to_legacy_pre2020": np.nan,
            "corr_to_benchmark_pre2020": np.nan,
            "beta_to_benchmark_pre2020": np.nan,
            "notes": normalize_text(row.get("notes")),
        }

        if download_ticker not in price_panel.columns:
            record["status"] = "missing_price_series"
            record["notes"] = (record["notes"] + " Price series not found in downloaded panel.").strip()
            rows.append(record)
            continue

        px = price_panel[download_ticker].astype(float).loc[:decision_date].dropna()
        if px.shape[0] < 3:
            record["status"] = "insufficient_history"
            record["notes"] = (record["notes"] + " Not enough pre-2020 history to compute metrics.").strip()
            rows.append(record)
            continue

        rets = px.pct_change().dropna()
        metrics = compute_performance_metrics(rets, turnover=0.0)
        growth = (1.0 + rets).cumprod()
        current_drawdown = compute_drawdown(growth).iloc[-1]

        record.update(
            {
                "first_date": px.index.min().strftime("%Y-%m-%d"),
                "last_date": px.index.max().strftime("%Y-%m-%d"),
                "n_obs": int(px.shape[0]),
                "total_return_pre2020": metrics["total_return"],
                "ann_return_pre2020": metrics["ann_return"],
                "ann_vol_pre2020": metrics["ann_vol"],
                "sharpe_pre2020": metrics["sharpe"],
                "max_drawdown_pre2020": metrics["max_drawdown"],
                "return_1y_pre2020": _window_return(px, 252),
                "return_3m_pre2020": _window_return(px, 63),
                "vol_63d_pre2020": _annualized_vol(rets.tail(63)),
                "current_drawdown_pre2020": float(current_drawdown),
                "corr_to_legacy_pre2020": _safe_correlation(rets, legacy_rets),
                "corr_to_benchmark_pre2020": _safe_correlation(rets, benchmark_rets),
                "beta_to_benchmark_pre2020": _safe_beta(rets, benchmark_rets),
            }
        )
        rows.append(record)

    screen = pd.DataFrame(rows)
    if screen.empty:
        return pd.DataFrame(columns=empty_columns)

    screen["selected_rank"] = screen["selected_for_final"].map(normalize_text).str.lower().eq("yes").astype(int)
    screen["add_rank"] = screen["add_decision"].map(normalize_text).str.lower().eq("add").astype(int)
    screen = screen.sort_values(
        ["selected_rank", "add_rank", "sharpe_pre2020", "ann_return_pre2020"],
        ascending=[False, False, False, False],
        na_position="last",
    ).drop(columns=["selected_rank", "add_rank"])

    return screen.reset_index(drop=True)


def prepare_revised_static_selection(
    inherited_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
) -> dict[str, Any]:
    inherited = inherited_df.copy()
    candidates = clean_candidate_rows(candidates_df)

    inherited["keep_norm"] = inherited["keep_in_revised"].map(normalize_text).str.lower()
    inherited["static_weight"] = pd.to_numeric(inherited["target_weight_2020"], errors="coerce")

    selected_inherited = inherited.loc[inherited["keep_norm"].isin({"yes", "keep"})].copy()
    selected_inherited["ticker"] = selected_inherited.apply(
        lambda r: _first_nonblank_text(r.get("download_ticker"), r.get("legacy_ticker")).upper(),
        axis=1,
    )
    selected_inherited["company_name"] = selected_inherited["company_name"].map(normalize_text)
    selected_inherited["sector"] = selected_inherited["sector"].map(normalize_text)
    selected_inherited["source"] = "Inherited"
    selected_inherited["keep_add_drop"] = "Keep"
    selected_inherited["latest_active_weight"] = np.nan
    selected_inherited["notes"] = selected_inherited["notes"].map(normalize_text)

    if candidates.empty:
        selected_candidates = pd.DataFrame(
            columns=[
                "ticker",
                "company_name",
                "sector",
                "source",
                "keep_add_drop",
                "static_weight",
                "latest_active_weight",
                "notes",
            ]
        )
    else:
        candidates["selected_norm"] = candidates["selected_for_final"].map(normalize_text).str.lower()
        candidates["static_weight"] = pd.to_numeric(candidates["target_weight_2020"], errors="coerce")
        selected_candidates = candidates.loc[candidates["selected_norm"].eq("yes")].copy()
        if not selected_candidates.empty:
            selected_candidates["ticker"] = selected_candidates.apply(
                lambda r: _first_nonblank_text(r.get("download_ticker"), r.get("candidate_ticker")).upper(),
                axis=1,
            )
            selected_candidates["company_name"] = selected_candidates["company_name"].map(normalize_text)
            selected_candidates["sector"] = selected_candidates["sector_theme"].map(normalize_text)
            selected_candidates["source"] = "Candidate"
            selected_candidates["keep_add_drop"] = "Add"
            selected_candidates["latest_active_weight"] = np.nan
            selected_candidates["notes"] = selected_candidates["notes"].map(normalize_text)

    summary_cols = [
        "ticker",
        "company_name",
        "sector",
        "source",
        "keep_add_drop",
        "static_weight",
        "latest_active_weight",
        "notes",
    ]

    frames: list[pd.DataFrame] = []
    if not selected_inherited.empty:
        frames.append(selected_inherited[summary_cols])
    if not selected_candidates.empty:
        frames.append(selected_candidates[summary_cols])

    if not frames:
        return {
            "ready": False,
            "selection": pd.DataFrame(columns=summary_cols),
            "warnings": [],
            "status": "not_started",
        }

    selection = pd.concat(frames, ignore_index=True, sort=False)
    warnings: list[str] = []

    # Drop duplicate tickers while preserving first occurrence.
    duplicate_tickers = selection["ticker"][selection["ticker"].duplicated()].tolist()
    if duplicate_tickers:
        warnings.append(
            f"Duplicate tickers were selected for the revised portfolio and kept only once: {', '.join(sorted(set(duplicate_tickers)))}."
        )
        selection = selection.drop_duplicates(subset=["ticker"], keep="first").reset_index(drop=True)

    selected_count = int(selection.shape[0])
    if selected_count != 10:
        warnings.append(
            f"Current revised portfolio selection count is {selected_count}. Final required portfolio should contain 10 stocks."
        )

    ready = False
    status = "incomplete"

    if selected_count == 10:
        weights = pd.to_numeric(selection["static_weight"], errors="coerce")
        if weights.isna().all():
            selection["static_weight"] = 1.0 / selected_count
            warnings.append(
                "All revised-static weights were blank. The starter pipeline defaulted to equal weights across the selected 10 stocks."
            )
            ready = True
            status = "default_equal_weight"
        elif weights.isna().any():
            warnings.append(
                "Some revised-static target weights are blank. Fill in all selected weights (or leave all blank for equal-weight default) before the static backtest can run."
            )
            ready = False
            status = "missing_weights"
        else:
            weight_sum = float(weights.sum())
            if weight_sum <= 0:
                warnings.append(
                    "Selected revised-static weights sum to zero or a negative value. The static backtest cannot run until weights are fixed."
                )
                ready = False
                status = "bad_weights"
            else:
                if not np.isclose(weight_sum, 1.0, atol=1e-4):
                    selection["static_weight"] = weights / weight_sum
                    warnings.append(
                        f"Selected revised-static weights summed to {weight_sum:.4f}. The starter pipeline normalized them to 1.0000 for the static backtest."
                    )
                ready = True
                status = "ready"

    return {
        "ready": ready,
        "selection": selection.reset_index(drop=True),
        "warnings": warnings,
        "status": status,
    }


def construct_revised_static_fund(
    price_panel: pd.DataFrame,
    selection_df: pd.DataFrame,
    legacy_fund_daily: pd.DataFrame,
    benchmark_daily: pd.DataFrame,
    oos_start: pd.Timestamp,
    oos_end: pd.Timestamp,
) -> dict[str, Any]:
    if selection_df.empty:
        raise ValueError("No revised-static selection was provided.")

    tickers = selection_df["ticker"].map(normalize_text).str.upper().tolist()
    missing = [ticker for ticker in tickers if ticker not in price_panel.columns]
    if missing:
        raise ValueError(f"Revised-static portfolio tickers are missing from the price panel: {', '.join(missing)}")

    px = price_panel[tickers].copy().sort_index()
    eligible = px.loc[(px.index >= oos_start) & (px.index <= oos_end)]
    start_mask = eligible.notna().all(axis=1)
    if not start_mask.any():
        raise ValueError("Could not find a common first trade date for the revised-static portfolio in the out-of-sample window.")
    first_trade_date = eligible.index[start_mask][0]

    legacy_df = _standardize_date_column(legacy_fund_daily)
    legacy_df["date"] = pd.to_datetime(legacy_df["date"], errors="coerce")
    legacy_df = legacy_df.dropna(subset=["date"]).set_index("date").sort_index()

    if first_trade_date not in legacy_df.index:
        raise ValueError("Legacy fund daily table does not contain the revised-static first trade date.")

    start_capital = float(legacy_df.loc[first_trade_date, "legacy_fund_value"])
    weights = pd.to_numeric(selection_df["static_weight"], errors="coerce").astype(float)
    start_prices = px.loc[first_trade_date].astype(float)
    shares = (start_capital * weights.values) / start_prices.values
    shares_series = pd.Series(shares, index=tickers, name="shares")

    position_values = px.loc[first_trade_date:oos_end].mul(shares_series, axis=1)
    total_value = position_values.sum(axis=1)
    daily_returns = total_value.pct_change().fillna(0.0)
    growth = (1.0 + daily_returns).cumprod()
    drawdown = compute_drawdown(growth)
    weights_daily = position_values.div(total_value, axis=0)

    static_daily = pd.DataFrame(
        {
            "date": total_value.index,
            "revised_static_value": total_value.values,
            "revised_static_return": daily_returns.values,
            "revised_static_growth_of_1": growth.values,
            "revised_static_drawdown": drawdown.values,
        }
    )

    weights_snapshot = selection_df.copy()
    weights_snapshot["snapshot_date"] = pd.Timestamp(first_trade_date).strftime("%Y-%m-%d")

    benchmark_df = _standardize_date_column(benchmark_daily)
    benchmark_df["date"] = pd.to_datetime(benchmark_df["date"], errors="coerce")
    benchmark_df = benchmark_df.dropna(subset=["date"]).set_index("date").sort_index()

    legacy_compare = legacy_df.loc[first_trade_date:oos_end, ["legacy_growth_of_1", "legacy_drawdown"]].copy()
    legacy_compare = legacy_compare.rename(
        columns={
            "legacy_growth_of_1": "legacy_growth_of_1_oos",
            "legacy_drawdown": "legacy_drawdown_oos",
        }
    )

    compare = legacy_compare.join(
        static_daily.set_index("date")[["revised_static_growth_of_1", "revised_static_drawdown"]],
        how="inner",
    )
    if {"benchmark_growth_of_1", "benchmark_drawdown"}.issubset(benchmark_df.columns):
        compare = compare.join(
            benchmark_df[["benchmark_growth_of_1", "benchmark_drawdown"]],
            how="inner",
        )

    compare = compare.reset_index().rename(columns={"index": "date"})
    static_metrics = compute_performance_metrics(daily_returns, turnover=0.0)

    return {
        "first_trade_date": pd.Timestamp(first_trade_date),
        "start_capital": start_capital,
        "static_daily": static_daily,
        "weights_daily": weights_daily,
        "weights_snapshot": weights_snapshot,
        "compare": compare,
        "metrics": static_metrics,
    }


def plot_candidate_risk_return(candidate_screen: pd.DataFrame) -> None:
    valid = candidate_screen.dropna(subset=["ann_vol_pre2020", "ann_return_pre2020"]).copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    if valid.empty:
        ax.text(0.5, 0.5, "No candidate metrics available yet.", ha="center", va="center")
        ax.set_axis_off()
        return

    ax.scatter(valid["ann_vol_pre2020"], valid["ann_return_pre2020"])
    for _, row in valid.iterrows():
        label = normalize_text(row.get("candidate_ticker"))
        if label:
            ax.annotate(label, (row["ann_vol_pre2020"], row["ann_return_pre2020"]), fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.set_title("Candidate Risk vs Return (through 2019-12-31)")
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized return")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.grid(alpha=0.25)


def plot_candidate_recent_returns(candidate_screen: pd.DataFrame) -> None:
    valid = candidate_screen.dropna(subset=["return_1y_pre2020"]).copy()
    valid = valid.sort_values("return_1y_pre2020", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    if valid.empty:
        ax.text(0.5, 0.5, "No candidate 1-year return metrics available yet.", ha="center", va="center")
        ax.set_axis_off()
        return

    ax.bar(valid["candidate_ticker"], valid["return_1y_pre2020"])
    ax.set_title("Candidate 1-Year Return into the January 2020 Decision Date")
    ax.set_xlabel("Candidate ticker")
    ax.set_ylabel("1-year return")
    ax.tick_params(axis="x", rotation=45)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.grid(axis="y", alpha=0.25)


def plot_revised_static_weights(selection_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    valid = selection_df.dropna(subset=["static_weight"]).copy()
    if valid.empty:
        ax.text(0.5, 0.5, "Revised-static weights are not ready yet.", ha="center", va="center")
        ax.set_axis_off()
        return

    valid = valid.sort_values("static_weight", ascending=False)
    ax.bar(valid["ticker"], valid["static_weight"])
    ax.set_title("Revised Static Portfolio Weights")
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Static target weight")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)


def plot_legacy_static_vs_benchmark(compare_df: pd.DataFrame, benchmark_ticker: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    if compare_df.empty:
        ax.text(0.5, 0.5, "Static-vs-legacy backtest is not ready yet.", ha="center", va="center")
        ax.set_axis_off()
        return

    date_col = pd.to_datetime(compare_df["date"], errors="coerce")
    if "legacy_growth_of_1_oos" in compare_df.columns:
        ax.plot(date_col, compare_df["legacy_growth_of_1_oos"], label="Legacy Fund", linewidth=1.7)
    if "revised_static_growth_of_1" in compare_df.columns:
        ax.plot(date_col, compare_df["revised_static_growth_of_1"], label="Revised Static Fund", linewidth=1.7)
    if "benchmark_growth_of_1" in compare_df.columns:
        ax.plot(date_col, compare_df["benchmark_growth_of_1"], label=normalize_text(benchmark_ticker).upper(), linewidth=1.4)
    ax.set_title("Legacy Fund vs Revised Static Fund vs Benchmark")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()


def _constraint_float(constraints: dict[str, Any], key: str, default: float) -> float:
    value = pd.to_numeric(constraints.get(key), errors="coerce")
    return default if pd.isna(value) else float(value)


def _constraint_int(constraints: dict[str, Any], key: str, default: int) -> int:
    value = pd.to_numeric(constraints.get(key), errors="coerce")
    if pd.isna(value):
        return default
    try:
        return int(value)
    except Exception:
        return default


def _normalize_rule_name(active_rule: Any, optimizer_objective: Any) -> str:
    rule = normalize_text(active_rule).lower().replace("-", "_").replace(" ", "_")
    objective = normalize_text(optimizer_objective).lower().replace("-", "_").replace(" ", "_")

    if "equal" in rule:
        return "equal_weight"
    if "inverse" in rule or "vol" in rule:
        return "inverse_vol"
    if "momentum" in rule or "score" in rule:
        return "momentum"
    if "optim" in rule or "sharpe" in rule or "risk" in rule:
        if "min_vol" in objective or "minimum_vol" in objective:
            return "inverse_vol"
        return "sharpe_tilt"
    return "equal_weight"


def _project_weights_with_bounds(raw_weights: np.ndarray, lower: float, upper: float) -> np.ndarray:
    raw = np.asarray(raw_weights, dtype=float).copy()
    n = raw.size
    if n == 0:
        return raw

    lower = float(max(0.0, lower))
    upper = float(min(1.0, upper))
    if not np.isfinite(lower):
        lower = 0.0
    if not np.isfinite(upper):
        upper = 1.0
    if lower * n > 1.0 + 1e-9 or upper * n < 1.0 - 1e-9:
        lower, upper = 0.0, 1.0

    raw[~np.isfinite(raw)] = 0.0
    raw = np.maximum(raw, 0.0)
    if raw.sum() <= 0:
        raw = np.repeat(1.0 / n, n)
    else:
        raw = raw / raw.sum()

    low = float(raw.min() - upper)
    high = float(raw.max() - lower)
    lam = 0.0
    for _ in range(120):
        lam = (low + high) / 2.0
        projected = np.clip(raw - lam, lower, upper)
        total = projected.sum()
        if abs(total - 1.0) < 1e-10:
            break
        if total > 1.0:
            low = lam
        else:
            high = lam

    projected = np.clip(raw - lam, lower, upper)
    if abs(projected.sum() - 1.0) > 1e-8:
        for _ in range(120):
            diff = 1.0 - projected.sum()
            if abs(diff) < 1e-10:
                break
            if diff > 0:
                free = projected < upper - 1e-10
            else:
                free = projected > lower + 1e-10
            if not free.any():
                break
            projected[free] += diff / free.sum()
            projected = np.clip(projected, lower, upper)

    if projected.sum() <= 0:
        projected = np.repeat(1.0 / n, n)
    else:
        projected = projected / projected.sum()
    return projected


def _apply_turnover_cap(
    current_weights: np.ndarray,
    target_weights: np.ndarray,
    turnover_cap: float,
) -> tuple[np.ndarray, float, bool]:
    current = np.asarray(current_weights, dtype=float)
    target = np.asarray(target_weights, dtype=float)
    turnover = float(0.5 * np.abs(target - current).sum())

    if not np.isfinite(turnover_cap) or turnover_cap <= 0 or turnover <= turnover_cap + 1e-12:
        return target, turnover, False

    alpha = float(turnover_cap / turnover)
    capped = current + alpha * (target - current)
    if capped.sum() <= 0:
        capped = current.copy()
    else:
        capped = capped / capped.sum()

    capped_turnover = float(0.5 * np.abs(capped - current).sum())
    return capped, capped_turnover, True


def _compute_active_target_weights(
    history_returns: pd.DataFrame,
    static_weights: np.ndarray,
    active_rule: Any,
    optimizer_objective: Any,
    lower: float,
    upper: float,
) -> tuple[np.ndarray, str]:
    n_assets = len(static_weights)
    rule_used = _normalize_rule_name(active_rule, optimizer_objective)

    if history_returns.empty or history_returns.shape[0] < 21:
        return _project_weights_with_bounds(static_weights, lower, upper), "static_fallback"

    hist = history_returns.dropna(how="any").copy()
    if hist.shape[0] < 21:
        return _project_weights_with_bounds(static_weights, lower, upper), "static_fallback"

    if rule_used == "equal_weight":
        raw = np.repeat(1.0 / n_assets, n_assets)
        return _project_weights_with_bounds(raw, lower, upper), rule_used

    if rule_used == "inverse_vol":
        vol = hist.std(ddof=1).replace(0.0, np.nan)
        signal = (1.0 / vol).replace([np.inf, -np.inf], np.nan)
        if signal.dropna().empty:
            raw = static_weights.copy()
        else:
            signal = signal.fillna(signal.dropna().mean() if not signal.dropna().empty else 1.0)
            raw = static_weights * signal.values
        return _project_weights_with_bounds(raw, lower, upper), rule_used

    if rule_used == "momentum":
        momentum = (1.0 + hist).prod() - 1.0
        signal = momentum.clip(lower=0.0)
        if float(signal.sum()) <= 0:
            raw = static_weights.copy()
            note = "momentum_fallback_static"
        else:
            raw = static_weights * (1.0 + signal.values)
            note = rule_used
        return _project_weights_with_bounds(raw, lower, upper), note

    # default: simple Sharpe-style tilt
    mean_ret = hist.mean()
    vol = hist.std(ddof=1).replace(0.0, np.nan)
    signal = (mean_ret / vol).replace([np.inf, -np.inf], np.nan).clip(lower=0.0)
    if signal.dropna().empty or float(signal.fillna(0.0).sum()) <= 0:
        inv = (1.0 / vol).replace([np.inf, -np.inf], np.nan)
        if inv.dropna().empty:
            raw = static_weights.copy()
            note = "sharpe_fallback_static"
        else:
            inv = inv.fillna(inv.dropna().mean() if not inv.dropna().empty else 1.0)
            raw = static_weights * inv.values
            note = "sharpe_fallback_inverse_vol"
    else:
        signal = signal.fillna(0.0)
        raw = static_weights * (1.0 + signal.values)
        note = rule_used

    return _project_weights_with_bounds(raw, lower, upper), note


def construct_revised_active_fund(
    price_panel: pd.DataFrame,
    selection_df: pd.DataFrame,
    legacy_fund_daily: pd.DataFrame,
    static_daily: pd.DataFrame,
    benchmark_daily: pd.DataFrame,
    constraints: dict[str, Any],
    oos_start: pd.Timestamp,
    oos_end: pd.Timestamp,
) -> dict[str, Any]:
    if selection_df.empty:
        raise ValueError("No revised-static selection was provided for the active layer.")

    tickers = selection_df["ticker"].map(normalize_text).str.upper().tolist()
    missing = [ticker for ticker in tickers if ticker not in price_panel.columns]
    if missing:
        raise ValueError(
            f"Revised-active portfolio tickers are missing from the price panel: {', '.join(missing)}"
        )

    px = price_panel[tickers].copy().sort_index()
    eligible = px.loc[(px.index >= oos_start) & (px.index <= oos_end)]
    eligible = eligible.dropna(how="any")
    if eligible.empty:
        raise ValueError("Could not find a common first trade date for the revised-active portfolio.")

    first_trade_date = pd.Timestamp(eligible.index[0])
    active_px = eligible.copy()
    active_dates = active_px.index
    if active_dates.shape[0] < 2:
        raise ValueError("Not enough daily observations to build the revised-active portfolio.")

    legacy_df = _standardize_date_column(legacy_fund_daily)
    legacy_df["date"] = pd.to_datetime(legacy_df["date"], errors="coerce")
    legacy_df = legacy_df.dropna(subset=["date"]).set_index("date").sort_index()
    if first_trade_date not in legacy_df.index:
        raise ValueError("Legacy fund daily table does not contain the revised-active first trade date.")

    static_df = _standardize_date_column(static_daily)
    static_df["date"] = pd.to_datetime(static_df["date"], errors="coerce")
    static_df = static_df.dropna(subset=["date"]).set_index("date").sort_index()

    benchmark_df = _standardize_date_column(benchmark_daily)
    benchmark_df["date"] = pd.to_datetime(benchmark_df["date"], errors="coerce")
    benchmark_df = benchmark_df.dropna(subset=["date"]).set_index("date").sort_index()

    static_weights = pd.to_numeric(selection_df["static_weight"], errors="coerce").astype(float).values
    if np.isnan(static_weights).all():
        static_weights = np.repeat(1.0 / len(tickers), len(tickers))
    else:
        static_weights = np.nan_to_num(static_weights, nan=0.0)
        if static_weights.sum() <= 0:
            static_weights = np.repeat(1.0 / len(tickers), len(tickers))
        else:
            static_weights = static_weights / static_weights.sum()

    max_weight = _constraint_float(constraints, "max_weight", 1.0)
    min_weight = _constraint_float(constraints, "min_weight", 0.0)
    turnover_cap = _constraint_float(constraints, "turnover_cap", np.inf)
    tc_bps = _constraint_float(constraints, "transaction_cost_bps", 0.0)
    est_months = max(1, _constraint_int(constraints, "estimation_window_months", 36))
    lookback_days = max(21, est_months * 21)
    active_rule = constraints.get("active_rule", "equal_weight")
    optimizer_objective = constraints.get("optimizer_objective", "max_sharpe")

    starting_weights = _project_weights_with_bounds(static_weights, min_weight, max_weight)

    daily_returns = active_px.pct_change()
    monthly_month_end = active_px.groupby(active_px.index.to_period("M")).apply(lambda df: df.index[-1]).tolist()

    target_map: dict[pd.Timestamp, np.ndarray] = {}
    rebalance_plan_rows: list[dict[str, Any]] = []

    for decision_dt in monthly_month_end:
        loc = active_dates.get_loc(decision_dt)
        if isinstance(loc, slice):
            loc = loc.start
        if loc is None or loc + 1 >= len(active_dates):
            continue

        effective_date = pd.Timestamp(active_dates[loc + 1])
        history = daily_returns.loc[:decision_dt].dropna(how="any").tail(lookback_days)
        target_weights, rule_used = _compute_active_target_weights(
            history_returns=history,
            static_weights=starting_weights,
            active_rule=active_rule,
            optimizer_objective=optimizer_objective,
            lower=min_weight,
            upper=max_weight,
        )
        target_map[effective_date] = target_weights
        rebalance_plan_rows.append(
            {
                "decision_date": pd.Timestamp(decision_dt).strftime("%Y-%m-%d"),
                "effective_date": effective_date.strftime("%Y-%m-%d"),
                "rule_used": rule_used,
                "lookback_obs": int(history.shape[0]),
            }
        )

    current_weights = pd.Series(starting_weights, index=tickers, dtype=float)
    start_capital = float(legacy_df.loc[first_trade_date, "legacy_fund_value"])
    portfolio_value = start_capital

    daily_rows: list[dict[str, Any]] = []
    weights_rows: list[dict[str, Any]] = []
    rebalance_log_rows: list[dict[str, Any]] = []

    first_row = {
        "date": first_trade_date,
        "revised_active_value": portfolio_value,
        "revised_active_return": 0.0,
        "revised_active_growth_of_1": 1.0,
        "revised_active_drawdown": 0.0,
        "rebalance_turnover": 0.0,
        "transaction_cost_rate": 0.0,
    }
    daily_rows.append(first_row)
    weights_record = {
        "date": first_trade_date.strftime("%Y-%m-%d"),
        "rebalance_turnover": 0.0,
        "transaction_cost_rate": 0.0,
        "rule_used": "initial_static",
    }
    weights_record.update({ticker: float(weight) for ticker, weight in current_weights.items()})
    weights_rows.append(weights_record)

    turnover_events: list[float] = []

    for i in range(1, len(active_dates)):
        date = pd.Timestamp(active_dates[i])
        prev_close_value = portfolio_value
        turnover_applied = 0.0
        cost_rate = 0.0
        rule_used = ""

        if date in target_map:
            target_weights = target_map[date]
            capped_weights, turnover_applied, capped = _apply_turnover_cap(
                current_weights.values,
                target_weights,
                turnover_cap,
            )
            cost_rate = turnover_applied * tc_bps / 10000.0
            portfolio_value = prev_close_value * (1.0 - cost_rate)
            current_weights = pd.Series(capped_weights, index=tickers, dtype=float)
            turnover_events.append(turnover_applied)
            rule_used = next(
                (
                    row["rule_used"]
                    for row in rebalance_plan_rows
                    if row["effective_date"] == date.strftime("%Y-%m-%d")
                ),
                _normalize_rule_name(active_rule, optimizer_objective),
            )
            rebalance_log_rows.append(
                {
                    "decision_date": next(
                        (
                            row["decision_date"]
                            for row in rebalance_plan_rows
                            if row["effective_date"] == date.strftime("%Y-%m-%d")
                        ),
                        "",
                    ),
                    "effective_date": date.strftime("%Y-%m-%d"),
                    "rule_used": rule_used,
                    "turnover": turnover_applied,
                    "transaction_cost_rate": cost_rate,
                    "capped_by_turnover": bool(capped),
                }
            )
        else:
            portfolio_value = prev_close_value

        asset_ret = daily_returns.loc[date, tickers].astype(float)
        gross_port_ret = float((current_weights * asset_ret).sum())
        portfolio_value = portfolio_value * (1.0 + gross_port_ret)
        total_daily_return = portfolio_value / prev_close_value - 1.0

        denom = 1.0 + gross_port_ret
        if abs(denom) < 1e-12:
            end_weights = current_weights.copy()
        else:
            end_weights = current_weights * (1.0 + asset_ret)
            if end_weights.sum() <= 0:
                end_weights = current_weights.copy()
            else:
                end_weights = end_weights / end_weights.sum()
        current_weights = end_weights.astype(float)

        daily_rows.append(
            {
                "date": date,
                "revised_active_value": portfolio_value,
                "revised_active_return": total_daily_return,
                "revised_active_growth_of_1": np.nan,
                "revised_active_drawdown": np.nan,
                "rebalance_turnover": turnover_applied,
                "transaction_cost_rate": cost_rate,
            }
        )

        weight_row = {
            "date": date.strftime("%Y-%m-%d"),
            "rebalance_turnover": turnover_applied,
            "transaction_cost_rate": cost_rate,
            "rule_used": rule_used,
        }
        weight_row.update({ticker: float(weight) for ticker, weight in current_weights.items()})
        weights_rows.append(weight_row)

    active_daily = pd.DataFrame(daily_rows)
    active_daily["revised_active_growth_of_1"] = (1.0 + active_daily["revised_active_return"].astype(float)).cumprod()
    active_daily["revised_active_drawdown"] = compute_drawdown(active_daily["revised_active_growth_of_1"].astype(float))

    weights_daily = pd.DataFrame(weights_rows)
    latest_weights = current_weights.copy()
    latest_date = pd.Timestamp(active_dates[-1]).strftime("%Y-%m-%d")
    weights_snapshot = pd.DataFrame(
        {
            "ticker": latest_weights.index,
            "latest_active_weight": latest_weights.values,
            "snapshot_date": latest_date,
        }
    )

    legacy_compare = legacy_df.loc[first_trade_date:oos_end, ["legacy_growth_of_1", "legacy_drawdown"]].copy()
    legacy_compare = legacy_compare.rename(
        columns={
            "legacy_growth_of_1": "legacy_growth_of_1_oos",
            "legacy_drawdown": "legacy_drawdown_oos",
        }
    )

    compare = legacy_compare.join(
        static_df.loc[first_trade_date:oos_end, ["revised_static_growth_of_1", "revised_static_drawdown"]],
        how="inner",
    )
    compare = compare.join(
        active_daily.set_index("date")[["revised_active_growth_of_1", "revised_active_drawdown"]],
        how="inner",
    )
    if {"benchmark_growth_of_1", "benchmark_drawdown"}.issubset(benchmark_df.columns):
        compare = compare.join(
            benchmark_df.loc[first_trade_date:oos_end, ["benchmark_growth_of_1", "benchmark_drawdown"]],
            how="inner",
        )
    compare = compare.reset_index().rename(columns={"index": "date"})

    avg_turnover = float(np.mean(turnover_events)) if turnover_events else 0.0
    active_metrics = compute_performance_metrics(active_daily["revised_active_return"], turnover=avg_turnover)

    rebalance_log = pd.DataFrame(rebalance_log_rows)
    if rebalance_log.empty:
        rebalance_log = pd.DataFrame(
            columns=[
                "decision_date",
                "effective_date",
                "rule_used",
                "turnover",
                "transaction_cost_rate",
                "capped_by_turnover",
            ]
        )

    return {
        "first_trade_date": first_trade_date,
        "start_capital": start_capital,
        "active_daily": active_daily,
        "weights_daily": weights_daily,
        "weights_snapshot": weights_snapshot,
        "rebalance_log": rebalance_log,
        "compare": compare,
        "metrics": active_metrics,
        "avg_turnover": avg_turnover,
        "n_rebalances": int(len(turnover_events)),
        "active_rule_used": _normalize_rule_name(active_rule, optimizer_objective),
    }


def plot_revised_active_weights(weights_daily: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    if weights_daily.empty or "date" not in weights_daily.columns:
        ax.text(0.5, 0.5, "Revised-active weight evolution is not ready yet.", ha="center", va="center")
        ax.set_axis_off()
        return

    chart_df = _standardize_date_column(weights_daily)
    chart_df["date"] = pd.to_datetime(chart_df["date"], errors="coerce")
    chart_df = chart_df.dropna(subset=["date"]).set_index("date")
    ignore_cols = {"rebalance_turnover", "transaction_cost_rate", "rule_used"}
    weight_cols = [c for c in chart_df.columns if c not in ignore_cols]
    if not weight_cols:
        ax.text(0.5, 0.5, "No active weights were exported.", ha="center", va="center")
        ax.set_axis_off()
        return

    for col in weight_cols:
        ax.plot(chart_df.index, chart_df[col], linewidth=1.2, label=col)
    ax.set_title("Revised Active Fund Weight Evolution")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()


def plot_legacy_static_active_vs_benchmark(compare_df: pd.DataFrame, benchmark_ticker: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    if compare_df.empty:
        ax.text(0.5, 0.5, "Active-vs-static backtest is not ready yet.", ha="center", va="center")
        ax.set_axis_off()
        return

    date_col = pd.to_datetime(compare_df["date"], errors="coerce")
    if "legacy_growth_of_1_oos" in compare_df.columns:
        ax.plot(date_col, compare_df["legacy_growth_of_1_oos"], label="Legacy Fund", linewidth=1.7)
    if "revised_static_growth_of_1" in compare_df.columns:
        ax.plot(date_col, compare_df["revised_static_growth_of_1"], label="Revised Static Fund", linewidth=1.7)
    if "revised_active_growth_of_1" in compare_df.columns:
        ax.plot(date_col, compare_df["revised_active_growth_of_1"], label="Revised Active Fund", linewidth=1.7)
    if "benchmark_growth_of_1" in compare_df.columns:
        ax.plot(date_col, compare_df["benchmark_growth_of_1"], label=normalize_text(benchmark_ticker).upper(), linewidth=1.4)
    ax.set_title("Legacy vs Revised Static vs Revised Active vs Benchmark")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
