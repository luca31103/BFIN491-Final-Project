
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

EXPECTED_TABLES = {
    "inputs": "tbl_inputs.csv",
    "constraints": "tbl_constraints.csv",
    "inherited": "tbl_inherited_fund.csv",
    "candidates": "tbl_candidates.csv",
    "portfolio": "tbl_portfolio_summary.csv",
    "performance": "tbl_performance_summary.csv",
    "risk": "tbl_risk_notes.csv",
    "manifest": "tbl_project_manifest.csv",
    "candidate_screen": "tbl_candidate_screen.csv",
    "legacy_daily": "tbl_legacy_fund_daily.csv",
    "static_daily": "tbl_revised_static_daily.csv",
    "active_daily": "tbl_revised_active_daily.csv",
    "backtest_static": "tbl_backtest_legacy_static_benchmark.csv",
    "backtest_active": "tbl_backtest_legacy_static_active_benchmark.csv",
    "active_rebalance_log": "tbl_revised_active_rebalance_log.csv",
    "price_quality": "tbl_price_quality_summary.csv",
    "factor_capm": "tbl_factor_capm_summary.csv",
    "factor_rolling_beta": "tbl_factor_rolling_beta.csv",
    "scenario_mc_summary": "tbl_scenario_monte_carlo_summary.csv",
    "scenario_mc_draws": "tbl_scenario_monte_carlo_draws.csv",
    "scenario_stress": "tbl_scenario_stress_summary.csv",
}

PAGES = [
    "Executive Overview",
    "Inherited Fund Review",
    "Candidate Stock Research",
    "Revised Portfolio Construction",
    "Backtest Comparison",
    "Risk and Diagnostics",
    "Scenario / Stress Test",
    "Final Recommendation",
]

st.set_page_config(
    page_title="Fund Management Dashboard Capstone",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()


@st.cache_data(show_spinner=False)
def safe_read_csv(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_key_value_table(path: str) -> dict[str, str]:
    df = safe_read_csv(path)
    if df.empty or not {"Parameter", "Value"}.issubset(df.columns):
        return {}
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        key = normalize_text(row.get("Parameter"))
        if key:
            out[key] = normalize_text(row.get("Value"))
    return out


@st.cache_data(show_spinner=False)
def load_all_tables() -> dict[str, pd.DataFrame | dict[str, str]]:
    payload: dict[str, pd.DataFrame | dict[str, str]] = {}
    for key, filename in EXPECTED_TABLES.items():
        path = TABLES_DIR / filename
        if key in {"inputs", "constraints"}:
            payload[key] = load_key_value_table(str(path))
        else:
            payload[key] = safe_read_csv(str(path))
    return payload


def coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def display_saved_figures(patterns: Iterable[str], heading: str) -> None:
    files: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        for file_path in sorted(FIGURES_DIR.glob(pattern)):
            if file_path.name not in seen:
                files.append(file_path)
                seen.add(file_path.name)

    if not files:
        st.info(f"No saved figures found yet for **{heading}**.")
        return

    st.subheader(heading)
    for file_path in files:
        if file_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            st.image(str(file_path), caption=file_path.name)
        else:
            st.write(file_path.name)


def render_sidebar(inputs: dict[str, str], constraints: dict[str, str], manifest_df: pd.DataFrame) -> str:
    st.sidebar.title("Capstone Dashboard")
    st.sidebar.caption("Starter dashboard for the Fund Management Dashboard Capstone")

    default_page = inputs.get("dashboard_default_page", PAGES[0])
    default_index = PAGES.index(default_page) if default_page in PAGES else 0
    page = st.sidebar.radio("Navigate", PAGES, index=default_index)

    st.sidebar.markdown("---")
    st.sidebar.write(f"**Project:** {inputs.get('project_title', 'N/A')}")
    st.sidebar.write(f"**Team:** {inputs.get('team_name', 'N/A')}")
    st.sidebar.write(f"**Decision date:** {inputs.get('decision_date', 'N/A')}")
    st.sidebar.write(f"**Benchmark:** {inputs.get('market_benchmark', 'N/A')}")
    st.sidebar.write(f"**Active rule:** {constraints.get('active_rule', 'N/A')}")

    if not manifest_df.empty and "created_at" in manifest_df.columns:
        ts = pd.to_datetime(manifest_df["created_at"], errors="coerce").max()
        if pd.notna(ts):
            st.sidebar.write(f"**Last artifact export:** {ts.strftime('%Y-%m-%d %H:%M:%S')}")

    st.sidebar.markdown("---")
    st.sidebar.code(
        "python run_pipeline.py\nstreamlit run dashboard_app.py",
        language="bash",
    )
    return page


def render_overview(payload: dict[str, pd.DataFrame | dict[str, str]]) -> None:
    inputs = payload["inputs"]  # type: ignore[assignment]
    constraints = payload["constraints"]  # type: ignore[assignment]
    perf_df = coerce_numeric(payload["performance"], ["total_return", "ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "turnover"])  # type: ignore[arg-type]
    candidate_screen = coerce_numeric(payload["candidate_screen"], ["ann_return_pre2020", "ann_vol_pre2020", "sharpe_pre2020", "max_drawdown_pre2020", "return_1y_pre2020"])  # type: ignore[arg-type]
    portfolio_df = coerce_numeric(payload["portfolio"], ["static_weight", "latest_active_weight"])  # type: ignore[arg-type]
    manifest_df = payload["manifest"]  # type: ignore[assignment]
    risk_df = payload["risk"]  # type: ignore[assignment]

    st.title(inputs.get("project_title", "Fund Management Dashboard Capstone"))
    st.caption(
        "Dashboard-first, Excel-backed starter app. The dashboard and workbook should both read the same Python-generated outputs."
    )

    legacy_row = perf_df.loc[perf_df["portfolio"].eq("Legacy Fund")] if not perf_df.empty else pd.DataFrame()
    static_row = perf_df.loc[perf_df["portfolio"].eq("Revised Static Fund")] if not perf_df.empty else pd.DataFrame()
    active_row = perf_df.loc[perf_df["portfolio"].eq("Revised Active Fund")] if not perf_df.empty else pd.DataFrame()
    candidate_ok = int(candidate_screen.get("status", pd.Series(dtype=str)).eq("ok").sum()) if not candidate_screen.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Legacy holdings", "10")
    c2.metric("Candidate rows", str(len(candidate_screen)))
    c3.metric("Candidate metrics ready", str(candidate_ok))
    c4.metric("Artifacts exported", str(len(manifest_df)))

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("### Quick status")
        st.write(f"**Team:** {inputs.get('team_name', 'N/A')}")
        st.write(f"**Decision date:** {inputs.get('decision_date', 'N/A')}")
        st.write(f"**Out-of-sample window:** {inputs.get('oos_start', 'N/A')} → {inputs.get('oos_end', 'N/A')}")
        st.write(f"**Benchmark:** {inputs.get('market_benchmark', 'N/A')}")
        st.write(f"**Rebalance frequency:** {constraints.get('rebalance_frequency', 'N/A')}")

        if not active_row.empty and active_row["ann_return"].notna().any():
            st.success("Revised Active Fund is ready. The starter pack now supports the full Legacy vs Static vs Active comparison layer.")
        elif not static_row.empty and static_row["ann_return"].notna().any():
            st.success("Revised Static Fund is ready. The starter pack now supports candidate screening and the first redesigned-fund backtest.")
        else:
            st.info("Revised Static Fund is not ready yet. Select the final 10 stocks and provide target weights in the workbook, then rerun the pipeline.")

        if not candidate_screen.empty:
            best = candidate_screen.sort_values("sharpe_pre2020", ascending=False, na_position="last").head(1)
            if not best.empty:
                st.write(
                    f"**Top candidate by pre-2020 Sharpe:** {normalize_text(best.iloc[0].get('candidate_ticker'))}"
                )

    with right:
        st.markdown("### Portfolio snapshot")
        if not portfolio_df.empty and {"ticker", "static_weight"}.issubset(portfolio_df.columns):
            chart_df = portfolio_df[["ticker", "static_weight"]].dropna().set_index("ticker")
            if not chart_df.empty:
                st.bar_chart(chart_df)
            else:
                st.info("Static weights are not ready yet.")
        else:
            st.info("Portfolio summary has not been exported yet.")

    st.markdown("### Headline metrics")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    if not legacy_row.empty:
        m1.metric("Legacy ann. return", f"{float(legacy_row['ann_return'].iloc[0]):.1%}" if pd.notna(legacy_row['ann_return'].iloc[0]) else "N/A")
        m2.metric("Legacy Sharpe", f"{float(legacy_row['sharpe'].iloc[0]):.2f}" if pd.notna(legacy_row['sharpe'].iloc[0]) else "N/A")
    else:
        m1.metric("Legacy ann. return", "N/A")
        m2.metric("Legacy Sharpe", "N/A")

    if not static_row.empty and static_row["ann_return"].notna().any():
        m3.metric("Static ann. return", f"{float(static_row['ann_return'].iloc[0]):.1%}")
        m4.metric("Static Sharpe", f"{float(static_row['sharpe'].iloc[0]):.2f}")
    else:
        m3.metric("Static ann. return", "Pending")
        m4.metric("Static Sharpe", "Pending")

    if not active_row.empty and active_row["ann_return"].notna().any():
        m5.metric("Active ann. return", f"{float(active_row['ann_return'].iloc[0]):.1%}")
        m6.metric("Active Sharpe", f"{float(active_row['sharpe'].iloc[0]):.2f}")
    else:
        m5.metric("Active ann. return", "Pending")
        m6.metric("Active Sharpe", "Pending")

    with st.expander("Show exported core tables"):
        st.write("**Performance summary**")
        st.dataframe(perf_df)
        st.write("**Risk notes**")
        st.dataframe(risk_df)


def render_inherited_fund_review(payload: dict[str, pd.DataFrame | dict[str, str]]) -> None:
    inherited_df = coerce_numeric(payload["inherited"], ["initial_weight_2010", "target_weight_2020"])  # type: ignore[arg-type]
    legacy_daily = coerce_numeric(payload["legacy_daily"], ["legacy_fund_value", "legacy_fund_return", "legacy_growth_of_1", "legacy_drawdown"])  # type: ignore[arg-type]
    perf_df = coerce_numeric(payload["performance"], ["total_return", "ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "turnover"])  # type: ignore[arg-type]

    st.title("Inherited Fund Review")
    st.write(
        "This section audits the inherited 10-stock fund: composition, concentration, historical performance, and drawdowns before redesign."
    )

    if inherited_df.empty:
        st.warning("InheritedFund export is missing.")
        return

    cols = [c for c in ["legacy_ticker", "company_name", "sector", "initial_weight_2010", "decision_2020", "keep_in_revised", "target_weight_2020", "notes"] if c in inherited_df.columns]
    st.dataframe(inherited_df[cols])

    legacy_row = perf_df.loc[perf_df["portfolio"].eq("Legacy Fund")] if not perf_df.empty else pd.DataFrame()
    if not legacy_row.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ann. return", f"{float(legacy_row['ann_return'].iloc[0]):.1%}" if pd.notna(legacy_row['ann_return'].iloc[0]) else "N/A")
        c2.metric("Ann. vol", f"{float(legacy_row['ann_vol'].iloc[0]):.1%}" if pd.notna(legacy_row['ann_vol'].iloc[0]) else "N/A")
        c3.metric("Sharpe", f"{float(legacy_row['sharpe'].iloc[0]):.2f}" if pd.notna(legacy_row['sharpe'].iloc[0]) else "N/A")
        c4.metric("Max drawdown", f"{float(legacy_row['max_drawdown'].iloc[0]):.1%}" if pd.notna(legacy_row['max_drawdown'].iloc[0]) else "N/A")

    if not legacy_daily.empty and {"date", "legacy_fund_value"}.issubset(legacy_daily.columns):
        chart_df = legacy_daily.copy()
        chart_df["date"] = pd.to_datetime(chart_df["date"], errors="coerce")
        chart_df = chart_df.dropna(subset=["date"]).set_index("date")
        st.line_chart(chart_df[["legacy_fund_value"]])

    display_saved_figures(
        ["fig_inherited_*", "*legacy*benchmark*"],
        "Saved figures for inherited-fund diagnostics",
    )


def render_candidate_stock_research(payload: dict[str, pd.DataFrame | dict[str, str]]) -> None:
    candidate_screen = coerce_numeric(
        payload["candidate_screen"],  # type: ignore[arg-type]
        [
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
        ],
    )

    st.title("Candidate Stock Research")
    st.write(
        "This page is the candidate-screening layer. Metrics here use only information available through 2019-12-31."
    )

    if candidate_screen.empty:
        st.warning("No candidate-screen table has been exported yet. Fill in the Candidates sheet and rerun the pipeline.")
        return

    st.dataframe(candidate_screen)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Candidates exported", str(len(candidate_screen)))
    metric_cols[1].metric("Usable histories", str(int(candidate_screen.get("status", pd.Series(dtype=str)).eq("ok").sum())))
    metric_cols[2].metric(
        "Marked Add",
        str(int(candidate_screen.get("add_decision", pd.Series(dtype=str)).map(normalize_text).str.lower().eq("add").sum())),
    )
    metric_cols[3].metric(
        "Selected for final",
        str(int(candidate_screen.get("selected_for_final", pd.Series(dtype=str)).map(normalize_text).str.lower().eq("yes").sum())),
    )

    valid = candidate_screen.dropna(subset=["ann_return_pre2020", "ann_vol_pre2020"])
    if not valid.empty:
        st.markdown("### Risk-return snapshot")
        chart_df = valid[["candidate_ticker", "ann_return_pre2020"]].set_index("candidate_ticker")
        st.bar_chart(chart_df)

    display_saved_figures(
        ["fig_candidate_*"],
        "Saved figures for candidate research",
    )


def render_revised_portfolio_construction(payload: dict[str, pd.DataFrame | dict[str, str]]) -> None:
    portfolio_df = coerce_numeric(payload["portfolio"], ["static_weight", "latest_active_weight"])  # type: ignore[arg-type]
    constraints = payload["constraints"]  # type: ignore[assignment]

    st.title("Revised Portfolio Construction")
    st.write(
        "This page explains the revised 10-stock fund: final selected names, target weights, and the logic behind the first redesigned portfolio."
    )

    if portfolio_df.empty:
        st.warning("Portfolio summary has not been exported yet.")
        return

    st.dataframe(portfolio_df)

    col_left, col_right = st.columns(2)
    if {"ticker", "static_weight"}.issubset(portfolio_df.columns):
        weight_df = portfolio_df[["ticker", "static_weight"]].dropna().set_index("ticker")
        with col_left:
            st.markdown("### Static target weights")
            if not weight_df.empty:
                st.bar_chart(weight_df)
            else:
                st.info("Static weights are not available yet.")

    if {"ticker", "latest_active_weight"}.issubset(portfolio_df.columns):
        active_weight_df = portfolio_df[["ticker", "latest_active_weight"]].dropna().set_index("ticker")
        with col_right:
            st.markdown("### Latest active weights")
            if not active_weight_df.empty:
                st.bar_chart(active_weight_df)
            else:
                st.info("Latest active weights are not available yet.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Long only", constraints.get("long_only", "N/A"))
    c2.metric("Max weight", normalize_text(constraints.get("max_weight")))
    c3.metric("Turnover cap", normalize_text(constraints.get("turnover_cap")))
    c4.metric("Objective", constraints.get("optimizer_objective", "N/A"))

    display_saved_figures(
        ["fig_revised_static_weights*", "fig_revised_active_weights*"],
        "Saved figures for revised-portfolio construction",
    )


def render_backtest_comparison(payload: dict[str, pd.DataFrame | dict[str, str]]) -> None:
    perf_df = coerce_numeric(payload["performance"], ["total_return", "ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "turnover"])  # type: ignore[arg-type]
    backtest_active = coerce_numeric(payload["backtest_active"], ["legacy_growth_of_1_oos", "revised_static_growth_of_1", "revised_active_growth_of_1", "benchmark_growth_of_1"])  # type: ignore[arg-type]
    backtest_static = coerce_numeric(payload["backtest_static"], ["legacy_growth_of_1_oos", "revised_static_growth_of_1", "benchmark_growth_of_1"])  # type: ignore[arg-type]

    st.title("Backtest Comparison")
    st.write(
        "This page compares the Legacy Fund, Revised Static Fund, Revised Active Fund, and benchmark using the exported backtest tables."
    )

    if not perf_df.empty:
        st.dataframe(perf_df)

    chart_source = backtest_active if not backtest_active.empty and "revised_active_growth_of_1" in backtest_active.columns else backtest_static
    if not chart_source.empty and "date" in chart_source.columns:
        chart_df = chart_source.copy()
        chart_df["date"] = pd.to_datetime(chart_df["date"], errors="coerce")
        chart_df = chart_df.dropna(subset=["date"]).set_index("date")
        chart_cols = [
            c
            for c in [
                "legacy_growth_of_1_oos",
                "revised_static_growth_of_1",
                "revised_active_growth_of_1",
                "benchmark_growth_of_1",
            ]
            if c in chart_df.columns
        ]
        if chart_cols:
            st.line_chart(chart_df[chart_cols])
    else:
        st.info("Backtest comparison tables are not ready yet.")

    display_saved_figures(
        ["fig_legacy_static_active_benchmark*", "fig_legacy_static_benchmark*", "fig_backtest_legacy_vs_benchmark*"],
        "Saved figures for backtest comparison",
    )



def render_risk_and_diagnostics(payload: dict[str, pd.DataFrame | dict[str, str]]) -> None:
    risk_df = payload["risk"]  # type: ignore[assignment]
    price_quality = payload["price_quality"]  # type: ignore[assignment]
    rebalance_log = payload["active_rebalance_log"]  # type: ignore[assignment]
    factor_capm = coerce_numeric(
        payload["factor_capm"],
        ["n_obs", "alpha_daily", "alpha_ann", "beta", "alpha_tstat", "alpha_pvalue", "beta_tstat", "beta_pvalue", "r_squared", "corr_to_benchmark"],
    )  # type: ignore[arg-type]
    factor_rolling = coerce_numeric(payload["factor_rolling_beta"], ["rolling_beta", "window"])  # type: ignore[arg-type]

    st.title("Risk and Diagnostics")
    st.write(
        "This page combines workbook-style notes with exported diagnostics from the pipeline, including CAPM-style regressions and rolling beta diagnostics."
    )

    if isinstance(risk_df, pd.DataFrame) and not risk_df.empty:
        st.markdown("### Risk notes")
        st.dataframe(risk_df)
    else:
        st.info("Risk notes are not available yet.")

    if isinstance(factor_capm, pd.DataFrame) and not factor_capm.empty:
        st.markdown("### CAPM summary")
        st.dataframe(factor_capm)

    if isinstance(factor_rolling, pd.DataFrame) and not factor_rolling.empty:
        st.markdown("### Rolling beta table")
        st.dataframe(factor_rolling)

    if isinstance(price_quality, pd.DataFrame) and not price_quality.empty:
        st.markdown("### Price quality checks")
        st.dataframe(price_quality)

    if isinstance(rebalance_log, pd.DataFrame) and not rebalance_log.empty:
        st.markdown("### Active rebalancing log")
        st.dataframe(rebalance_log)

    display_saved_figures(
        [
            "fig_inherited_drawdown*",
            "fig_revised_active_weights*",
            "fig_factor_alpha_beta*",
            "fig_factor_rolling_beta*",
            "*risk*",
            "*diagnostic*",
        ],
        "Saved figures for risk and diagnostics",
    )


def render_scenario_and_stress(payload: dict[str, pd.DataFrame | dict[str, str]]) -> None:
    mc_summary = coerce_numeric(
        payload["scenario_mc_summary"],
        [
            "horizon_days",
            "n_sims",
            "mean_terminal_return",
            "median_terminal_return",
            "pct_05_terminal_return",
            "pct_01_terminal_return",
            "var_95",
            "cvar_95",
            "prob_loss",
        ],
    )  # type: ignore[arg-type]
    stress_df = coerce_numeric(
        payload["scenario_stress"],
        ["benchmark_return", "equal_weight_universe_return", "static_portfolio_impact", "active_portfolio_impact"],
    )  # type: ignore[arg-type]

    st.title("Scenario / Stress Test")
    st.write(
        "This page surfaces the starter pack's Monte Carlo distribution summary and stress-scenario table for the redesigned fund."
    )

    if isinstance(mc_summary, pd.DataFrame) and not mc_summary.empty:
        st.markdown("### Monte Carlo summary")
        st.dataframe(mc_summary)
    else:
        st.info("Monte Carlo summary is not available yet.")

    if isinstance(stress_df, pd.DataFrame) and not stress_df.empty:
        st.markdown("### Stress scenarios")
        st.dataframe(stress_df)
    else:
        st.info("Stress-scenario table is not available yet.")

    display_saved_figures(
        ["fig_scenario_monte_carlo_distribution*", "fig_scenario_stress_impacts*"],
        "Saved figures for scenario / stress testing",
    )


def render_final_recommendation(payload: dict[str, pd.DataFrame | dict[str, str]]) -> None:
    st.title("Scenario / Stress Test")
    st.write(
        "Scenario analysis is still pending in the starter pack. This page is reserved for Monte Carlo, scenario, or stress-test outputs in the next build."
    )
    st.info(
        "Next major extension: simulate or stress-test the redesigned fund and export scenario-risk tables/figures."
    )


def render_final_recommendation(payload: dict[str, pd.DataFrame | dict[str, str]]) -> None:
    perf_df = coerce_numeric(payload["performance"], ["ann_return", "ann_vol", "sharpe", "max_drawdown", "turnover"])  # type: ignore[arg-type]
    risk_df = payload["risk"]  # type: ignore[assignment]
    factor_capm = coerce_numeric(payload["factor_capm"], ["beta", "alpha_ann", "r_squared"])  # type: ignore[arg-type]
    mc_summary = coerce_numeric(payload["scenario_mc_summary"], ["var_95", "cvar_95", "prob_loss"])  # type: ignore[arg-type]

    st.title("Final Recommendation")
    st.write(
        "This page should end with a manager-style recommendation supported by the exported analytics."
    )

    legacy_row = perf_df.loc[perf_df["portfolio"].eq("Legacy Fund")] if not perf_df.empty else pd.DataFrame()
    static_row = perf_df.loc[perf_df["portfolio"].eq("Revised Static Fund")] if not perf_df.empty else pd.DataFrame()
    active_row = perf_df.loc[perf_df["portfolio"].eq("Revised Active Fund")] if not perf_df.empty else pd.DataFrame()

    if not active_row.empty and active_row["ann_return"].notna().any():
        legacy_ann = float(legacy_row["ann_return"].iloc[0]) if not legacy_row.empty and pd.notna(legacy_row["ann_return"].iloc[0]) else np.nan
        static_ann = float(static_row["ann_return"].iloc[0]) if not static_row.empty and pd.notna(static_row["ann_return"].iloc[0]) else np.nan
        active_ann = float(active_row["ann_return"].iloc[0])
        legacy_sharpe = float(legacy_row["sharpe"].iloc[0]) if not legacy_row.empty and pd.notna(legacy_row["sharpe"].iloc[0]) else np.nan
        static_sharpe = float(static_row["sharpe"].iloc[0]) if not static_row.empty and pd.notna(static_row["sharpe"].iloc[0]) else np.nan
        active_sharpe = float(active_row["sharpe"].iloc[0]) if pd.notna(active_row["sharpe"].iloc[0]) else np.nan
        active_turnover = float(active_row["turnover"].iloc[0]) if pd.notna(active_row["turnover"].iloc[0]) else np.nan

        st.success("Starter recommendation is available.")
        st.write("The starter pack can now compare the inherited fund directly against both the revised static and revised active designs.")
        if not np.isnan(legacy_ann) and not np.isnan(legacy_sharpe):
            st.write(f"- Legacy Fund annualized return: **{legacy_ann:.1%}**; Sharpe: **{legacy_sharpe:.2f}**")
        if not np.isnan(static_ann) and not np.isnan(static_sharpe):
            st.write(f"- Revised Static Fund annualized return: **{static_ann:.1%}**; Sharpe: **{static_sharpe:.2f}**")
        st.write(
            f"- Revised Active Fund annualized return: **{active_ann:.1%}**; Sharpe: **{active_sharpe:.2f}**; average monthly turnover: **{active_turnover:.2%}**"
            if not np.isnan(active_ann) and not np.isnan(active_sharpe) and not np.isnan(active_turnover)
            else "- Revised Active Fund metrics are available but not fully populated."
        )
        comp = {
            "Legacy Fund": legacy_ann,
            "Revised Static Fund": static_ann,
            "Revised Active Fund": active_ann,
        }
        comp = {k: v for k, v in comp.items() if not np.isnan(v)}
        if comp:
            best = max(comp.items(), key=lambda kv: kv[1])[0]
            st.write(f"**Current headline:** Based on the starter backtest, the strongest annualized-return story belongs to **{best}**.")
        st.info("This is still not the final capstone conclusion. Factor and scenario diagnostics should still be used to qualify the recommendation.")

        if isinstance(factor_capm, pd.DataFrame) and not factor_capm.empty:
            st.markdown("### Factor lens")
            for _, row in factor_capm.dropna(subset=["beta"], how="all").iterrows():
                label = normalize_text(row.get("portfolio"))
                beta = row.get("beta")
                alpha_ann = row.get("alpha_ann")
                r2 = row.get("r_squared")
                beta_txt = f"{float(beta):.2f}" if pd.notna(beta) else "N/A"
                alpha_txt = f"{float(alpha_ann):.2%}" if pd.notna(alpha_ann) else "N/A"
                r2_txt = f"{float(r2):.2f}" if pd.notna(r2) else "N/A"
                st.write(f"- {label}: beta **{beta_txt}**, annualized alpha **{alpha_txt}**, R² **{r2_txt}**.")

        if isinstance(mc_summary, pd.DataFrame) and not mc_summary.empty and "var_95" in mc_summary.columns:
            st.markdown("### Scenario lens")
            for _, row in mc_summary.iterrows():
                label = normalize_text(row.get("portfolio"))
                var95 = row.get("var_95")
                cvar95 = row.get("cvar_95")
                loss_prob = row.get("prob_loss")
                var_txt = f"{float(var95):.1%}" if pd.notna(var95) else "N/A"
                cvar_txt = f"{float(cvar95):.1%}" if pd.notna(cvar95) else "N/A"
                loss_txt = f"{float(loss_prob):.1%}" if pd.notna(loss_prob) else "N/A"
                st.write(f"- {label}: 95% VaR **{var_txt}**, 95% CVaR **{cvar_txt}**, probability of loss **{loss_txt}**.")
    elif not static_row.empty and static_row["ann_return"].notna().any():
        st.info("Revised Static Fund is ready, but the revised active layer is still pending.")
    else:
        st.warning(
            "A formal recommendation is still premature. The starter pack needs the final 10-stock selection and target weights before the redesigned portfolios can be evaluated."
        )

    if isinstance(risk_df, pd.DataFrame) and not risk_df.empty:
        with st.expander("Supporting notes"):
            st.dataframe(risk_df)


def main() -> None:
    payload = load_all_tables()
    inputs = payload["inputs"] if isinstance(payload["inputs"], dict) else {}
    constraints = payload["constraints"] if isinstance(payload["constraints"], dict) else {}
    manifest_df = payload["manifest"] if isinstance(payload["manifest"], pd.DataFrame) else pd.DataFrame()

    page = render_sidebar(inputs, constraints, manifest_df)

    if page == "Executive Overview":
        render_overview(payload)
    elif page == "Inherited Fund Review":
        render_inherited_fund_review(payload)
    elif page == "Candidate Stock Research":
        render_candidate_stock_research(payload)
    elif page == "Revised Portfolio Construction":
        render_revised_portfolio_construction(payload)
    elif page == "Backtest Comparison":
        render_backtest_comparison(payload)
    elif page == "Risk and Diagnostics":
        render_risk_and_diagnostics(payload)
    elif page == "Scenario / Stress Test":
        render_scenario_and_stress(payload)
    elif page == "Final Recommendation":
        render_final_recommendation(payload)


if __name__ == "__main__":
    main()
