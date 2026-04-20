# Capstone Starter Pack: Fund Management Dashboard Capstone

This starter pack is the release-ready foundation for **Handout 04: Fund Management Dashboard Capstone**.

The project is built around one case:
- You inherit a 10-stock equity fund that starts on **2010-01-01** with **$1,000,000**
- Your manager takeover date is **2020-01-01**
- You may use only information available through **2019-12-31** to redesign the fund
- You evaluate results out of sample from **2020-01-01** through **2025-12-31**

The intended workflow is:

**Excel inputs → Python pipeline → saved outputs → dashboard display**

## What is included

- `fund_manager_control.xlsx` — control workbook for inputs, candidate research, constraints, and summary outputs
- `run_pipeline.py` — main analytics pipeline
- `dashboard_app.py` — Streamlit dashboard
- `utils/` — helper modules for data, portfolio, factor, and risk logic
- `notebooks/capstone_report.ipynb` — narrative notebook shell
- `report/final_report_template.md` — formal report outline
- `slides/final_presentation_outline.md` — slide-by-slide presentation outline
- `MILESTONE_BUILD_GUIDE.md` — step-by-step build guide for Milestones 1, 2, and 3

## Expected folder structure

```text
Capstone_Starter_Pack_Release/
    dashboard_app.py
    run_pipeline.py
    README.md
    QUICKSTART.md
    MILESTONE_BUILD_GUIDE.md
    fund_manager_control.xlsx
    notebooks/
        capstone_report.ipynb
    utils/
        data_utils.py
        portfolio_utils.py
        factor_utils.py
        risk_utils.py
        dashboard_utils.py
    data/
        raw/
        clean/
    outputs/
        figures/
        tables/
        logs/
    report/
        final_report_template.md
    slides/
        final_presentation_outline.md
```

## Before you start

Use the **shared course virtual environment** created at the course root (for example `Desktop/BFIN491/.venv`), not a random environment from another folder.

Typical workflow:

### macOS / Linux
```bash
cd ~/Desktop/BFIN491
source .venv/bin/activate
cd Capstone_Starter_Pack_Release
```

### Windows PowerShell
```powershell
cd ~/Desktop/BFIN491
.\.venv\Scripts\activate
cd Capstone_Starter_Pack_Release
```

If you are missing packages, reinstall them **inside the shared course environment**.

Core packages used by the starter pack:
- pandas
- numpy
- yfinance
- matplotlib
- streamlit
- openpyxl

## First run: 5-step startup

### 1. Open the Excel workbook
Open `fund_manager_control.xlsx`.

At minimum, review these sheets:
- `Inputs`
- `InheritedFund`
- `Candidates`
- `Constraints`
- `Outputs`
- `Notes_Manifest`

### 2. Fill in the workbook carefully
Starter-pack expectations:
- the inherited fund is already prefilled
- candidate rows are for your **5–10 new ideas**
- the revised fund must contain **exactly 10 stocks**
- if you select final names but leave static weights blank, the starter pipeline may fall back to equal weights and warn you

### 3. Run the pipeline
```bash
python run_pipeline.py
```

Optional:
```bash
python run_pipeline.py --no-download
```
Use `--no-download` only if you already have a cached clean price panel in `data/clean/prices_adjclose_daily.csv`.

### 4. Launch the dashboard
```bash
streamlit run dashboard_app.py
```

### 5. Review generated outputs
Check:
- `outputs/tables/`
- `outputs/figures/`
- `fund_manager_control.xlsx` → `Outputs` sheet
- `fund_manager_control.xlsx` → `Notes_Manifest` sheet

## What the starter pack gives you

This starter pack already includes a working analytics foundation. You are **not** expected to rewrite the project from scratch.

The codebase already supports:
- price download and data preparation
- inherited-fund reconstruction
- candidate-screen exports
- revised static and revised active portfolio layers
- factor / regression outputs
- scenario / stress outputs
- dashboard structure
- notebook, report, and presentation scaffolds

This is intentional. The goal of the course is **not** to spend your time rebuilding easy plumbing from zero.

## What you must build or modify

You still need to do the real capstone work.

At minimum, your team must:
- research and justify **5–10 candidate stocks**
- decide which inherited holdings to **keep, drop, or replace**
- define the **revised 10-stock universe**
- set and defend **target weights**
- choose and justify an **active-management rule**
- validate the starter outputs instead of accepting them blindly
- improve or customize charts, tables, and dashboard pages where needed
- interpret factor / regression results in finance language
- interpret scenario / stress results in finance language
- write the formal report
- build the presentation deck
- defend the final recommendation clearly

### Important principle
This starter pack gives you a strong base, but it is **not** your final project unless you make it your own.

Leaving the starter logic untouched and simply rerunning it is **not enough**.

## Suggested student build order

Use this order to avoid chaos:

1. confirm the workbook is readable
2. run `python run_pipeline.py`
3. open the dashboard
4. review the inherited fund layer
5. fill in candidate research
6. finalize the revised 10-stock fund
7. test and refine the active rule
8. interpret factor and scenario outputs
9. improve the dashboard and workbook summaries
10. complete the notebook, report, and slide deck

For the full step-by-step plan by checkpoint, open:

`MILESTONE_BUILD_GUIDE.md`

## Main exported tables

You will usually see files like:
- `tbl_inputs.csv`
- `tbl_constraints.csv`
- `tbl_inherited_fund.csv`
- `tbl_candidates.csv`
- `tbl_portfolio_summary.csv`
- `tbl_performance_summary.csv`
- `tbl_price_quality_summary.csv`
- `tbl_legacy_fund_daily.csv`
- `tbl_candidate_screen.csv`
- `tbl_revised_static_daily.csv`
- `tbl_revised_active_daily.csv`
- `tbl_factor_capm_summary.csv`
- `tbl_scenario_monte_carlo_summary.csv`
- `tbl_project_manifest.csv`

## Main exported figures

You will usually see files like:
- `fig_inherited_fund_overview.png`
- `fig_inherited_drawdown.png`
- `fig_inherited_weights_snapshot.png`
- `fig_candidate_risk_return.png`
- `fig_candidate_recent_return.png`
- `fig_revised_static_weights.png`
- `fig_factor_alpha_beta.png`
- `fig_factor_rolling_beta.png`
- `fig_scenario_monte_carlo_distribution.png`
- `fig_scenario_stress_impacts.png`

## Common problems and fixes

### Problem: `ModuleNotFoundError`
Cause:
- wrong interpreter / wrong environment

Fix:
- activate the course `.venv`
- verify the interpreter path
- reinstall missing packages inside that environment

### Problem: workbook not found
Cause:
- you ran `python run_pipeline.py` from the wrong folder

Fix:
- change into the starter-pack folder first
- or pass a workbook path explicitly:
```bash
python run_pipeline.py --workbook fund_manager_control.xlsx
```

### Problem: `BRK.B` vs `BRK-B`
Cause:
- different platforms use different ticker naming

Fix:
- use `download_ticker` for API calls
- use `legacy_ticker` / display ticker for human-readable output

### Problem: revised static fund is “pending”
Most common causes:
- not exactly 10 selected final names
- blank or invalid target weights
- selected tickers do not have enough price history

### Problem: Streamlit launches but looks sparse
That usually means:
- the pipeline ran, but key workbook selections are still incomplete
- or the team has not yet customized the revised portfolio logic and interpretation

## AI / vibe-coding rules for this project

Use AI as a helper, not a replacement.

Good use:
- ask for one function at a time
- debug one error at a time
- verify every output with tables, plots, and saved files

Bad use:
- generate the entire project in one shot
- submit code you cannot explain
- trust outputs you did not check

## Milestone alignment

- **Milestone 1** — inherited fund reconstruction, project shell, dashboard wireframe
- **Milestone 2** — candidate analysis, revised static logic, first backtest
- **Milestone 3** — active layer, factor/scenario sections, final dashboard, report, and deck

Open `MILESTONE_BUILD_GUIDE.md` for the detailed build plan.

## Final student deliverables

The capstone requires:
- dashboard app
- Excel control workbook
- Python pipeline files
- utility modules
- one notebook that runs top-to-bottom
- saved figures
- saved tables
- `README.md`
- formal written report
- presentation deck

## Practical note for students

The easiest way to fall behind is to treat the starter pack like a black box.

The easiest way to succeed is to:
- run it early
- change one thing at a time
- verify each new result
- keep notes on decisions and assumptions
- translate output into finance reasoning
