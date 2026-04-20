# QUICKSTART: Capstone Starter Pack

## 1. Activate the shared course environment

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

## 2. Open the workbook

Open:

`fund_manager_control.xlsx`

Review these tabs:
- Inputs
- InheritedFund
- Candidates
- Constraints
- Outputs
- Notes_Manifest

## 3. Run the pipeline

```bash
python run_pipeline.py
```

Optional cached-data mode:
```bash
python run_pipeline.py --no-download
```

## 4. Launch the dashboard

```bash
streamlit run dashboard_app.py
```

## 5. Check the results

After a successful run, review:
- `outputs/tables/`
- `outputs/figures/`
- `Outputs` sheet in Excel
- `Notes_Manifest` sheet in Excel

## 6. Open the notebook shell

Open:

`notebooks/capstone_report.ipynb`

Use it as the narrative notebook for your report build.

## 7. Use the communication templates

- `report/final_report_template.md`
- `slides/final_presentation_outline.md`

## 8. Build by milestone, not all at once

Open:

`MILESTONE_BUILD_GUIDE.md`

That file tells you, step by step, what to build for:
- Milestone 1
- Milestone 2
- Milestone 3

## 9. If something breaks

Check these first:
- Are you using the correct `.venv`?
- Are you in the starter-pack folder?
- Does the workbook still have the required sheet names?
- Did you accidentally enter bad ticker symbols?
- Did the pipeline export files into `outputs/tables/`?

## 10. Final reminder

You are not expected to rewrite the starter pack from scratch.

You **are** expected to:
- research candidates carefully
- make real portfolio decisions
- validate and interpret outputs
- improve the dashboard
- complete the report and presentation
