# MILESTONE BUILD GUIDE

This guide tells you **what to build, in order**, for each milestone.

## Important mindset

You are **not** expected to rebuild the starter pack from scratch.

The starter pack already gives you a working technical base. Your job is to use that base to do the real capstone work:
- candidate research
- portfolio redesign
- active-rule design
- interpretation
- dashboard refinement
- report writing
- presentation

Work in **small verified steps**:
1. change something
2. rerun the pipeline
3. inspect the workbook, tables, figures, and dashboard
4. only then move to the next step

---

# Milestone 1
## Project Skeleton and Inherited Fund Audit

### Main goal
Get the project running cleanly and understand the inherited fund before you try to redesign it.

### What should already work from the starter pack
- workbook opens
- `python run_pipeline.py` runs
- `streamlit run dashboard_app.py` runs
- inherited-fund outputs are generated
- notebook / report / slide templates exist

### What your team needs to do

#### Step 1. Set up the project identity
In `Inputs`:
- enter your project title
- enter team name(s)
- review the benchmark and date settings
- make sure the case timeline is unchanged unless your instructor approved something different

#### Step 2. Run the starter pack once without major edits
Do one clean baseline run:
```bash
python run_pipeline.py
streamlit run dashboard_app.py
```

#### Step 3. Inspect the inherited fund carefully
Use the workbook, tables, figures, and dashboard to answer:
- which legacy names dominate by the decision date?
- which names lagged?
- how concentrated did the fund become?
- what are the biggest performance and drawdown risks?

#### Step 4. Start writing your inherited-fund audit
In your notebook and report draft, start these sections:
- project background
- data and methodology
- inherited fund audit

#### Step 5. Sketch the dashboard story
At Milestone 1, the dashboard does not need to be perfect.
It should already have a wireframe or early structure for:
- Executive Overview
- Inherited Fund Review
- Candidate Stock Research
- Revised Portfolio Construction
- Backtest Comparison
- Risk and Diagnostics
- Scenario / Stress Test
- Final Recommendation

#### Step 6. Pick a candidate-search direction
Do **not** pick final stocks yet.
Instead, decide the research direction:
- sector theme
- diversification idea
- factor story
- risk-reduction idea
- portfolio-improvement idea

### Milestone 1 deliverables
You should be ready to show:
- working folder structure
- working workbook
- working pipeline
- working dashboard shell
- inherited fund reconstruction
- inherited fund audit draft
- report outline
- presentation outline

### What not to do yet
- do not spend hours polishing the deck
- do not invent a fancy active rule before you know the candidate universe
- do not treat default starter outputs as your final answer

---

# Milestone 2
## Candidate Analysis and First Portfolio Version

### Main goal
Turn the project from an inherited-fund audit into a real redesign.

### What your team needs to do

#### Step 1. Fill in the Candidates sheet
Add **5–10 candidate stocks**.

For each one, complete:
- ticker
- company name
- sector/theme
- one-line thesis
- screening note
- add / reject / watchlist decision

#### Step 2. Research candidates with evidence
Use the starter outputs as a base, but go beyond them.

For each candidate or candidate group, build evidence such as:
- return and risk comparisons
- drawdown behavior
- diversification benefit
- recent return behavior
- technical / feature signals
- factor / regression logic if helpful

#### Step 3. Make keep / drop / add decisions
Using only information available through **2019-12-31**, decide:
- which inherited holdings to keep
- which inherited holdings to drop
- which candidate stocks to add

#### Step 4. Finalize the revised static fund
Your required revised fund must contain **exactly 10 stocks**.

In the workbook:
- mark the final selected names
- enter target weights
- check that the weights make finance sense
- rerun the pipeline

#### Step 5. Review the static backtest
After rerunning, inspect:
- Legacy Fund
- Revised Static Fund
- Benchmark

Ask:
- did the redesign improve return?
- did it improve risk-adjusted performance?
- did it improve concentration or drawdown behavior?
- what tradeoffs appeared?

#### Step 6. Draft the active rule
You do not need the final polished active strategy yet, but you should define the first version:
- equal weight
- inverse vol
- momentum
- sharpe tilt
- or another instructor-approved rule

Also review:
- rebalance frequency
- turnover cap
- transaction cost assumption
- estimation window

#### Step 7. Advance the communication pieces
By the end of Milestone 2, your report and deck should contain real content, not just empty headings.

### Milestone 2 deliverables
You should be ready to show:
- completed candidate research table
- revised 10-stock universe
- static portfolio logic
- first active-rule design
- first backtest comparison
- report draft with major sections started
- slide draft or storyboard
- optional extension choice, if any

### What not to do yet
- do not overpolish the dashboard before the finance logic is stable
- do not keep changing candidate names casually after you commit to the revised fund
- do not ignore weak results; explain them honestly

---

# Milestone 3
## Final Dashboard and Recommendation

### Main goal
Finish the full manager recommendation and make the product presentation-ready.

### What your team needs to do

#### Step 1. Finalize the active layer
Lock the active rule and confirm:
- monthly rebalancing
- fixed post-2020 universe
- constraints are sensible
- turnover is acceptable
- the rule is clearly explainable

#### Step 2. Interpret the factor / regression outputs
Use the starter factor outputs as a base, but write the finance story:
- what does alpha mean here?
- what does beta mean here?
- what does the rolling beta pattern suggest?
- how do the factor results help explain the portfolio?

#### Step 3. Interpret the scenario / stress outputs
Use the starter scenario outputs as a base, but explain:
- what is the worst vulnerability?
- how do static and active portfolios differ under stress?
- what should a fund manager worry about?

#### Step 4. Polish the dashboard
Make the dashboard presentation-ready:
- clean labels
- readable charts
- coherent page flow
- concise recommendation language
- no placeholder text left behind

#### Step 5. Clean the workbook
Before final submission:
- make sure the workbook is readable
- remove confusing notes
- keep the Outputs sheet clean
- keep Notes_Manifest useful as an audit trail

#### Step 6. Finish the notebook, report, and deck
You need all three:
- narrative notebook
- formal written report
- presentation deck

Make sure they all tell the **same story**:
- inherited problem
- redesign logic
- static vs active results
- factor / risk / scenario interpretation
- final recommendation
- limitations

#### Step 7. Rehearse the oral presentation
Practice answering questions like:
- why did you keep or drop these names?
- why is your active rule financially sensible?
- what is the biggest remaining risk?
- what should a non-technical audience learn from the dashboard?

### Milestone 3 deliverables
You should be ready to submit:
- full dashboard
- cleaned workbook
- final code
- final figures and tables
- final formal report
- final presentation deck

### Final quality check
Before submitting, confirm:
- the dashboard runs
- the pipeline runs
- the workbook looks clean
- the notebook runs top-to-bottom
- the report and deck are complete
- your team can explain the important code and finance logic

---

# Simplest success rule

If you feel stuck, come back to this order:

1. inherited fund
2. candidates
3. revised static fund
4. revised active fund
5. factor interpretation
6. scenario interpretation
7. dashboard polish
8. report
9. slides
