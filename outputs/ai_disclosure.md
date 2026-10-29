# AI Disclosure — Appendix

## Use of AI Tools in This Project

**Project:** Optimized Retail Inventory Management System
**Programme:** HCLTech Career Shaper Programme — AIML Domain (4-Week Individual Project)
**Author:** Priya Singh

---

### Summary

AI-assisted tools were used during the development of this project for the
purposes described below. All outputs were reviewed, tested, and verified by
the author before inclusion in the project deliverables.

---

### Scope of AI Assistance

| Area | AI Tool Used | Nature of Assistance |
|---|---|---|
| Code generation & debugging | AI coding assistant (Antigravity / Gemini) | Drafting Python scripts (`generate_dataset.py`, `src/eda.py`, `src/forecast_baseline.py`, `src/forecast_model.py`, `app.py`), fixing deprecation warnings, structuring sklearn Pipelines |
| Synthetic data design | AI coding assistant | Designing the demand-generation rules (seasonal multipliers, category ranges, weekend uplift) used in `generate_dataset.py` |
| Documentation | AI coding assistant | Drafting `README.md`, `data/data_dictionary.csv`, `outputs/data_cleaning_log.md`, and this disclosure |
| Code review & debugging | AI coding assistant | Identifying redundant code, deprecated API calls, and Unicode encoding issues on Windows |

---

### Limitations and Author Responsibility

- All synthetic demand assumptions (summer +20%, Diwali +40%, weekend +8%)
  are **illustrative only** and do not represent validated real-world retail
  patterns.
- The dataset combines 500 rows from a supplied source file with 4,512
  synthetically generated rows. Any conclusions drawn from this data must
  acknowledge its largely synthetic nature.
- The ML model metrics reported in `outputs/model_metrics.csv` reflect
  performance on this synthetic/combined dataset. Results should **not** be
  extrapolated to real retail environments without retraining on genuine
  operational data.
- The author reviewed, tested, and validated all AI-generated outputs. The
  intellectual responsibility for the project's design choices, assumptions,
  and interpretations rests with the author.

---

### Recommended Disclosure Statement (for Report/Presentation)

> "AI coding-assistant tools (specifically Google Antigravity / Gemini) were
> used to accelerate code drafting, debugging, and documentation writing in
> this project. The synthetic data generation rules, model selection rationale,
> and all analytical conclusions were designed and validated by the author.
> Results are on illustrative synthetic data and are not claims of real-world
> accuracy."

---

*This appendix was prepared in accordance with HCLTech Career Shaper Programme
guidelines on transparency in AI tool usage.*

