# Developer Burnout Risk Classifier

**ECON 3916 — Statistical & Machine Learning for Economics · Final Project**
Ian Menachery · Northeastern University · Spring 2026

A multi-class Random Forest that classifies developers into **Low / Medium / High** burnout tiers from work patterns and self-reported stress, paired with a Streamlit app that wraps the model as a screening tool for engineering team leads.

- **Headline metric:** macro F1 = **0.991 ± 0.002** (5-fold cross-validation)
- **Data:** 7,000 developers (public Kaggle dataset, likely synthetic)
- **Stakeholder decision:** surface candidates for a wellness check-in before attrition, *not* performance review

> **Intended use:** human-in-the-loop screening aid only. Predictions must not drive performance reviews, compensation, hiring, firing, or any adverse employment decision. See [Caveats](#caveats-and-intended-use).

---

## Repository structure

```
Econ3916-Final-Project/
├── README.md                    ← you are here
├── requirements.txt                  ← Python dependencies
├── 3916-final-project-starter.ipynb  ← EDA, preprocessing, modeling, evaluation
├── data/
│   └── developer_burnout.csv    ← NOT checked into git (see Data acquisition)
├── model/
│   └── model.pkl                ← trained Random Forest (produced by notebook)
└── streamlit_app/
    ├── app.py                   ← Streamlit screening tool
    ├── run.sh                   ← Mac/Linux launcher
    ├── run.bat                  ← Windows launcher
    └── README.md                ← app-specific quick-start
```

---

## Prerequisites

- **Python 3.10 or newer** — check with `python --version` or `python3 --version`. If missing, install from [python.org](https://www.python.org/downloads/) (on macOS, check "Add Python to PATH").
- **git** — for cloning this repo
- **A Kaggle account** — needed to download the dataset (free signup)

---

## 1. Environment setup

Clone the repo and create an isolated virtual environment. This keeps dependencies pinned to what the notebook was developed against and avoids conflicts with other Python projects on your machine.

```bash
# Clone
git clone https://github.com/ian-menachery/Econ3916-Final-Project.git
cd Econ3916-Final-Project

# Create virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows (PowerShell: .venv\Scripts\Activate.ps1)

# Install dependencies
pip install -r requirements.txt
```

Pinned dependencies (`requirements.txt`):

```
streamlit>=1.31.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.4.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
jupyter>=1.0.0
```

> **Why pin scikit-learn?** Models serialized with `joblib` are version-sensitive. If you retrain the model with a newer scikit-learn than the Streamlit app's environment, loading `model.pkl` will throw a warning or fail. Keep the notebook and the app in the same environment.

---

## 2. Data acquisition

**Source:** [Developer Burnout Prediction Dataset (7,000 samples)](https://www.kaggle.com/datasets/asifxzaman/developer-burnout-prediction-dataset7000-samples) — Kaggle, accessed April 15, 2026.

You have two options. Pick whichever is less friction for you.

### Option A — Manual download (easiest)

1. Open the Kaggle dataset page linked above and click **Download**
2. Unzip the archive
3. Place the CSV at `data/developer_burnout.csv` (create the `data/` folder if needed)

### Option B — Kaggle API (scriptable)

1. Install the Kaggle CLI: `pip install kaggle`
2. Grab an API token from your Kaggle account: **Settings → API → Create New Token** → this downloads `kaggle.json`
3. Place it at `~/.kaggle/kaggle.json` (macOS/Linux) or `C:\Users\<you>\.kaggle\kaggle.json` (Windows), then `chmod 600 ~/.kaggle/kaggle.json` on Unix
4. From the repo root:

```bash
mkdir -p data
kaggle datasets download -d asifxzaman/developer-burnout-prediction-dataset7000-samples -p data --unzip
```

### Dataset schema

| Column | Type | Description |
|---|---|---|
| `age` | int | Developer age |
| `experience_years` | int | Years in industry |
| `daily_work_hours` | float | Self-reported daily hours |
| `sleep_hours` | float | Self-reported sleep per night |
| `caffeine_intake` | int | Servings per day |
| `bugs_per_day` | int | Bugs introduced or reported |
| `commits_per_day` | int | Git commits |
| `meetings_per_day` | int | Scheduled meetings |
| `screen_time` | float | Daily screen hours |
| `exercise_hours` | float | Daily exercise hours |
| `stress_level` | int (1–10) | Self-reported stress |
| `Burn Rate` | float (0.0–1.0) | Target: continuous burnout score |

The notebook converts the continuous `Burn Rate` into the three-tier target (`Low` / `Medium` / `High`) used for classification. Tier cutoffs are documented in the preprocessing section of the notebook.

---

## 3. Reproducing the analysis (run the notebook)

From the repo root with the venv activated:

```bash
jupyter notebook 3916-final-project-starter.ipynb
```

Run cells top to bottom. The notebook walks through:

1. **EDA** — distributions, correlation matrix, tier prevalence
2. **Preprocessing** — tier binning, train/test split, feature ordering
3. **Modeling** — baseline logistic regression → Random Forest with `class_weight="balanced"`
4. **Evaluation** — 5-fold cross-validated macro F1, normalized confusion matrix, feature importance
5. **Model export** — saves the trained Random Forest to `model/model.pkl` via `joblib.dump`

Expected runtime: **~2 minutes** on a modern laptop. No GPU needed.

**Seed:** the notebook fixes `random_state=42` for the split, the model, and the CV splitter, so your numbers should match the reported macro F1 to within rounding.

### Running in Colab instead

If you'd rather not set up a local environment:

1. Open the notebook in Colab via [this link](https://colab.research.google.com/github/ian-menachery/Econ3916-Final-Project/blob/main/3916-final-project-starter.ipynb)
2. Upload the Kaggle CSV to the Colab session (📁 icon → upload) or `!pip install kaggle` and use the API there
3. At the end, run `files.download('model.pkl')` to pull the trained model to your machine for the Streamlit app

---

## 4. Launching the Streamlit app locally

The app loads `model.pkl` and exposes the 11 features as sliders. It returns the predicted tier, class probabilities, a recommended action, and a low-margin warning when the top two tiers are within 15 percentage points.

### Prerequisite

Make sure `model.pkl` from step 3 is at `streamlit_app/model.pkl`. If you ran the notebook locally, copy it:

```bash
cp model/model.pkl streamlit_app/model.pkl
```

### Run it

```bash
cd streamlit_app
streamlit run app.py
```

Your browser should open to `http://localhost:8501` automatically. If not, open that URL manually.

**Shortcut launchers:** double-click `run.sh` (macOS/Linux) or `run.bat` (Windows) from the `streamlit_app/` folder — these handle venv creation and dependency install on first run.

To stop the app: `Ctrl+C` in the terminal.

### Deployed version

A hosted version is available at `[STREAMLIT CLOUD URL HERE]` — identical behavior, no setup required.

---

## Results summary

| Model | Macro F1 (5-fold CV) | Notes |
|---|---|---|
| Logistic Regression (baseline) | *see notebook* | Standardized features, multinomial |
| **Random Forest (selected)** | **0.991 ± 0.002** | `class_weight="balanced"`, 200 trees |

Feature importance (top 3):

1. `stress_level` — ~70% of total importance
2. `daily_work_hours` — ~8%
3. `screen_time` — ~6%

The concentration on `stress_level` is a real limitation: that feature correlates 0.60 / 0.55 / 0.49 with work hours, screen time, and bugs per day respectively, so the model is partly restating a self-report rather than predicting independent risk from behavior. The notebook's discussion section works through what this implies for deployment.

---

## Caveats and intended use

- **Synthetic-looking data.** The Kaggle source is likely synthetic. Real-world performance on any specific engineering org has not been validated.
- **Predictive ≠ causal.** Feature importance is a variance-decomposition signal inside the model, not a causal effect. Reducing caffeine will not "move" a developer's predicted tier in any meaningful real-world sense.
- **Tier boundaries are an analyst choice.** Low / Medium / High come from cutoffs applied to the continuous `Burn Rate`, not from ground-truth categories.
- **Human-in-the-loop only.** This tool is a screening aid for team leads and wellness coordinators. It must not be used for performance reviews, compensation, hiring, firing, or any other adverse employment decision.

---

## Troubleshooting

**`model.pkl not found`** — Run the notebook end-to-end (step 3) before launching the app, and confirm the file is at `streamlit_app/model.pkl`.

**`InconsistentVersionWarning` or model load fails** — The scikit-learn version used to save the model differs from the one used to load it. Either retrain in the current environment, or pin `scikit-learn==<your version>` in `requirements.txt` and reinstall.

**Port 8501 in use** — Another Streamlit app is running. Either close it or launch on a different port: `streamlit run app.py --server.port 8502`.

**Kaggle CLI: 403 Forbidden** — You haven't accepted the dataset's terms yet. Visit the dataset page in a browser once and click any download button to accept, then retry the CLI.

**Feature order mismatch / weird predictions** — The sliders in `app.py` must feed the model in the same column order the model was trained on. In the notebook, run `print(list(X.columns))` and confirm it matches `FEATURE_COLS` at the top of `app.py`.

---

## License and attribution

- **Dataset:** © original Kaggle author (asifxzaman). See dataset page for license.
- **Code:** MIT (or whichever you prefer — update this line).
- **Course:** ECON 3916, Northeastern University, Spring 2026.
