# Developer Burnout Risk Classifier

ECON 3916 Final Project — Streamlit screening tool for engineering team leads.

---

## Quick start (3 steps, ~2 minutes)

### 1. Get your `model.pkl` from Colab

In your Colab notebook, run this cell (if you haven't already):

```python
import joblib
joblib.dump(model_2, 'model.pkl')
```

Then download the file:
- **Colab**: click the 📁 folder icon in the left sidebar → right-click `model.pkl` → **Download**
- Move the downloaded `model.pkl` into this folder (the same folder as `app.py`)

### 2. Run the app

**Mac:**
- Double-click `run.sh`
- If it opens in a text editor instead, right-click it → **Open With** → **Terminal**
- If macOS blocks it with a security warning: open Terminal, `cd` into this folder, and run `bash run.sh`

**Windows:**
- Double-click `run.bat`

**First run only:** this will take 1–2 minutes to set up a Python environment and install dependencies. Every run after that starts in ~5 seconds.

### 3. That's it

Your browser should open automatically to `http://localhost:8501`. If not, open that URL manually in any browser.

To stop the app, press `Ctrl+C` in the terminal window.

---

## What's in this folder

| File | What it does |
|---|---|
| `app.py` | The Streamlit app |
| `requirements.txt` | Python packages needed |
| `run.sh` | Mac/Linux setup + launch script |
| `run.bat` | Windows setup + launch script |
| `model.pkl` | **You provide this** — trained Random Forest from your notebook |
| `README.md` | This file |

---

## Troubleshooting

**"Python is not installed"**
Install Python 3.10+ from https://www.python.org/downloads/. On macOS, make sure to check "Add Python to PATH" during install.

**"model.pkl not found"**
Re-download it from Colab and drop it into this folder (not into a subfolder).

**"Port 8501 already in use"**
Another Streamlit app is already running. Either close it, or edit `run.sh` / `run.bat` and change the last line to:
`streamlit run app.py --server.port 8502`

**App opens but says "Error loading model"**
Your `model.pkl` was saved with a different scikit-learn version than the one installed here. Easiest fix: in Colab, check your version with `import sklearn; print(sklearn.__version__)`, then edit `requirements.txt` to pin that exact version (e.g., `scikit-learn==1.4.2`) and delete the `.venv` folder to force reinstall.

**Feature order mismatch / weird predictions**
The sliders in `app.py` must match the column order your model was trained on. In Colab, run `print(list(X.columns))` and make sure the order matches the `FEATURE_COLS` list near the top of `app.py`.

---

## Deploying to Streamlit Cloud (for Canvas submission)

1. Create a new GitHub repo and upload `app.py`, `requirements.txt`, and `model.pkl`
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and sign in with GitHub
3. Click "New app" → select your repo → main file is `app.py`
4. Deploy. The permanent URL it gives you is what you submit on Canvas.

**Note:** `model.pkl` must be under 100 MB for GitHub. If it's larger, reduce `n_estimators` or `max_depth` on the Random Forest and re-save.
