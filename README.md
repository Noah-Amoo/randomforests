# Loan Default Risk Demo

A simple educational project using React + TypeScript, Flask, and a
scikit-learn Random Forest. Card 1 provides the frontend shell and backend
health endpoint. Card 2 adds reproducible synthetic data generation.
Card 3 trains, evaluates, and saves the Random Forest. The prediction API
and form are not implemented yet.

**Educational demo only. Not for lending decisions.** This project uses
synthetic data; metrics on that data will not establish real-world accuracy.

## Prerequisites

- Node.js 22.12+ (or 20.19+) and npm, as required by Vite
- Python 3.12

## Install

Run these commands in PowerShell from the project root:

```powershell
python -m venv backend\.venv
.\backend\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
Set-Location frontend
npm install
```

The explicit Python executable avoids requiring virtual environment activation.
The virtual environment and frontend dependencies are excluded from version control.

## Run locally

In one terminal, from the project root:

```powershell
.\backend\.venv\Scripts\python.exe -m flask --app backend.app run --host localhost --port 5000
```

In another terminal, from the project root:

```powershell
Set-Location frontend
npm run dev -- --host localhost --port 5173 --strictPort
```

Open http://localhost:5173 for the frontend. The Flask development server is
for local use only. Stop either server with Ctrl+C.

### Health check

```powershell
Invoke-RestMethod http://localhost:5000/health
```

`GET /health` returns HTTP 200 with JSON:

```json
{"status": "ok"}
```

CORS currently permits the local frontend origin `http://localhost:5173`
for this endpoint only.

## Generate synthetic data (Card 2)

From the project root:

```powershell
.\backend\.venv\Scripts\python.exe backend\generate_data.py
```

Or, from `backend` with the virtual environment active:

```powershell
python generate_data.py
```

This writes 2,000 rows to `backend\data\synthetic_loans.csv`, regardless of
your current directory. Re-running overwrites that file with identical
contents using seed 42 in the same Python environment.

| Column | Meaning | Generated range |
| --- | --- | --- |
| `annual_income` | Annual income in USD | 20,000-150,000 |
| `loan_amount` | Requested loan amount in USD | 1,000-80,000 |
| `credit_score` | Illustrative credit score | 300-850, integer |
| `monthly_debt_payments` | Existing monthly debt payments in USD | 0-3,000 |
| `years_employed` | Years employed | 0-40, integer |
| `defaulted` | Synthetic outcome | 0 = no default, 1 = default |

Features are randomly sampled within these bounds; monetary amounts have
two decimal places. An illustrative logistic formula makes defaults more
likely with larger loan-to-income and debt-to-income ratios, lower credit
scores, and fewer years employed. A random draw creates each outcome, so
the labels are not a deterministic threshold rule.

These distributions and coefficients are made up, not fitted to customer
records. The data omits real-world constraints and correlations and does
not represent actual borrowers or calibrated default probabilities.

## Train and evaluate (Card 3)

Generate the CSV first, then run from the project root:

```powershell
.\backend\.venv\Scripts\python.exe backend\train_model.py
```

Or, from `backend` with the virtual environment active:

```powershell
python train_model.py
```

The script uses the five feature columns as `X` and `defaulted` as `y`.
It reserves 20% of the rows for testing, stratified by the target to
preserve class proportions, using seed 42. A Random Forest with 100 trees
and a minimum of five samples per leaf fits only the training rows.
There is no hyperparameter search or fitting on the test set.

The terminal reports these held-out test metrics, treating default (`1`)
as the positive class:

- **Precision:** fraction of predicted defaults that were actual defaults.
- **Recall:** fraction of actual defaults detected.
- **ROC-AUC:** class separation measured using default probabilities, not
  predicted labels.
- **Confusion matrix:** rows are actual `[0, 1]`, columns predicted `[0, 1]`;
  entries are `[[true negatives, false positives], [false negatives, true positives]]`.

The script saves `backend\models\random_forest.joblib`. This is a dictionary
containing `model` (the fitted classifier) and `feature_names` (the ordered
input columns). Paths are independent of the current directory. Re-running
training overwrites the artifact; it is excluded from version control.
Results are reproducible with the same data and dependency versions.

For later inference, load the dictionary with `joblib.load`, arrange input
columns using `artifact["feature_names"]`, and use `artifact["model"]`.
Only load trusted joblib files: loading can execute arbitrary code.
Use the same dependency versions for training and loading; retrain after
upgrading scikit-learn.

Metrics describe this synthetic dataset only, not real lending performance.
The estimated probabilities are not calibrated financial risk scores.

## Validate

Backend, from the project root:

```powershell
Set-Location backend
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Frontend, in another terminal from the project root:

```powershell
Set-Location frontend
npm run build
npm run lint
```

## Layout

- `frontend/`: Vite React TypeScript app
- `backend/app.py`: Flask application and health route
- `backend/requirements.txt`: API and machine learning dependencies
- `backend/generate_data.py`: fixed-seed synthetic dataset generator
- `backend/data/synthetic_loans.csv`: generated educational dataset
- `backend/train_model.py`: training, held-out evaluation, and model export
- `backend/models/random_forest.joblib`: generated model and feature order
- `backend/tests/`: health, data generation, and training tests
