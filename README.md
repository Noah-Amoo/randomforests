# Loan Default Risk Demo

A simple educational project using React + TypeScript, Flask, and a
scikit-learn Random Forest. Card 1 provides the frontend shell and backend
health endpoint. Card 2 adds reproducible synthetic data generation.
Training and predictions are not implemented yet.

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
- `backend/tests/`: health endpoint tests
