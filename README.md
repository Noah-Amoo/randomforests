# Loan Default Risk Demo

A simple educational project using React + TypeScript, Flask, and a
scikit-learn Random Forest. Card 1 provides the frontend shell and backend
health endpoint. Data generation, training, and predictions are not implemented yet.

**Educational demo only. Not for lending decisions.** Later cards will use
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
- `backend/tests/`: health endpoint tests
