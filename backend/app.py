from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/health": {"origins": "http://localhost:5173"}})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
