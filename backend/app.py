from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted
from werkzeug.exceptions import BadRequest

# Resolve relative to this file so Flask can start from either project directory.
MODEL_PATH = Path(__file__).resolve().parent / "models" / "random_forest.joblib"
# Inclusive input limits match the synthetic training data, not loan approval rules.
FEATURE_BOUNDS = {
    "annual_income": (20_000, 150_000),
    "loan_amount": (1_000, 80_000),
    "credit_score": (300, 850),
    "monthly_debt_payments": (0, 3_000),
    "years_employed": (0, 40),
}


def create_app(model_path: Path = MODEL_PATH) -> Flask:
    app = Flask(__name__)
    # Allow the local React frontend to read responses from these routes.
    CORS(
        app,
        resources={
            r"/health": {"origins": "http://localhost:5173"},
            r"/predict": {"origins": "http://localhost:5173"},
        },
    )
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run train_model.py before starting Flask."
        )
    # Load once per application, not per request. Only load trusted joblib files.
    # The artifact contains both the fitted model and its ordered feature names.
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict):
        raise ValueError("Invalid model artifact: expected a dictionary. Run train_model.py.")
    model = artifact.get("model")
    feature_names = artifact.get("feature_names")
    if (
        not isinstance(model, RandomForestClassifier)
        or not isinstance(feature_names, list)
        or not all(isinstance(name, str) for name in feature_names)
        or len(feature_names) != len(FEATURE_BOUNDS)
        or set(feature_names) != set(FEATURE_BOUNDS)
    ):
        raise ValueError("Invalid model or feature names. Run train_model.py.")
    # Fail at startup if the model is untrained or its schema is incompatible.
    check_is_fitted(model)
    if (
        not hasattr(model, "feature_names_in_")
        or list(model.feature_names_in_) != feature_names
        or list(model.classes_) != [0, 1]
    ):
        raise ValueError("Model schema or classes do not match the demo. Run train_model.py.")
    # Find the probability column for label 1 (default); labels and indices differ.
    default_class_index = list(model.classes_).index(1)

    def invalid_request(message: str) -> tuple[Response, int]:
        # Log the validation error without recording the customer's input values.
        app.logger.warning("Prediction request rejected: %s", message)
        return jsonify(error=message), 400

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict")
    def predict() -> Response | tuple[Response, int]:
        if not request.is_json:
            return invalid_request("Content-Type must be application/json.")
        try:
            payload = request.get_json()
        except BadRequest:
            return invalid_request("Request body must contain valid JSON.")
        if not isinstance(payload, dict):
            return invalid_request("Request body must be a JSON object.")
        # Require exactly the five inputs; the target label must not be supplied.
        if set(payload) != set(FEATURE_BOUNDS):
            return invalid_request(
                "Provide exactly these fields: " + ", ".join(FEATURE_BOUNDS)
            )
        for name, (minimum, maximum) in FEATURE_BOUNDS.items():
            value = payload[name]
            # Python treats booleans as integers, so reject them explicitly.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return invalid_request(f"{name} must be a number.")
            # This also rejects NaN and infinity, which cannot satisfy these bounds.
            if not minimum <= value <= maximum:
                return invalid_request(
                    f"{name} must be between {minimum} and {maximum} (inclusive)."
                )
            if name in ("credit_score", "years_employed") and value % 1 != 0:
                return invalid_request(f"{name} must be a whole number.")

        # Build one customer row in training order, regardless of JSON key order.
        features = pd.DataFrame(
            [[payload[name] for name in feature_names]], columns=feature_names
        )
        prediction = model.predict(features)[0]  # First row's class label: 0 or 1.
        # Select the first row's default probability, not its no-default probability.
        probability = model.predict_proba(features)[0, default_class_index]
        # Convert scikit-learn's numeric outputs to JSON-compatible Python types.
        return jsonify(
            predicted_default=bool(prediction == 1),
            default_probability=float(probability),
        )

    return app


# The Flask CLI imports this application, triggering model loading at startup.
app = create_app()
