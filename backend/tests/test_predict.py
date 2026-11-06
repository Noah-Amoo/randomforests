from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from app import FEATURE_BOUNDS, MODEL_PATH, app, create_app


class PredictionTests(unittest.TestCase):
    def setUp(self) -> None:
        app.config.update(TESTING=True)
        self.client = app.test_client()
        self.payload = {
            "annual_income": 60_000,
            "loan_amount": 15_000,
            "credit_score": 700,
            "monthly_debt_payments": 500,
            "years_employed": 5,
        }

    def test_prediction_matches_saved_model(self) -> None:
        response = self.client.post("/predict", json=self.payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/json")
        result = response.get_json()
        self.assertEqual(set(result), {"predicted_default", "default_probability"})
        self.assertIsInstance(result["predicted_default"], bool)
        self.assertIsInstance(result["default_probability"], float)
        self.assertTrue(0 <= result["default_probability"] <= 1)
        artifact = joblib.load(MODEL_PATH)
        features = pd.DataFrame([self.payload], columns=artifact["feature_names"])
        model = artifact["model"]
        self.assertEqual(result["predicted_default"], bool(model.predict(features)[0]))
        self.assertEqual(
            result["default_probability"],
            float(model.predict_proba(features)[0, list(model.classes_).index(1)]),
        )

    def test_request_field_order_does_not_change_prediction(self) -> None:
        first = self.client.post("/predict", json=self.payload).get_json()
        reordered = dict(reversed(list(self.payload.items())))
        self.assertEqual(
            first, self.client.post("/predict", json=reordered).get_json()
        )

    def test_invalid_fields(self) -> None:
        for payload in (
            {},
            {key: value for key, value in self.payload.items() if key != "credit_score"},
            {**self.payload, "defaulted": 1},
        ):
            with self.subTest(payload=payload):
                response = self.client.post("/predict", json=payload)
                self.assertEqual(response.status_code, 400)
                self.assertIn("fields", response.get_json()["error"])

    def test_invalid_numeric_values(self) -> None:
        for name, (minimum, maximum) in FEATURE_BOUNDS.items():
            for value in (
                None, True, False, "500", [], {}, minimum - 1, maximum + 1,
                float("nan"), float("inf"), -float("inf"), 10**400,
            ):
                with self.subTest(field=name, value=value):
                    response = self.client.post(
                        "/predict", json={**self.payload, name: value}
                    )
                    self.assertEqual(response.status_code, 400)
                    self.assertIn(name, response.get_json()["error"])
        for name in ("credit_score", "years_employed"):
            response = self.client.post(
                "/predict", json={**self.payload, name: self.payload[name] + 0.5}
            )
            self.assertEqual(response.status_code, 400)
            self.assertIn("whole number", response.get_json()["error"])

    def test_inclusive_boundaries(self) -> None:
        for index in (0, 1):
            payload = {name: bounds[index] for name, bounds in FEATURE_BOUNDS.items()}
            response = self.client.post("/predict", json=payload)
            self.assertEqual(response.status_code, 200)

    def test_invalid_json_bodies(self) -> None:
        for body in ('{"broken":', 'null', '[]', '"text"', '42', ''):
            with self.subTest(body=body):
                response = self.client.post(
                    "/predict", data=body, content_type="application/json"
                )
                self.assertEqual(response.status_code, 400)
                self.assertIn("error", response.get_json())
        response = self.client.post("/predict", data="not json")
        self.assertEqual(response.status_code, 400)
        self.assertIn("Content-Type", response.get_json()["error"])

    def test_cors_and_preflight(self) -> None:
        origin = "http://localhost:5173"
        response = self.client.post(
            "/predict", json=self.payload, headers={"Origin": origin}
        )
        self.assertEqual(response.headers["Access-Control-Allow-Origin"], origin)
        response = self.client.options(
            "/predict",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Access-Control-Allow-Origin"], origin)
        self.assertIn("POST", response.headers["Access-Control-Allow-Methods"])
        self.assertIn("content-type", response.headers["Access-Control-Allow-Headers"])
        response = self.client.post(
            "/predict", json={}, headers={"Origin": origin}
        )
        self.assertEqual(response.headers["Access-Control-Allow-Origin"], origin)
        response = self.client.post(
            "/predict", json=self.payload, headers={"Origin": "https://example.com"}
        )
        self.assertNotIn("Access-Control-Allow-Origin", response.headers)

    def test_model_loads_once_per_app_not_per_request(self) -> None:
        with patch("app.joblib.load", wraps=joblib.load) as load:
            client = create_app().test_client()
            client.post("/predict", json=self.payload)
            client.post("/predict", json=self.payload)
            self.assertEqual(load.call_count, 1)

    def test_missing_model_stops_startup(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "Run train_model.py"):
                create_app(Path(directory) / "missing.joblib")

    def test_invalid_artifact_stops_startup(self) -> None:
        for artifact in (
            [],
            {},
            {"model": "invalid", "feature_names": list(FEATURE_BOUNDS)},
            {"model": RandomForestClassifier(), "feature_names": ["wrong"]},
        ):
            with self.subTest(artifact=artifact):
                with patch("app.joblib.load", return_value=artifact):
                    with self.assertRaisesRegex(ValueError, "Run train_model.py"):
                        create_app()

    def test_mismatched_feature_order_stops_startup(self) -> None:
        artifact = joblib.load(MODEL_PATH)
        artifact["feature_names"] = list(reversed(artifact["feature_names"]))
        with patch("app.joblib.load", return_value=artifact):
            with self.assertRaisesRegex(ValueError, "schema"):
                create_app()


if __name__ == "__main__":
    unittest.main()
