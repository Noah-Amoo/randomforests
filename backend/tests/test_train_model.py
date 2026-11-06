from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from generate_data import SEED, generate_dataset
from train_model import FEATURE_NAMES, train_model


class TrainModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = generate_dataset()
        cls.artifact, cls.metrics, cls.matrix = train_model(cls.dataset)

    def test_feature_order_and_target_exclusion(self) -> None:
        self.assertEqual(self.artifact["feature_names"], FEATURE_NAMES)
        self.assertEqual(
            list(self.artifact["model"].feature_names_in_), FEATURE_NAMES
        )
        self.assertNotIn("defaulted", self.artifact["feature_names"])

    def test_training_uses_only_stratified_training_rows(self) -> None:
        features = self.dataset[FEATURE_NAMES]
        target = self.dataset["defaulted"]
        x_train, x_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=SEED, stratify=target
        )
        self.assertEqual((len(x_train), len(x_test)), (1600, 400))
        self.assertTrue(set(x_train.index).isdisjoint(x_test.index))
        self.assertAlmostEqual(y_train.mean(), y_test.mean(), delta=0.005)

        model = RandomForestClassifier(n_estimators=2, random_state=SEED)
        with patch("train_model.RandomForestClassifier", return_value=model):
            with patch.object(model, "fit", wraps=model.fit) as fit:
                train_model(self.dataset)
        actual_x, actual_y = fit.call_args.args
        pd.testing.assert_frame_equal(actual_x, x_train)
        pd.testing.assert_series_equal(actual_y, y_train)

    def test_metrics_use_held_out_predictions_and_probabilities(self) -> None:
        _, x_test, _, y_test = train_test_split(
            self.dataset[FEATURE_NAMES],
            self.dataset["defaulted"],
            test_size=0.2,
            random_state=SEED,
            stratify=self.dataset["defaulted"],
        )
        model = self.artifact["model"]
        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test)[:, list(model.classes_).index(1)]
        self.assertEqual(
            self.metrics,
            {
                "precision": precision_score(y_test, predictions),
                "recall": recall_score(y_test, predictions),
                "roc_auc": roc_auc_score(y_test, probabilities),
            },
        )
        self.assertEqual(
            self.matrix, confusion_matrix(y_test, predictions, labels=[0, 1]).tolist()
        )
        self.assertEqual(sum(sum(row) for row in self.matrix), 400)
        for value in self.metrics.values():
            self.assertTrue(0 <= value <= 1)

    def test_saved_artifact_round_trip(self) -> None:
        sample = self.dataset[self.artifact["feature_names"]].head(10)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            joblib.dump(self.artifact, path)
            loaded = joblib.load(path)
        self.assertEqual(loaded["feature_names"], FEATURE_NAMES)
        self.assertEqual(
            loaded["model"].predict(sample).tolist(),
            self.artifact["model"].predict(sample).tolist(),
        )
        probabilities = loaded["model"].predict_proba(sample)
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        self.assertEqual(
            probabilities.tolist(),
            self.artifact["model"].predict_proba(sample).tolist(),
        )

    def test_training_is_reproducible(self) -> None:
        artifact, metrics, matrix = train_model(self.dataset)
        sample = self.dataset[FEATURE_NAMES].head(10)
        self.assertEqual(metrics, self.metrics)
        self.assertEqual(matrix, self.matrix)
        self.assertEqual(
            artifact["model"].predict_proba(sample).tolist(),
            self.artifact["model"].predict_proba(sample).tolist(),
        )

    def test_reordered_csv_columns_preserve_feature_order(self) -> None:
        reordered = self.dataset[list(reversed(self.dataset.columns))]
        artifact, metrics, matrix = train_model(reordered)
        self.assertEqual(list(artifact["model"].feature_names_in_), FEATURE_NAMES)
        self.assertEqual(metrics, self.metrics)
        self.assertEqual(matrix, self.matrix)

    def test_invalid_datasets_raise_clear_errors(self) -> None:
        missing = self.dataset.drop(columns=["credit_score"])
        extra = self.dataset.assign(unexpected=1)
        nonnumeric = self.dataset.assign(annual_income="invalid")
        missing_value = self.dataset.copy()
        missing_value.loc[0, "annual_income"] = float("nan")
        infinite = self.dataset.copy()
        infinite.loc[0, "loan_amount"] = float("inf")
        invalid_label = self.dataset.copy()
        invalid_label.loc[0, "defaulted"] = 2
        single_class = self.dataset.assign(defaulted=0)
        rare_class = self.dataset.assign(defaulted=0)
        rare_class.loc[0, "defaulted"] = 1
        for dataset, message in (
            (missing, "exactly these columns"),
            (extra, "exactly these columns"),
            (nonnumeric, "numeric"),
            (missing_value, "missing or infinite"),
            (infinite, "missing or infinite"),
            (invalid_label, "both 0 and 1"),
            (single_class, "both 0 and 1"),
            (rare_class, "at least two rows"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    train_model(dataset)


if __name__ == "__main__":
    unittest.main()
