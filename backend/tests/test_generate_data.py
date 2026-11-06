import unittest

import pandas as pd

from generate_data import ROW_COUNT, SEED, generate_dataset


class GenerateDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = generate_dataset()

    def test_shape_and_columns(self) -> None:
        self.assertEqual(self.dataset.shape, (ROW_COUNT, 6))
        self.assertEqual(
            list(self.dataset.columns),
            [
                "annual_income",
                "loan_amount",
                "credit_score",
                "monthly_debt_payments",
                "years_employed",
                "defaulted",
            ],
        )

    def test_valid_values(self) -> None:
        self.assertFalse(self.dataset.isna().any().any())
        bounds = {
            "annual_income": (20_000, 150_000),
            "loan_amount": (1_000, 80_000),
            "credit_score": (300, 850),
            "monthly_debt_payments": (0, 3_000),
            "years_employed": (0, 40),
        }
        for column, (minimum, maximum) in bounds.items():
            with self.subTest(column=column):
                self.assertTrue(pd.api.types.is_numeric_dtype(self.dataset[column]))
                self.assertTrue(
                    self.dataset[column].between(minimum, maximum).all()
                )
        for column in ("credit_score", "years_employed", "defaulted"):
            with self.subTest(column=column):
                self.assertTrue(pd.api.types.is_integer_dtype(self.dataset[column]))

    def test_both_outcomes_are_present(self) -> None:
        self.assertEqual(set(self.dataset["defaulted"]), {0, 1})

    def test_same_seed_reproduces_dataset(self) -> None:
        pd.testing.assert_frame_equal(self.dataset, generate_dataset(SEED))

    def test_different_seed_changes_dataset(self) -> None:
        self.assertFalse(self.dataset.equals(generate_dataset(SEED + 1)))

    def test_lower_credit_group_has_more_defaults(self) -> None:
        low_credit = self.dataset.loc[
            self.dataset["credit_score"] < 500, "defaulted"
        ]
        high_credit = self.dataset.loc[
            self.dataset["credit_score"] > 750, "defaulted"
        ]
        self.assertGreater(low_credit.mean(), high_credit.mean())


if __name__ == "__main__":
    unittest.main()
