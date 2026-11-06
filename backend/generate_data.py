import math
from pathlib import Path
import random

import pandas as pd

ROW_COUNT = 2_000
SEED = 42
DATA_PATH = Path(__file__).resolve().parent / "data" / "synthetic_loans.csv"


def generate_dataset(seed: int = SEED) -> pd.DataFrame:
    rng = random.Random(seed)
    rows: list[dict[str, int | float]] = []

    for _ in range(ROW_COUNT):
        annual_income = round(rng.uniform(20_000, 150_000), 2)
        loan_amount = round(rng.uniform(1_000, 80_000), 2)
        credit_score = rng.randint(300, 850)
        monthly_debt_payments = round(rng.uniform(0, 3_000), 2)
        years_employed = rng.randint(0, 40)

        loan_to_income = loan_amount / annual_income
        debt_to_income = monthly_debt_payments / (annual_income / 12)
        # Illustrative coefficients, not estimated from real lending data.
        log_odds = (
            -2.0
            + 1.4 * loan_to_income
            + 2.0 * debt_to_income
            + (650 - credit_score) / 100
            - 0.04 * years_employed
        )
        probability = 1 / (1 + math.exp(-log_odds))
        defaulted = int(rng.random() < probability)

        rows.append(
            {
                "annual_income": annual_income,
                "loan_amount": loan_amount,
                "credit_score": credit_score,
                "monthly_debt_payments": monthly_debt_payments,
                "years_employed": years_employed,
                "defaulted": defaulted,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    dataset = generate_dataset()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(
        DATA_PATH, index=False, float_format="%.2f", lineterminator="\n"
    )
    defaults = int(dataset["defaulted"].sum())
    print(f"Saved {len(dataset)} synthetic rows to {DATA_PATH}")
    print(f"Default: {defaults}; no default: {len(dataset) - defaults}")
    print("Educational synthetic data only. Not for lending decisions.")


if __name__ == "__main__":
    main()
