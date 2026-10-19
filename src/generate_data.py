import numpy as np
import pandas as pd

from src.config import N_SAMPLES, RANDOM_STATE, RAW_DATA_PATH


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def generate_dataset(n_samples: int = N_SAMPLES, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 66, n_samples)
    employment_status = rng.choice(
        ["employed", "self_employed", "unemployed"], n_samples, p=[0.65, 0.25, 0.10]
    )
    credit_history = rng.choice(["good", "fair", "poor"], n_samples, p=[0.55, 0.30, 0.15])
    home_ownership = rng.choice(["own", "rent", "mortgage"], n_samples, p=[0.25, 0.45, 0.30])

    base_income = rng.lognormal(mean=15.9, sigma=0.45, size=n_samples) / 12
    income_factor = np.where(employment_status == "unemployed", 0.25, 1.0)
    monthly_income = (base_income * income_factor).round(-3)

    loan_term_months = rng.choice([12, 24, 36, 48, 60], n_samples)
    loan_amount = (monthly_income * rng.uniform(2, 14, n_samples)).round(-3)

    credit_base = {"good": 740, "fair": 640, "poor": 540}
    credit_score = np.array([credit_base[c] for c in credit_history]) + rng.normal(0, 35, n_samples)
    credit_score = np.clip(credit_score, 300, 850).round()

    monthly_installment = loan_amount / loan_term_months
    debt_to_income_ratio = np.clip(
        monthly_installment / monthly_income + rng.normal(0.12, 0.05, n_samples), 0.01, 1.5
    ).round(3)

    score = (
        -1.2
        + 0.012 * (credit_score - 650)
        - 4.5 * (debt_to_income_ratio - 0.3)
        + 0.00000025 * (monthly_income - 6_000_000)
        + np.where(employment_status == "unemployed", -2.0, 0.0)
        + np.where(employment_status == "employed", 0.4, 0.0)
        + np.where(credit_history == "good", 0.9, np.where(credit_history == "poor", -1.1, 0.0))
        + np.where(home_ownership == "own", 0.3, 0.0)
        + rng.normal(0, 0.6, n_samples)
    )
    approved = rng.random(n_samples) < _sigmoid(score + 1.2)

    df = pd.DataFrame(
        {
            "age": age,
            "monthly_income": monthly_income,
            "employment_status": employment_status,
            "credit_history": credit_history,
            "home_ownership": home_ownership,
            "loan_amount": loan_amount,
            "loan_term_months": loan_term_months,
            "credit_score": credit_score,
            "debt_to_income_ratio": debt_to_income_ratio,
            "loan_status": np.where(approved, "Approved", "Rejected"),
        }
    )

    for column, rate in [("monthly_income", 0.04), ("credit_score", 0.03), ("employment_status", 0.02)]:
        mask = rng.random(n_samples) < rate
        df.loc[mask, column] = np.nan

    duplicates = df.sample(25, random_state=seed)
    return pd.concat([df, duplicates], ignore_index=True)


def main() -> None:
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset()
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Dataset disimpan: {RAW_DATA_PATH} ({len(df)} baris)")


if __name__ == "__main__":
    main()
