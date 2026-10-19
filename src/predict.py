import argparse

import joblib
import pandas as pd

from src.config import CATEGORICAL_FEATURES, MODEL_PATH, NUMERIC_FEATURES


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def predict_applicant(model, applicant: dict) -> dict:
    frame = pd.DataFrame([applicant])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    probability = float(model.predict_proba(frame)[0, 1])
    return {
        "decision": "Approved" if probability >= 0.5 else "Rejected",
        "approval_probability": round(probability, 4),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prediksi persetujuan pinjaman")
    parser.add_argument("--age", type=int, required=True)
    parser.add_argument("--monthly-income", type=float, required=True)
    parser.add_argument("--loan-amount", type=float, required=True)
    parser.add_argument("--loan-term-months", type=int, required=True)
    parser.add_argument("--credit-score", type=float, required=True)
    parser.add_argument("--debt-to-income-ratio", type=float, required=True)
    parser.add_argument(
        "--employment-status", choices=["employed", "self_employed", "unemployed"], required=True
    )
    parser.add_argument("--credit-history", choices=["good", "fair", "poor"], required=True)
    parser.add_argument("--home-ownership", choices=["own", "rent", "mortgage"], required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    applicant = {
        "age": args.age,
        "monthly_income": args.monthly_income,
        "loan_amount": args.loan_amount,
        "loan_term_months": args.loan_term_months,
        "credit_score": args.credit_score,
        "debt_to_income_ratio": args.debt_to_income_ratio,
        "employment_status": args.employment_status,
        "credit_history": args.credit_history,
        "home_ownership": args.home_ownership,
    }
    result = predict_applicant(load_model(), applicant)
    print(f"Keputusan             : {result['decision']}")
    print(f"Probabilitas disetujui: {result['approval_probability']:.2%}")


if __name__ == "__main__":
    main()
