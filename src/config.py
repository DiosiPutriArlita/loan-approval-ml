from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

RAW_DATA_PATH = DATA_DIR / "loan_data.csv"
MODEL_PATH = MODEL_DIR / "loan_model.joblib"
METRICS_PATH = REPORT_DIR / "metrics.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SAMPLES = 2000

TARGET = "loan_status"
NUMERIC_FEATURES = [
    "age",
    "monthly_income",
    "loan_amount",
    "loan_term_months",
    "credit_score",
    "debt_to_income_ratio",
]
CATEGORICAL_FEATURES = ["employment_status", "credit_history", "home_ownership"]
