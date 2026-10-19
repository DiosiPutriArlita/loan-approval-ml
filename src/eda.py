import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURE_DIR, RAW_DATA_PATH, TARGET
from src.preprocessing import load_data


def summarize(df: pd.DataFrame) -> None:
    missing = df.isna().sum()
    print("Dimensi data         :", df.shape)
    print("Baris duplikat       :", int(df.duplicated().sum()))
    print("\nMissing value per kolom:")
    print(missing[missing > 0].to_string())
    print("\nDistribusi target:")
    print(df[TARGET].value_counts(normalize=True).round(3).to_string())


def plot_target_distribution(df: pd.DataFrame) -> None:
    ax = df[TARGET].value_counts().plot(kind="bar", color=["#2a9d8f", "#e76f51"], rot=0)
    ax.set_title("Distribusi Status Pinjaman")
    ax.set_xlabel("")
    ax.set_ylabel("Jumlah")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "target_distribution.png", dpi=150)
    plt.close()


def plot_approval_by_category(df: pd.DataFrame, column: str) -> None:
    rate = (
        df.assign(approved=(df[TARGET] == "Approved").astype(int))
        .groupby(column)["approved"]
        .mean()
        .sort_values()
    )
    ax = rate.plot(kind="barh", color="#264653")
    ax.set_title(f"Tingkat Persetujuan berdasarkan {column}")
    ax.set_xlabel("Proporsi disetujui")
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"approval_by_{column}.png", dpi=150)
    plt.close()


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data(RAW_DATA_PATH)
    summarize(df)
    plot_target_distribution(df)
    plot_approval_by_category(df, "credit_history")
    plot_approval_by_category(df, "employment_status")
    print(f"\nGrafik EDA disimpan di: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
