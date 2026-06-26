from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SOLEXS = Path("data/processed/solexs.parquet")
HEL1OS = Path("data/processed/hel1os.parquet")


def plot_day(day: str):
    """
    day example:
        "2024-06-23"
    """

    print("Loading data...")

    soft = pd.read_parquet(SOLEXS)
    hard = pd.read_parquet(HEL1OS)

    day = pd.Timestamp(day).date()

    soft = soft[
        soft["timestamp"].dt.date == day
    ].copy()

    hard = hard[
        hard["timestamp"].dt.date == day
    ].copy()

    if soft.empty:
        print("No SoLEXS data for this day.")
    else:
        print(f"SoLEXS samples : {len(soft):,}")

    if hard.empty:
        print("No HEL1OS data for this day.")
    else:
        print(f"HEL1OS samples : {len(hard):,}")

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 8),
        sharex=True,
    )

    if not soft.empty:
        axes[0].plot(
            soft["timestamp"],
            soft["soft_counts"],
            linewidth=0.7,
        )

    axes[0].set_title("SoLEXS")
    axes[0].set_ylabel("Counts/sec")
    axes[0].grid(True)

    if not hard.empty:
        axes[1].plot(
            hard["timestamp"],
            hard["hard_total"],
            linewidth=0.7,
            label="Total",
        )

        axes[1].plot(
            hard["timestamp"],
            hard["hard_5_20"],
            linewidth=0.5,
            alpha=0.7,
            label="5-20 keV",
        )

        axes[1].plot(
            hard["timestamp"],
            hard["hard_20_30"],
            linewidth=0.5,
            alpha=0.7,
            label="20-30 keV",
        )

        axes[1].plot(
            hard["timestamp"],
            hard["hard_30_40"],
            linewidth=0.5,
            alpha=0.7,
            label="30-40 keV",
        )

        axes[1].plot(
            hard["timestamp"],
            hard["hard_40_60"],
            linewidth=0.5,
            alpha=0.7,
            label="40-60 keV",
        )

    axes[1].set_title("HEL1OS")
    axes[1].set_ylabel("Counts/sec")
    axes[1].set_xlabel("Time (UTC)")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    plot_day("2024-04-23")
