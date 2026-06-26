from __future__ import annotations

from pathlib import Path

import pandas as pd


class DataMerger:
    """
    Merge SoLEXS and HEL1OS time-series.

    Output:
        timestamp
        soft_counts
        hard_total
        hard_5_20
        hard_20_30
        hard_30_40
        hard_40_60
    """

    def __init__(
        self,
        solexs_file: str | Path,
        hel1os_file: str | Path,
    ):
        self.solexs_file = Path(solexs_file)
        self.hel1os_file = Path(hel1os_file)

    def load(self):

        print("Loading SoLEXS...")
        solexs = pd.read_parquet(self.solexs_file)

        print("Loading HEL1OS...")
        hel1os = pd.read_parquet(self.hel1os_file)

        return solexs, hel1os

    def merge(self) -> pd.DataFrame:

        solexs, hel1os = self.load()

        # keep only required columns
        solexxs = solexs[
            [
                "timestamp",
                "soft_counts",
                "observation",
                "date",
            ]
        ].copy()

        hel1os = hel1os[
            [
                "timestamp",
                "hard_total",
                "hard_5_20",
                "hard_20_30",
                "hard_30_40",
                "hard_40_60",
            ]
        ].copy()

        # round HEL1OS timestamps to nearest second
        hel1os["timestamp"] = hel1os["timestamp"].dt.round("1s")

        # average duplicate timestamps after rounding
        hel1os = (
            hel1os
            .groupby("timestamp", as_index=False)
            .mean()
        )

        merged = pd.merge(
            solexxs,
            hel1os,
            on="timestamp",
            how="inner",
        )

        merged = merged.sort_values("timestamp")

        return merged.reset_index(drop=True)

    def save(
        self,
        df: pd.DataFrame,
        output: str | Path,
    ):

        output = Path(output)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        df.to_parquet(
            output,
            index=False,
        )


if __name__ == "__main__":

    merger = DataMerger(
        "data/processed/solexs.parquet",
        "data/processed/hel1os.parquet",
    )

    df = merger.merge()

    merger.save(
        df,
        "data/processed/merged.parquet",
    )

    print()

    print(df.head())

    print()

    print(df.tail())

    print()

    print(df.shape)

    print()

    print(df.dtypes)

    print()

    print(df.isna().sum())
