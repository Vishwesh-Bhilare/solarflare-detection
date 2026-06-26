from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from astropy.io import fits
from tqdm import tqdm


class SoLEXSReader:
    """
    Reads SoLEXS Level-1 light curve files.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)

        if not self.root.exists():
            raise FileNotFoundError(self.root)

    def list_observations(self) -> List[Path]:
        return sorted(
            p for p in self.root.iterdir()
            if p.is_dir() and p.name.startswith("AL1_SLX")
        )

    def find_lightcurve(self, observation: Path) -> Path | None:
        """
        Prefer SDD2 over SDD1.

        Accept both:
            *.lc
            *.lc.gz
        """

        search_order = ["SDD2", "SDD1"]

        for detector in search_order:
            folder = observation / detector

            if not folder.exists():
                continue

            # uncompressed
            files = sorted(folder.glob("*.lc"))
            if files:
                return files[0]

            # compressed
            files = sorted(folder.glob("*.lc.gz"))
            if files:
                return files[0]

        return None

    def list_lightcurves(self) -> List[Path]:
        lightcurves = []

        for obs in self.list_observations():
            lc = self.find_lightcurve(obs)

            if lc is not None:
                lightcurves.append(lc)

        return lightcurves

    def read_lightcurve(self, file: Path) -> pd.DataFrame:
        """
        Read a SoLEXS light curve (.lc or .lc.gz)
        and return a standardized DataFrame.
        """

        with fits.open(file) as hdul:

            table = hdul["RATE"].data

            df = pd.DataFrame({
                "timestamp": pd.to_datetime(
                    table["TIME"],
                    unit="s",
                    utc=True,
                ),
                "soft_counts": table["COUNTS"].astype(np.float64),
            })

        df = df.dropna(subset=["soft_counts"])

        return df.reset_index(drop=True)

    def read_day(self, observation: Path) -> pd.DataFrame:
        """
        Read a single observation directory.
        """

        lc = self.find_lightcurve(observation)

        if lc is None:
            raise FileNotFoundError(f"No lightcurve found in {observation}")

        detector = lc.parent.name

        df = self.read_lightcurve(lc)

        date = observation.name.split("_")[3]

        df["detector"] = detector
        df["date"] = pd.to_datetime(date, format="%Y%m%d")
        df["observation"] = observation.name

        return df

    def read_all(self) -> pd.DataFrame:
        """
        Read every available SoLEXS observation.
        """

        frames = []

        observations = self.list_observations()

        print(f"Reading {len(observations)} observations...")

        for obs in tqdm(observations, desc="Reading SoLEXS"):
            try:
                frames.append(self.read_day(obs))
            except Exception as e:
                print(f"Skipping {obs.name}: {e}")

        return pd.concat(frames, ignore_index=True)

    def save_parquet(
        self,
        df: pd.DataFrame,
        output: str | Path,
    ) -> None:

        output = Path(output)

        output.parent.mkdir(parents=True, exist_ok=True)

        df.to_parquet(output, index=False)


if __name__ == "__main__":

    reader = SoLEXSReader("data/raw/solexs")

    df = reader.read_all()

    reader.save_parquet(
        df,
        "data/processed/solexs.parquet",
    )

    print(df.head())

    print()

    print(df.shape)
