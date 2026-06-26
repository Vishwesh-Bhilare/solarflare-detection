from __future__ import annotations

from pathlib import Path
from typing import List
import zipfile
import io

import numpy as np
import pandas as pd
from astropy.io import fits
from tqdm import tqdm


class HEL1OSReader:
    """
    Reads HEL1OS Level-1 observations.

    Expected structure:

    data/raw/hel1os/
        2024/
            06/
                23/
                    N00_0000/
                        HLS_....zip
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)

        if not self.root.exists():
            raise FileNotFoundError(self.root)

    def list_zip_files(self) -> List[Path]:
        """
        Find every HEL1OS observation ZIP.
        """

        return sorted(self.root.rglob("HLS_*_lev1_*.zip"))

    def read_fits_from_zip(self, zip_path: Path, fits_path: str) -> fits.HDUList:
        """
        Read a FITS file directly from a ZIP archive without extracting.
        
        Parameters
        ----------
        zip_path : Path
            Path to the ZIP file
        fits_path : str
            Path to the FITS file inside the ZIP (e.g., "cdte/lightcurve_cdte1.fits")
        
        Returns
        -------
        fits.HDUList
            The opened FITS file
        """
        with zipfile.ZipFile(zip_path, 'r') as z:
            # Find the correct file inside the ZIP
            # The ZIP structure might have a root folder, so we need to search
            for name in z.namelist():
                if name.endswith(fits_path):
                    with z.open(name) as f:
                        # Read the bytes and open with fits
                        fits_bytes = f.read()
                        return fits.open(io.BytesIO(fits_bytes))
            
            # If not found by endswith, try a more flexible search
            for name in z.namelist():
                if fits_path.replace("/", "\\") in name or fits_path in name:
                    with z.open(name) as f:
                        fits_bytes = f.read()
                        return fits.open(io.BytesIO(fits_bytes))
            
            raise FileNotFoundError(f"Could not find {fits_path} in {zip_path}")

    def hdu_to_dataframe(
        self,
        hdu,
        column_name: str,
    ) -> pd.DataFrame:
        """
        Convert one HDU into a DataFrame.
        """

        table = hdu.data

        # ISOT is already a chararray with string data
        # Convert to numpy array of strings first
        isot_data = np.array(table["ISOT"]).astype(str)
        
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(isot_data, utc=True),
            column_name: table["CTR"].astype(np.float32),
        })

        return df

    def read_lightcurve_from_zip(
        self,
        zip_path: Path,
    ) -> pd.DataFrame:
        """
        Read all HEL1OS energy bands directly from ZIP.
        """

        # Read the FITS file directly from ZIP
        hdul = self.read_fits_from_zip(zip_path, "lightcurve_cdte1.fits")

        try:
            band_5_20 = self.hdu_to_dataframe(
                hdul[1],
                "hard_5_20",
            )

            band_20_30 = self.hdu_to_dataframe(
                hdul[2],
                "hard_20_30",
            )

            band_30_40 = self.hdu_to_dataframe(
                hdul[3],
                "hard_30_40",
            )

            band_40_60 = self.hdu_to_dataframe(
                hdul[4],
                "hard_40_60",
            )

            band_total = self.hdu_to_dataframe(
                hdul[5],
                "hard_total",
            )

        finally:
            hdul.close()

        df = band_total

        df = df.merge(
            band_5_20,
            on="timestamp",
            how="outer",
        )

        df = df.merge(
            band_20_30,
            on="timestamp",
            how="outer",
        )

        df = df.merge(
            band_30_40,
            on="timestamp",
            how="outer",
        )

        df = df.merge(
            band_40_60,
            on="timestamp",
            how="outer",
        )

        df = df.sort_values("timestamp")

        # Fill NaN with 0 (missing counts effectively represent 0 counts)
        df = df.fillna(0)

        # Convert to float32 to save memory
        cols = [
            "hard_total",
            "hard_5_20",
            "hard_20_30",
            "hard_30_40",
            "hard_40_60",
        ]
        df[cols] = df[cols].astype(np.float32)

        return df.reset_index(drop=True)

    def read_observation(
        self,
        zip_file: Path,
    ) -> pd.DataFrame:

        df = self.read_lightcurve_from_zip(zip_file)

        date = zip_file.stem.split("_")[1]

        df["date"] = pd.to_datetime(date, format="%Y%m%d")

        df["observation"] = zip_file.stem

        return df

    def read_all(self) -> pd.DataFrame:

        files = self.list_zip_files()

        print(f"Reading {len(files)} HEL1OS observations...")

        frames = []

        for file in tqdm(files, desc="Reading HEL1OS"):

            try:
                frames.append(
                    self.read_observation(file)
                )

            except Exception as e:
                print(file.name, e)

        return pd.concat(
            frames,
            ignore_index=True,
        )

    def save_parquet(
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

    reader = HEL1OSReader("data/raw/hel1os")

    df = reader.read_all()

    reader.save_parquet(
        df,
        "data/processed/hel1os.parquet",
    )

    print()

    print(df.head())

    print()

    print(df.shape)

    print()

    print(df.columns)

    print()

    print(df.dtypes)

    print()

    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
