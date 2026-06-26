from pathlib import Path

import pandas as pd
import numpy as np


DATA = Path("data/processed/merged.parquet")


class FlareDetector:

    def __init__(self, window=300, sigma=2.5):
        """
        Parameters
        ----------
        window : int
            Rolling background window in seconds.

        sigma : float
            Threshold above background.
        """

        self.window = window
        self.sigma = sigma

    def load(self):

        print("Loading merged dataset...")

        return pd.read_parquet(DATA)

    def compute_background(self, df):

        background = (
            df["soft_counts"]
            .rolling(
                self.window,
                center=True,
                min_periods=1,
            )
            .median()
        )

        noise = (
            (df["soft_counts"] - background)
            .rolling(
                self.window,
                center=True,
                min_periods=1,
            )
            .std()
        )

        df["background"] = background
        df["noise"] = noise.fillna(0)

        return df

    def detect(self, df):

        threshold = (
            df["background"]
            + self.sigma * df["noise"]
        )

        df["flare"] = df["soft_counts"] > threshold

        return df

    def build_events(
        self,
        df,
        min_duration=60,
        merge_gap=120,
    ):

        flare = df["flare"].values

        events = []

        start = None

        for i, value in enumerate(flare):

            if value and start is None:
                start = i

            elif not value and start is not None:

                end = i - 1

                duration = end - start + 1

                if duration >= min_duration:

                    events.append([start, end])

                start = None

        # merge nearby events

        merged = []

        for event in events:

            if not merged:
                merged.append(event)
                continue

            previous = merged[-1]

            gap = event[0] - previous[1]

            if gap <= merge_gap:

                previous[1] = event[1]

            else:

                merged.append(event)

        catalogue = []

        for start, end in merged:

            block = df.iloc[start:end + 1]

            peak = block["soft_counts"].idxmax()

            row = df.loc[peak]

            catalogue.append(
                {
                    "start": block.iloc[0]["timestamp"],
                    "peak": row["timestamp"],
                    "end": block.iloc[-1]["timestamp"],
                    "duration_sec": len(block),
                    "peak_soft": row["soft_counts"],
                    "peak_hard": row["hard_total"],
                }
            )

        return pd.DataFrame(catalogue)


if __name__ == "__main__":

    detector = FlareDetector()

    df = detector.load()

    df = detector.compute_background(df)

    df = detector.detect(df)

    # Debugging information
    print("\n" + "=" * 80)
    print("FLARE DETECTION DEBUG INFO")
    print("=" * 80)
    
    print("\nFlare value counts:")
    print(df["flare"].value_counts())
    
    print("\n" + "-" * 80)
    print("Flare segment lengths (consecutive True values):")
    flare_lengths = (
        df["flare"]
        .astype(int)
        .groupby(
            df["flare"].ne(df["flare"].shift()).cumsum()
        )
        .sum()
    )
    
    print(flare_lengths.describe())
    
    print("\n" + "-" * 80)
    print("Top 20 longest flare segments:")
    print(flare_lengths.sort_values(ascending=False).head(20))
    
    print("\n" + "=" * 80)
    print("BUILDING EVENTS...")
    print("=" * 80 + "\n")

    events = detector.build_events(df)

    print()

    print(events.head())

    print()

    print(f"Detected flares: {len(events)}")

    print()

    # Show some summary statistics
    if len(events) > 0:
        print("Event duration statistics:")
        print(events["duration_sec"].describe())

        print("\nSample events:")
        for idx, row in events.head(5).iterrows():
            print(f"\nEvent {idx + 1}:")
            print(f"  Start: {row['start']}")
            print(f"  Peak:  {row['peak']}")
            print(f"  End:   {row['end']}")
            print(f"  Duration: {row['duration_sec']} sec")
            print(f"  Peak soft: {row['peak_soft']:.2f}")
            print(f"  Peak hard: {row['peak_hard']:.2f}")
