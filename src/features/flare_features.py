# src/features/flare_features.py

from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MERGE_PATH = Path("data/processed/merged.parquet")
CATALOG_PATH = Path("data/processed/flare_catalog.parquet")
OUTPUT_PATH = Path("data/processed/flare_features.parquet")


class FlareFeatures:
    """
    Compute comprehensive features for each detected flare.
    
    Features computed:
    - Peak soft X-ray flux
    - Background level
    - Prominence
    - Duration, rise_time, decay_time
    - Area under curve
    - Hard X-ray features (peak, integral, band ratios)
    """
    
    def __init__(self):
        self.data = None
        self.catalog = None
        
    def load_data(self):
        """Load merged dataset and flare catalog."""
        logger.info("Loading merged dataset...")
        self.data = pd.read_parquet(MERGE_PATH)
        logger.info(f"Loaded {len(self.data):,} rows")
        
        logger.info("Loading flare catalog...")
        self.catalog = pd.read_parquet(CATALOG_PATH)
        logger.info(f"Loaded {len(self.catalog)} flares")
        
    def compute_soft_features(self, block: pd.DataFrame) -> dict:
        """Compute soft X-ray features for a flare block."""
        features = {}
        
        # Basic statistics
        features["peak_soft"] = block["soft_counts"].max()
        features["background"] = block["soft_counts"].median()  # Approximate background
        
        # Prominence (peak - background)
        features["prominence"] = features["peak_soft"] - features["background"]
        
        # Duration
        features["duration_sec"] = len(block)
        
        # Find peak position
        peak_idx = block["soft_counts"].idxmax()
        peak_pos = block.index.get_loc(peak_idx)
        
        # Rise time (from start to peak)
        features["rise_time"] = peak_pos
        
        # Decay time (from peak to end)
        features["decay_time"] = len(block) - peak_pos - 1
        
        # Area under curve (integral)
        features["area_under_curve"] = block["soft_counts"].sum()
        
        # Additional statistics
        features["soft_mean"] = block["soft_counts"].mean()
        features["soft_std"] = block["soft_counts"].std()
        features["soft_max"] = features["peak_soft"]
        features["soft_min"] = block["soft_counts"].min()
        
        # Slope (linear fit)
        if len(block) > 1:
            x = np.arange(len(block))
            y = block["soft_counts"].values
            slope = np.polyfit(x, y, 1)[0]
            features["soft_slope"] = slope
        else:
            features["soft_slope"] = 0
        
        return features
    
    def compute_hard_features(self, block: pd.DataFrame) -> dict:
        """Compute hard X-ray features for a flare block."""
        features = {}
        
        # Peak values for each band
        features["peak_hard"] = block["hard_total"].max()
        features["peak_hard_5_20"] = block["hard_5_20"].max()
        features["peak_hard_20_30"] = block["hard_20_30"].max()
        features["peak_hard_30_40"] = block["hard_30_40"].max()
        features["peak_hard_40_60"] = block["hard_40_60"].max()
        
        # Integrals
        features["hard_total_integral"] = block["hard_total"].sum()
        features["hard_5_20_integral"] = block["hard_5_20"].sum()
        features["hard_20_30_integral"] = block["hard_20_30"].sum()
        features["hard_30_40_integral"] = block["hard_30_40"].sum()
        features["hard_40_60_integral"] = block["hard_40_60"].sum()
        
        # Averages
        features["hard_total_mean"] = block["hard_total"].mean()
        features["hard_5_20_mean"] = block["hard_5_20"].mean()
        features["hard_20_30_mean"] = block["hard_20_30"].mean()
        features["hard_30_40_mean"] = block["hard_30_40"].mean()
        features["hard_40_60_mean"] = block["hard_40_60"].mean()
        
        # Band ratios
        features["hard_ratio_20_30_5_20"] = (
            features["peak_hard_20_30"] / features["peak_hard_5_20"] 
            if features["peak_hard_5_20"] > 0 else 0
        )
        features["hard_ratio_30_40_5_20"] = (
            features["peak_hard_30_40"] / features["peak_hard_5_20"]
            if features["peak_hard_5_20"] > 0 else 0
        )
        
        # Hard/Soft ratio
        features["hard_soft_ratio"] = (
            features["peak_hard"] / features["peak_soft"]
            if features["peak_soft"] > 0 else 0
        )
        
        return features
    
    def compute_features_for_all_flares(self) -> pd.DataFrame:
        """Compute all features for every flare in the catalog."""
        logger.info("Computing features for all flares...")
        
        all_features = []
        
        for idx, row in tqdm(self.catalog.iterrows(), total=len(self.catalog), desc="Processing flares"):
            start_idx = row["start_idx"]
            end_idx = row["end_idx"]
            
            # Get the flare block
            block = self.data.iloc[start_idx:end_idx + 1].copy()
            
            # Compute features
            soft_features = self.compute_soft_features(block)
            hard_features = self.compute_hard_features(block)
            
            # Combine with flare metadata
            features = {
                "start": row["start"],
                "peak": row["peak"],
                "end": row["end"],
                "start_idx": start_idx,
                "peak_idx": row["peak_idx"],
                "end_idx": end_idx,
                **soft_features,
                **hard_features,
            }
            
            all_features.append(features)
        
        features_df = pd.DataFrame(all_features)
        
        # Sort by start time
        features_df = features_df.sort_values("start").reset_index(drop=True)
        
        logger.info(f"Computed features for {len(features_df)} flares")
        
        return features_df
    
    def run(self) -> pd.DataFrame:
        """Run the complete feature extraction pipeline."""
        self.load_data()
        features_df = self.compute_features_for_all_flares()
        
        # Save to parquet
        features_df.to_parquet(OUTPUT_PATH)
        logger.info(f"Saved flare features to {OUTPUT_PATH}")
        
        return features_df


if __name__ == "__main__":
    extractor = FlareFeatures()
    features = extractor.run()
    
    print("\n" + "=" * 80)
    print("FLARE FEATURES SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal flares: {len(features)}")
    print(f"\nColumns: {len(features.columns)}")
    print(features.columns.tolist())
    
    print("\nFirst 5 flares:")
    print(features[["start", "peak", "end", "duration_sec", "peak_soft", "peak_hard"]].head())
    
    print("\nFeature statistics:")
    print(features[["duration_sec", "rise_time", "decay_time", "peak_soft", "peak_hard"]].describe())
