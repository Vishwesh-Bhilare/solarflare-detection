# src/visualization/validate_flares.py

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MERGE_PATH = Path("data/processed/merged.parquet")
CATALOG_PATH = Path("data/processed/flare_catalog.parquet")
SMOOTHED_PATH = Path("data/processed/smoothed_data.parquet")


class FlareValidator:
    """
    Validate detected flares by visualizing them with both soft and hard X-ray data.
    """
    
    def __init__(self):
        self.data = None
        self.catalog = None
        self.smoothed = None
        
    def load_data(self):
        """Load merged dataset, flare catalog, and smoothed data."""
        logger.info("Loading data...")
        self.data = pd.read_parquet(MERGE_PATH)
        self.catalog = pd.read_parquet(CATALOG_PATH)
        self.smoothed = pd.read_parquet(SMOOTHED_PATH)
        logger.info(f"Loaded {len(self.data):,} rows, {len(self.catalog)} flares")
        
    def validate_flare(self, flare_id: int, save: bool = True, show: bool = True):
        """
        Validate a single flare with detailed visualization.
        """
        if self.data is None or self.catalog is None:
            self.load_data()
        
        if flare_id >= len(self.catalog):
            logger.error(f"Flare ID {flare_id} out of range (max {len(self.catalog)-1})")
            return
        
        # Get flare data
        flare = self.catalog.iloc[flare_id]
        
        # Get the time range (add padding)
        start = flare["start"] - timedelta(minutes=10)
        end = flare["end"] + timedelta(minutes=10)
        
        # Get data slice
        mask = (self.data["timestamp"] >= start) & (self.data["timestamp"] <= end)
        df_slice = self.data[mask].copy()
        
        # Get smoothed data for the same period
        smooth_mask = (self.smoothed["timestamp"] >= start) & (self.smoothed["timestamp"] <= end)
        smooth_slice = self.smoothed[smooth_mask].copy()
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
        
        # ============ TOP: Soft X-ray ============
        
        # Plot raw data
        ax1.plot(
            df_slice["timestamp"],
            df_slice["soft_counts"],
            color="blue",
            linewidth=1,
            alpha=0.5,
            label="Raw SoLEXS",
        )
        
        # Plot smoothed data
        if not smooth_slice.empty:
            ax1.plot(
                smooth_slice["timestamp"],
                smooth_slice["soft_smoothed"],
                color="cyan",
                linewidth=2,
                alpha=0.8,
                label="Smoothed",
            )
            
            # Plot background
            if "soft_background" in smooth_slice.columns:
                ax1.plot(
                    smooth_slice["timestamp"],
                    smooth_slice["soft_background"],
                    color="gray",
                    linewidth=2,
                    linestyle="--",
                    alpha=0.7,
                    label="Background (median)",
                )
        
        # Mark flare region
        flare_mask = (df_slice["timestamp"] >= flare["start"]) & (df_slice["timestamp"] <= flare["end"])
        ax1.fill_between(
            df_slice["timestamp"],
            0,
            df_slice["soft_counts"],
            where=flare_mask,
            alpha=0.3,
            color="orange",
            label="Detected flare",
        )
        
        # Mark start, peak, end with vertical lines
        ax1.axvline(
            flare["start"],
            color="green",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label="Start",
        )
        ax1.axvline(
            flare["peak"],
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label="Peak",
        )
        ax1.axvline(
            flare["end"],
            color="purple",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label="End",
        )
        
        # Mark peak with a point
        peak_time = flare["peak"]
        peak_value = flare["peak_soft"]
        ax1.scatter(
            [peak_time],
            [peak_value],
            color="red",
            s=100,
            zorder=5,
            label=f"Peak: {peak_value:.1f}",
        )
        
        # Add information box
        duration = flare.get("duration_sec", 0)
        prominence = flare.get("prominence", 0)
        obs = flare.get("observation", "Unknown")
        
        ax1.text(
            0.02, 0.95,
            f"Flare #{flare_id}\n"
            f"Observation: {obs}\n"
            f"Start: {flare['start'].strftime('%H:%M:%S')}\n"
            f"Peak: {flare['peak'].strftime('%H:%M:%S')}\n"
            f"End: {flare['end'].strftime('%H:%M:%S')}\n"
            f"Duration: {duration:.0f}s ({duration/60:.1f} min)\n"
            f"Peak: {peak_value:.1f} counts\n"
            f"Prominence: {prominence:.1f}",
            transform=ax1.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black"),
            fontsize=10,
        )
        
        ax1.set_ylabel("Soft X-ray Counts", fontsize=12)
        ax1.legend(loc="upper right", fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f"SoLEXS Soft X-ray - Flare #{flare_id}", fontsize=14, fontweight="bold")
        
        # ============ BOTTOM: Hard X-ray ============
        
        hard_bands = ["hard_total", "hard_5_20", "hard_20_30", "hard_30_40", "hard_40_60"]
        colors = ["red", "orange", "green", "blue", "purple"]
        labels = ["Total", "5-20 keV", "20-30 keV", "30-40 keV", "40-60 keV"]
        
        has_hard = False
        for band, color, label in zip(hard_bands, colors, labels):
            if band in df_slice.columns and not df_slice[band].isna().all():
                ax2.plot(
                    df_slice["timestamp"],
                    df_slice[band],
                    color=color,
                    linewidth=1.5,
                    alpha=0.7,
                    label=label,
                )
                has_hard = True
        
        if not has_hard:
            ax2.text(0.5, 0.5, "No hard X-ray data available", 
                    ha="center", va="center", transform=ax2.transAxes, fontsize=12)
        
        # Mark flare region on hard X-ray
        if "hard_total" in df_slice.columns and not df_slice["hard_total"].isna().all():
            ax2.fill_between(
                df_slice["timestamp"],
                0,
                df_slice["hard_total"].max() * 1.1,
                where=flare_mask,
                alpha=0.15,
                color="orange",
            )
        
        # Mark start, peak, end on hard X-ray
        ax2.axvline(flare["start"], color="green", linestyle="--", linewidth=1.5, alpha=0.5)
        ax2.axvline(flare["peak"], color="red", linestyle="--", linewidth=1.5, alpha=0.5)
        ax2.axvline(flare["end"], color="purple", linestyle="--", linewidth=1.5, alpha=0.5)
        
        # Add hard X-ray peak information
        if has_hard:
            hard_peak = df_slice["hard_total"].max() if "hard_total" in df_slice.columns else 0
            ax2.text(
                0.02, 0.95,
                f"Hard X-ray peak: {hard_peak:.1f} counts\n"
                f"Hard/Soft ratio: {hard_peak/peak_value:.2f}",
                transform=ax2.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black"),
                fontsize=10,
            )
        
        ax2.set_ylabel("Hard X-ray Counts", fontsize=12)
        if has_hard:
            ax2.legend(loc="upper right", ncol=2, fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_title("HEL1OS Hard X-ray", fontsize=14, fontweight="bold")
        
        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax2.set_xlabel("Time (UTC)", fontsize=12)
        
        # Rotate x-axis labels
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
        
        plt.tight_layout()
        
        if save:
            save_path = Path(f"plots/validation/flare_{flare_id:04d}.png")
            save_path.parent.mkdir(exist_ok=True, parents=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved validation plot to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def validate_all_flares(self):
        """Validate all detected flares."""
        if self.catalog is None:
            self.load_data()
        
        if len(self.catalog) == 0:
            logger.warning("No flares found in catalog")
            return
        
        logger.info(f"Validating {len(self.catalog)} flares...")
        for flare_id in range(len(self.catalog)):
            self.validate_flare(flare_id, save=True, show=False)
            if (flare_id + 1) % 10 == 0:
                logger.info(f"Validated {flare_id + 1}/{len(self.catalog)} flares")
        
        logger.info(f"Validated all {len(self.catalog)} flares")
    
    def validate_random_flares(self, n: int = 10):
        """Validate a random sample of flares."""
        if self.catalog is None:
            self.load_data()
        
        if len(self.catalog) == 0:
            logger.warning("No flares found in catalog")
            return
        
        n = min(n, len(self.catalog))
        flare_ids = np.random.choice(len(self.catalog), n, replace=False)
        flare_ids.sort()
        
        logger.info(f"Validating {n} random flares...")
        for flare_id in flare_ids:
            self.validate_flare(flare_id, save=True, show=False)
        
        logger.info(f"Validated {n} random flares")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate detected flares")
    parser.add_argument("--id", type=int, help="Specific flare ID to validate")
    parser.add_argument("--random", type=int, default=0, help="Number of random flares to validate")
    parser.add_argument("--all", action="store_true", help="Validate all flares")
    parser.add_argument("--save", action="store_true", default=True, help="Save the plots")
    parser.add_argument("--show", action="store_true", default=False, help="Show the plots")
    
    args = parser.parse_args()
    
    validator = FlareValidator()
    
    if args.id is not None:
        validator.validate_flare(args.id, save=args.save, show=args.show)
    elif args.random > 0:
        validator.validate_random_flares(args.random)
    elif args.all:
        validator.validate_all_flares()
    else:
        print("\nUsage:")
        print("  --id N     Validate a specific flare")
        print("  --random N Validate N random flares")
        print("  --all      Validate all flares")
        print("\nExamples:")
        print("  python validate_flares.py --id 0")
        print("  python validate_flares.py --random 5")
        print("  python validate_flares.py --all")
