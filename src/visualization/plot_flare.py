# src/visualization/plot_flare.py

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


class FlarePlotter:
    """
    Plot individual flares for visual verification.
    Uses flare_catalog.parquet for flare definitions.
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
        
    def plot_flare(self, flare_id: int, save: bool = True, show: bool = True):
        """
        Plot a single flare.
        
        Parameters
        ----------
        flare_id : int
            Index of the flare in the catalog DataFrame
        save : bool
            Whether to save the figure
        show : bool
            Whether to display the figure
        """
        if self.data is None or self.catalog is None:
            self.load_data()
        
        if flare_id >= len(self.catalog):
            logger.error(f"Flare ID {flare_id} out of range (max {len(self.catalog)-1})")
            return
        
        # Get flare data
        flare = self.catalog.iloc[flare_id]
        
        # Get the time range (add padding)
        start = flare["start"] - timedelta(minutes=5)
        end = flare["end"] + timedelta(minutes=5)
        
        # Get data slice
        mask = (self.data["timestamp"] >= start) & (self.data["timestamp"] <= end)
        df_slice = self.data[mask].copy()
        
        # Get smoothed data for the same period
        smooth_mask = (self.smoothed["timestamp"] >= start) & (self.smoothed["timestamp"] <= end)
        smooth_slice = self.smoothed[smooth_mask].copy()
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Plot soft X-ray with smoothed signal
        ax1.plot(
            df_slice["timestamp"],
            df_slice["soft_counts"],
            color="blue",
            linewidth=1,
            alpha=0.7,
            label="Raw SoLEXS",
        )
        
        # Add smoothed signal if available
        if not smooth_slice.empty:
            ax1.plot(
                smooth_slice["timestamp"],
                smooth_slice["soft_smoothed"],
                color="cyan",
                linewidth=2,
                alpha=0.8,
                label="Smoothed",
            )
            
            # Add background
            if "soft_background" in smooth_slice.columns:
                ax1.plot(
                    smooth_slice["timestamp"],
                    smooth_slice["soft_background"],
                    color="gray",
                    linewidth=1.5,
                    linestyle="--",
                    alpha=0.7,
                    label="Background",
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
            label="Flare region",
        )
        
        # Mark start, peak, end
        ax1.axvline(
            flare["start"],
            color="green",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="Start",
        )
        ax1.axvline(
            flare["peak"],
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="Peak",
        )
        ax1.axvline(
            flare["end"],
            color="purple",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="End",
        )
        
        # Add annotations
        duration = flare.get("duration_sec", 0)
        peak_soft = flare.get("peak_soft", 0)
        prominence = flare.get("prominence", 0)
        obs = flare.get("observation", "Unknown")
        
        ax1.text(
            0.02, 0.95,
            f"Flare #{flare_id}\n"
            f"Observation: {obs}\n"
            f"Duration: {duration:.0f}s ({duration/60:.1f} min)\n"
            f"Peak: {peak_soft:.1f} counts\n"
            f"Prominence: {prominence:.1f}",
            transform=ax1.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        
        ax1.set_ylabel("Soft X-ray Counts", fontsize=12)
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)
        ax1.set_title("SoLEXS Soft X-ray", fontsize=14, fontweight="bold")
        
        # Plot hard X-ray
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
                    linewidth=1.2,
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
                df_slice["hard_total"].max(),
                where=flare_mask,
                alpha=0.2,
                color="orange",
            )
        
        # Mark start, peak, end on hard X-ray
        ax2.axvline(flare["start"], color="green", linestyle="--", linewidth=1.5, alpha=0.5)
        ax2.axvline(flare["peak"], color="red", linestyle="--", linewidth=1.5, alpha=0.5)
        ax2.axvline(flare["end"], color="purple", linestyle="--", linewidth=1.5, alpha=0.5)
        
        ax2.set_ylabel("Hard X-ray Counts", fontsize=12)
        if has_hard:
            ax2.legend(loc="upper right", ncol=2)
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
            save_path = Path(f"plots/flare_{flare_id:04d}.png")
            save_path.parent.mkdir(exist_ok=True, parents=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved plot to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_summary(self, num_flares: int = 5):
        """Plot a summary grid of flares."""
        if self.catalog is None:
            self.load_data()
        
        # Select flares at regular intervals
        if len(self.catalog) > 0:
            flare_ids = np.linspace(0, len(self.catalog) - 1, min(num_flares, len(self.catalog))).astype(int)
            
            for fid in flare_ids:
                self.plot_flare(fid, save=True, show=False)
            
            logger.info(f"Plotted {len(flare_ids)} flares to plots/ directory")
        else:
            logger.warning("No flares found in catalog")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a specific flare")
    parser.add_argument("--id", type=int, help="Flare ID to plot")
    parser.add_argument("--summary", action="store_true", help="Plot summary of multiple flares")
    parser.add_argument("--num", type=int, default=5, help="Number of flares in summary")
    parser.add_argument("--save", action="store_true", default=True, help="Save the plot")
    parser.add_argument("--show", action="store_true", default=False, help="Show the plot")
    
    args = parser.parse_args()
    
    plotter = FlarePlotter()
    
    if args.summary:
        plotter.plot_summary(args.num)
    elif args.id is not None:
        plotter.plot_flare(args.id, save=args.save, show=args.show)
    else:
        print("Please specify either --id or --summary")
