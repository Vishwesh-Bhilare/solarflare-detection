# src/detection/peak_detector.py

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = Path("data/processed/merged.parquet")
OUTPUT_PEAKS = Path("data/processed/peak_indices.parquet")
OUTPUT_CATALOG = Path("data/processed/flare_catalog.parquet")
OUTPUT_SMOOTHED = Path("data/processed/smoothed_data.parquet")


class PeakDetector:
    """
    Detects flares using peak detection on smoothed SoLEXS signal.
    
    Key improvements:
    1. Per-observation smoothing (no cross-day contamination)
    2. Rolling median background estimation
    3. Intelligent flare expansion
    4. Peak merging for multi-peak flares
    """
    
    def __init__(
        self,
        smooth_window: int = 31,
        smooth_order: int = 3,
        prominence: float = 10.0,
        width: float = 30,
        min_duration: int = 60,
        max_duration: int = 3600,
        merge_gap: int = 300,  # 5 minutes gap to merge peaks
        background_window: int = 300,  # 5 minutes for background median
    ):
        """
        Parameters
        ----------
        smooth_window : int
            Window size for Savitzky-Golay smoothing (must be odd)
        smooth_order : int
            Polynomial order for smoothing
        prominence : float
            Minimum prominence of peaks (counts)
        width : float
            Minimum width of peaks (seconds)
        min_duration : int
            Minimum flare duration in seconds
        max_duration : int
            Maximum flare duration in seconds
        merge_gap : int
            Maximum gap between peaks to merge into one flare (seconds)
        background_window : int
            Window size for rolling median background (seconds)
        """
        self.smooth_window = smooth_window
        self.smooth_order = smooth_order
        self.prominence = prominence
        self.width = width
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.merge_gap = merge_gap
        self.background_window = background_window
        
    def load_data(self) -> pd.DataFrame:
        """Load merged dataset."""
        logger.info(f"Loading data from {DATA_PATH}")
        df = pd.read_parquet(DATA_PATH)
        logger.info(f"Loaded {len(df):,} rows")
        logger.info(f"Unique observations: {df['observation'].nunique()}")
        return df
    
    def smooth_observation(self, df_obs: pd.DataFrame) -> pd.DataFrame:
        """
        Smooth a single observation's soft X-ray signal.
        No cross-day contamination!
        """
        signal = df_obs["soft_counts"].values
        
        # Apply Savitzky-Golay smoothing
        window_length = min(self.smooth_window, len(signal) if len(signal) % 2 == 1 else len(signal) - 1)
        if window_length < 3:
            window_length = 3
        polyorder = min(self.smooth_order, window_length - 1)
        
        smoothed = savgol_filter(
            signal,
            window_length=window_length,
            polyorder=polyorder,
        )
        
        df_obs = df_obs.copy()
        df_obs["soft_smoothed"] = smoothed
        
        # Compute rolling median background (quiet Sun level)
        # Use the same window size in seconds
        window_seconds = self.background_window
        
        # Convert to indices (approximate)
        if len(df_obs) > 1:
            time_diff = (df_obs["timestamp"].iloc[1] - df_obs["timestamp"].iloc[0]).total_seconds()
            window_indices = max(1, int(window_seconds / time_diff))
            
            # Rolling median with min_periods to handle edges
            background = (
                df_obs["soft_counts"]
                .rolling(window_indices, center=True, min_periods=1)
                .median()
            )
        else:
            background = df_obs["soft_counts"]
        
        df_obs["soft_background"] = background
        
        return df_obs
    
    def detect_peaks_in_observation(self, df_obs: pd.DataFrame) -> pd.DataFrame:
        """
        Detect peaks in a single observation.
        """
        smoothed = df_obs["soft_smoothed"].values
        
        # Only detect if we have enough points
        if len(smoothed) < 10:
            return pd.DataFrame()
        
        # Find peaks
        peaks, properties = find_peaks(
            smoothed,
            prominence=self.prominence,
            width=self.width,
            rel_height=0.5,
        )
        
        if len(peaks) == 0:
            return pd.DataFrame()
        
        # Create peak DataFrame
        peak_df = pd.DataFrame({
            "index": peaks,
            "timestamp": df_obs.iloc[peaks]["timestamp"].values,
            "peak_soft": smoothed[peaks],
            "prominence": properties["prominences"],
            "width": properties["widths"],
            "left_base": properties["left_bases"],
            "right_base": properties["right_bases"],
        })
        
        return peak_df
    
    def expand_peak_to_event(self, df_obs: pd.DataFrame, peak_idx: int, peak_value: float) -> dict:
        """
        Expand a single peak to find flare start and end.
        Stops when signal drops below background + noise threshold.
        """
        # Get background at peak
        background = df_obs.iloc[peak_idx]["soft_background"]
        
        # Calculate noise level (standard deviation of quiet signal)
        quiet_mask = df_obs["soft_counts"] <= background * 1.2
        if quiet_mask.sum() > 10:
            noise_std = df_obs.loc[quiet_mask, "soft_counts"].std()
        else:
            noise_std = 1.0
        
        threshold = background + 2 * noise_std  # 2-sigma above background
        
        # Walk left until signal <= threshold
        start_idx = peak_idx
        for i in range(peak_idx - 1, -1, -1):
            if df_obs.iloc[i]["soft_counts"] <= threshold:
                start_idx = i + 1
                break
            start_idx = i
        
        # Walk right until signal <= threshold
        end_idx = peak_idx
        for i in range(peak_idx + 1, len(df_obs)):
            if df_obs.iloc[i]["soft_counts"] <= threshold:
                end_idx = i - 1
                break
            end_idx = i
        
        # Calculate duration
        duration = (df_obs.iloc[end_idx]["timestamp"] - df_obs.iloc[start_idx]["timestamp"]).total_seconds()
        
        # Filter by duration
        if duration < self.min_duration or duration > self.max_duration:
            return None
        
        # Get the block data
        block = df_obs.iloc[start_idx:end_idx + 1]
        
        # Find actual peak (max in the region)
        actual_peak_idx = block["soft_counts"].idxmax()
        actual_peak_row = df_obs.loc[actual_peak_idx]
        
        return {
            "start": df_obs.iloc[start_idx]["timestamp"],
            "peak": actual_peak_row["timestamp"],
            "end": df_obs.iloc[end_idx]["timestamp"],
            "duration_sec": duration,
            "peak_soft": actual_peak_row["soft_counts"],
            "peak_smoothed": peak_value,
            "background": background,
            "prominence": peak_value - background,
            "start_idx": start_idx,
            "peak_idx": actual_peak_idx,
            "end_idx": end_idx,
            "observation": df_obs.iloc[0]["observation"],
        }
    
    def process_observation(self, df_obs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Process a single observation: smooth, detect peaks, expand to events.
        
        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            (peaks, events, smoothed_data_for_this_observation)
        """
        # Skip if too short
        if len(df_obs) < 10:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # Smooth - this returns a copy with smoothed columns
        df_obs_smoothed = self.smooth_observation(df_obs)
        
        # Detect peaks
        peaks = self.detect_peaks_in_observation(df_obs_smoothed)
        
        # Extract smoothed data for this observation
        smoothed_data = df_obs_smoothed[["timestamp", "soft_counts", "soft_smoothed", "soft_background"]].copy()
        smoothed_data["observation"] = df_obs_smoothed.iloc[0]["observation"]
        
        if len(peaks) == 0:
            return peaks, pd.DataFrame(), smoothed_data
        
        # Expand each peak to event
        events = []
        for _, peak_row in peaks.iterrows():
            event = self.expand_peak_to_event(
                df_obs_smoothed,
                peak_row["index"],
                peak_row["peak_soft"]
            )
            if event is not None:
                events.append(event)
        
        if len(events) == 0:
            return peaks, pd.DataFrame(), smoothed_data
        
        # Convert to DataFrame
        events_df = pd.DataFrame(events)
        
        # Merge overlapping events (multi-peak flares)
        if len(events_df) > 1:
            events_df = self.merge_overlapping_events(events_df, df_obs_smoothed)
        
        return peaks, events_df, smoothed_data
    
    def merge_overlapping_events(self, events_df: pd.DataFrame, df_obs: pd.DataFrame) -> pd.DataFrame:
        """
        Merge events that overlap or are close in time.
        """
        # Sort by start time
        events_df = events_df.sort_values("start_idx").reset_index(drop=True)
        
        merged_events = []
        
        for _, event in events_df.iterrows():
            if not merged_events:
                merged_events.append(event.to_dict())
                continue
            
            last = merged_events[-1]
            
            # Check if events overlap or are close
            gap = event["start_idx"] - last["end_idx"]
            
            if gap <= self.merge_gap:
                # Merge: extend end, update peak if higher
                if event["peak_soft"] > last["peak_soft"]:
                    last["peak"] = event["peak"]
                    last["peak_idx"] = event["peak_idx"]
                    last["peak_soft"] = event["peak_soft"]
                    last["peak_smoothed"] = event["peak_smoothed"]
                
                last["end"] = event["end"]
                last["end_idx"] = event["end_idx"]
                last["duration_sec"] = (event["end"] - last["start"]).total_seconds()
            else:
                merged_events.append(event.to_dict())
        
        return pd.DataFrame(merged_events)
    
    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run the complete peak detection pipeline.
        
        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            (all_peaks, all_events, smoothed_data)
        """
        # Load data
        df = self.load_data()
        
        all_peaks = []
        all_events = []
        smoothed_frames = []
        
        # Process each observation separately
        for obs_name, df_obs in tqdm(df.groupby("observation"), desc="Processing observations"):
            try:
                peaks, events, smoothed_data = self.process_observation(df_obs)
                
                if len(peaks) > 0:
                    all_peaks.append(peaks)
                
                if len(events) > 0:
                    all_events.append(events)
                
                # Always save smoothed data if we have it
                if len(smoothed_data) > 0:
                    smoothed_frames.append(smoothed_data)
                    
            except Exception as e:
                logger.warning(f"Error processing {obs_name}: {e}")
                continue
        
        # Combine results
        peaks_df = pd.concat(all_peaks, ignore_index=True) if all_peaks else pd.DataFrame()
        events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
        smoothed_df = pd.concat(smoothed_frames, ignore_index=True) if smoothed_frames else pd.DataFrame()
        
        # Sort by timestamp
        if not peaks_df.empty:
            peaks_df = peaks_df.sort_values("timestamp").reset_index(drop=True)
        if not events_df.empty:
            events_df = events_df.sort_values("start").reset_index(drop=True)
        if not smoothed_df.empty:
            smoothed_df = smoothed_df.sort_values("timestamp").reset_index(drop=True)
        
        # Save outputs
        logger.info(f"Saving {len(peaks_df)} peaks to {OUTPUT_PEAKS}")
        peaks_df.to_parquet(OUTPUT_PEAKS)
        
        logger.info(f"Saving {len(events_df)} flares to {OUTPUT_CATALOG}")
        events_df.to_parquet(OUTPUT_CATALOG)
        
        logger.info(f"Saving {len(smoothed_df)} rows of smoothed data to {OUTPUT_SMOOTHED}")
        smoothed_df.to_parquet(OUTPUT_SMOOTHED)
        
        return peaks_df, events_df, smoothed_df


if __name__ == "__main__":
    # Test the detector
    detector = PeakDetector(
        smooth_window=31,
        smooth_order=3,
        prominence=10.0,
        width=30,
        min_duration=60,
        max_duration=3600,
        merge_gap=300,  # Merge peaks within 5 minutes
        background_window=300,  # 5-minute background window
    )
    
    peaks, events, smoothed = detector.run()
    
    print("\n" + "=" * 80)
    print("PEAK DETECTION RESULTS")
    print("=" * 80)
    
    print(f"\nDetected {len(peaks)} peaks")
    print(f"Detected {len(events)} flare events")
    print(f"Smoothed data: {len(smoothed)} rows")
    
    if len(events) > 0:
        print("\nFlare catalog summary:")
        cols = ["start", "peak", "end", "duration_sec", "peak_soft", "observation"]
        if all(col in events.columns for col in cols):
            print(events[cols].head(10))
        
        print("\nDuration statistics:")
        print(events["duration_sec"].describe())
        
        print(f"\nTotal flare time: {events['duration_sec'].sum() / 60:.1f} minutes")
        
        # Count flares per observation
        if "observation" in events.columns:
            print("\nFlares per observation (top 10):")
            print(events["observation"].value_counts().head(10))
