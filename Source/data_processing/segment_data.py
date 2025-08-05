import os
import pandas as pd

def segment_adsb_data(df, segment_length_sec=7200, timestamp_col='timestamp', save_segments=False):
    """
    Split the cleaned ADS-B DataFrame into segments based on fixed time windows.

    Args:
        df (pd.DataFrame): Cleaned ADS-B data with a timestamp column.
        segment_length_sec (int): Length of segment window in seconds (default 2 hours = 7200).
        timestamp_col (str): Name of the timestamp column in df (default 'timestamp').
        save_segments (bool): If True, saves each segment as CSV in 'data/processed/segments/'.

    Returns:
        dict: Dictionary with keys as (start_ts, end_ts) tuples and values as DataFrames.
    """

    # Ensure the timestamp column is integer and sorted
    df = df.copy()
    df[timestamp_col] = df[timestamp_col].astype(int)
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    min_ts = df[timestamp_col].min()
    max_ts = df[timestamp_col].max()

    print(f"Segmenting data from timestamp {min_ts} to {max_ts} in intervals of {segment_length_sec} seconds")

    segments = {}
    start_ts = min_ts
    segment_id = 0

    while start_ts <= max_ts:
        end_ts = start_ts + segment_length_sec

        # Filter rows within the current segment
        segment_df = df[(df[timestamp_col] >= start_ts) & (df[timestamp_col] < end_ts)].copy()

        if not segment_df.empty:
            segments[(start_ts, end_ts)] = segment_df

            if save_segments:
                # Prepare output path
                output_dir = os.path.join(os.path.dirname(__file__), '../../data/processed/segments/')
                os.makedirs(output_dir, exist_ok=True)
                file_name = f"segment_{segment_id}_{start_ts}_{end_ts}.csv"
                output_path = os.path.join(output_dir, file_name)
                segment_df.to_csv(output_path, index=False)
                print(f"Saved segment {segment_id} ({start_ts} - {end_ts}) with {len(segment_df)} records to {output_path}")

            segment_id += 1

        start_ts += segment_length_sec

    print(f"Total segments created: {len(segments)}")
    return segments


# Example usage inside this script (can be removed if importing as a module)
if __name__ == "__main__":
    # Example: load cleaned combined data produced by your cleaning step
    cleaned_file_path = os.path.join(os.path.dirname(__file__), '../../data/processed/cleaned_combined.csv')
    if not os.path.exists(cleaned_file_path):
        raise FileNotFoundError(f"Cleaned combined data file not found at {cleaned_file_path}")

    df_cleaned = pd.read_csv(cleaned_file_path)
    print(f"Loaded cleaned data shape: {df_cleaned.shape}")

    # Segment data into 2-hour chunks and save CSV files for each segment
    segments = segment_adsb_data(df_cleaned, segment_length_sec=7200, save_segments=True)

    # Optional: Access the first segment DataFrame
    first_segment_key = list(segments.keys())[0]
    print(f"First segment timestamps: {first_segment_key}")
    print(segments[first_segment_key].head())



