import pandas as pd
import itertools
import os

def generate_aircraft_pairs(segment_df, time_col='timestamp', ac_id_col='icao24'):
    """
    For each timestamp in the segment, generate all unique unordered pairs of aircraft.
    Returns a DataFrame with one row per pair (at a given timestamp), containing columns for both aircraft.

    Parameters:
        segment_df (pd.DataFrame): Data for one segment (columns include time_col, ac_id_col, features).
        time_col (str): Name of timestamp column.
        ac_id_col (str): Name of aircraft ID column.

    Returns:
        pd.DataFrame: Pairwise DataFrame, with columns for "A_" and "B_" for each aircraft in the pair.
    """

    # Ensure sorting for efficiency
    segment_df = segment_df.sort_values([time_col, ac_id_col]).reset_index(drop=True)
    result_rows = []

    # For each unique timestamp (or relax to a rolling interval if appropriate)
    for tstamp, grp in segment_df.groupby(time_col):
        aircrafts = grp[ac_id_col].unique()
        if len(aircrafts) < 2:
            continue  # Not enough for a pair

        # Get all unique unordered pairs (i, j), i < j
        pairs = list(itertools.combinations(aircrafts, 2))

        for acA, acB in pairs:
            recA = grp[grp[ac_id_col] == acA].iloc[0]
            recB = grp[grp[ac_id_col] == acB].iloc[0]

            # Side-by-side for each aircraft's relevant features (prefix with A_, B_)
            row = {}
            row[time_col] = tstamp
            row['A_' + ac_id_col] = acA
            row['B_' + ac_id_col] = acB
            for col in grp.columns:
                if col not in [time_col, ac_id_col]:
                    row['A_' + col] = recA[col]
                    row['B_' + col] = recB[col]
            result_rows.append(row)

    pairs_df = pd.DataFrame(result_rows)
    print(f"Total pairs generated: {len(pairs_df)}")
    return pairs_df

# ========== Example Usage ==========
if __name__ == "__main__":
    # Example: process a single segment file
    SEGMENT_PATH = '../../data/processed/segments/segment_0_1561593600_1561600800.csv'
    OUT_PATH = '../../data/processed/pairs/pairs_segment_0.csv'

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    # Load segment
    segment_df = pd.read_csv(SEGMENT_PATH)
    print(f"Loaded segment: {segment_df.shape}")

    # Generate pair DataFrame
    pairs_df = generate_aircraft_pairs(segment_df)

    # Save result
    pairs_df.to_csv(OUT_PATH, index=False)
    print(f"Saved aircraft pairs for segment to {OUT_PATH}")
