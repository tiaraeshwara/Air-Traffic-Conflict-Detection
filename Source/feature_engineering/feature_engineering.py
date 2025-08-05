import pandas as pd
from geographiclib.geodesic import Geodesic
import numpy as np
import os


def compute_relative_features(pairs_df):
    """
    Given a DataFrame of aircraft pairs, compute standard relative features for ML conflict detection.
    The DataFrame should have columns prefixed with 'A_' and 'B_' for each aircraft.

    Returns:
        pd.DataFrame: The input DataFrame with extra feature columns appended.
    """
    # Convenience aliases
    A_lat = pairs_df['A_latitude']
    A_lon = pairs_df['A_longitude']
    B_lat = pairs_df['B_latitude']
    B_lon = pairs_df['B_longitude']

    # Horizontal separation (geodesic distance in meters)
    def geo_dist(row):
        g = Geodesic.WGS84.Inverse(row['A_latitude'], row['A_longitude'],
                                   row['B_latitude'], row['B_longitude'])
        return g['s12']

    pairs_df['horizontal_sep_m'] = pairs_df.apply(geo_dist, axis=1)
    pairs_df['horizontal_sep_NM'] = pairs_df['horizontal_sep_m'] / 1852.0

    # Vertical separation (absolute altitude difference, typically in feet or meters)
    if 'A_baro_altitude' in pairs_df.columns and 'B_baro_altitude' in pairs_df.columns:
        pairs_df['vertical_sep_ft'] = (pairs_df['A_baro_altitude'] - pairs_df['B_baro_altitude']).abs()

    # Heading difference (wrap around 360)
    if 'A_heading' in pairs_df.columns and 'B_heading' in pairs_df.columns:
        diff = (pairs_df['A_heading'] - pairs_df['B_heading']).abs()
        pairs_df['heading_diff_deg'] = diff.apply(lambda x: min(x, 360 - x))

    # Groundspeed difference (absolute value, knots or m/s)
    if 'A_groundspeed' in pairs_df.columns and 'B_groundspeed' in pairs_df.columns:
        pairs_df['groundspeed_diff'] = (pairs_df['A_groundspeed'] - pairs_df['B_groundspeed']).abs()

    # Vertical rate difference
    if 'A_vertical_rate' in pairs_df.columns and 'B_vertical_rate' in pairs_df.columns:
        pairs_df['vertical_rate_diff'] = (pairs_df['A_vertical_rate'] - pairs_df['B_vertical_rate']).abs()

    # Track/course angle between aircraft (direction from A to B)
    def bearing(row):
        g = Geodesic.WGS84.Inverse(row['A_latitude'], row['A_longitude'],
                                   row['B_latitude'], row['B_longitude'])
        return g['azi1']

    pairs_df['bearing_A_to_B'] = pairs_df.apply(bearing, axis=1)
    if 'A_heading' in pairs_df.columns:
        diff_course = (pairs_df['A_heading'] - pairs_df['bearing_A_to_B']).abs()
        pairs_df['track_course_diff'] = diff_course.apply(lambda x: min(x, 360 - x))

    # Remove temp columns if desired (bearing), or keep for feature selection
    return pairs_df


# -------------- Example Usage --------------

if __name__ == "__main__":
    INPUT_PAIRS = '../../data/processed/pairs/pairs_segment_0.csv'
    OUTPUT_PATH = '../../data/processed/features/features_segment_0.csv'
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Load aircraft pairs generated previously
    pairs_df = pd.read_csv(INPUT_PAIRS)
    print("Loaded pairs: ", pairs_df.shape)

    # Compute engineered features
    features_df = compute_relative_features(pairs_df)
    print("Engineered features: ", features_df.shape)

    features_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved engineered features to {OUTPUT_PATH}")

    # Preview
    print(features_df.head())
