import pandas as pd
from geographiclib.geodesic import Geodesic
import numpy as np
import os
import glob


def compute_relative_features(pairs_df):
    """
    Given a DataFrame of aircraft pairs, compute standard relative features for ML conflict detection.
    The DataFrame should have columns prefixed with 'A_' and 'B_' for each aircraft.

    Returns:
        pd.DataFrame: The input DataFrame with extra feature columns appended.
    """
    # Convenience aliases
    A_lat = pairs_df['A_lat']
    A_lon = pairs_df['A_lon']
    B_lat = pairs_df['B_lat']
    B_lon = pairs_df['B_lon']

    # Horizontal separation (geodesic distance in meters)
    def geo_dist(row):
        g = Geodesic.WGS84.Inverse(row['A_lat'], row['A_lon'],
                                   row['B_lat'], row['B_lon'])
        return g['s12']

    pairs_df['horizontal_sep_m'] = pairs_df.apply(geo_dist, axis=1)
    pairs_df['horizontal_sep_NM'] = pairs_df['horizontal_sep_m'] / 1852.0

    # Vertical separation (absolute altitude difference, typically in feet or meters)
    if 'A_baroaltitude' in pairs_df.columns and 'B_baroaltitude' in pairs_df.columns:
        pairs_df['vertical_sep_ft'] = (pairs_df['A_baroaltitude'] - pairs_df['B_baroaltitude']).abs()

    # Heading difference (wrap around 360)
    if 'A_heading' in pairs_df.columns and 'B_heading' in pairs_df.columns:
        diff = (pairs_df['A_heading'] - pairs_df['B_heading']).abs()
        pairs_df['heading_diff_deg'] = diff.apply(lambda x: min(x, 360 - x))

    # velocity difference (absolute value, knots or m/s)
    if 'A_velocity' in pairs_df.columns and 'B_velocity' in pairs_df.columns:
        pairs_df['velocity_diff'] = (pairs_df['A_velocity'] - pairs_df['B_velocity']).abs()

    # Vertical rate difference
    if 'A_vertrate' in pairs_df.columns and 'B_vertrate' in pairs_df.columns:
        pairs_df['vertical_rate_diff'] = (pairs_df['A_vertrate'] - pairs_df['B_vertrate']).abs()

    # Track angle between aircraft (direction from A to B)
    def bearing(row):
        g = Geodesic.WGS84.Inverse(row['A_lat'], row['A_lon'],
                                   row['B_lat'], row['B_lon'])
        return g['azi1']

    pairs_df['bearing_A_to_B'] = pairs_df.apply(bearing, axis=1)
    if 'A_heading' in pairs_df.columns:
        diff_course = (pairs_df['A_heading'] - pairs_df['bearing_A_to_B']).abs()
        pairs_df['track_course_diff'] = diff_course.apply(lambda x: min(x, 360 - x))


    return pairs_df

def load_and_process_all_pairs(input_dir, output_path):

    # Find all pairs CSV files inside input_dir
    all_files = glob.glob(os.path.join(input_dir, '*.csv'))
    if not all_files:
        print(f"No CSV files found in directory {input_dir}")
        return

    all_features = []

    for file_path in sorted(all_files):
        print(f"Processing file: {file_path}")
        pairs_df = pd.read_csv(file_path)
        features_df = compute_relative_features(pairs_df)
        all_features.append(features_df)

    # Concatenate all features DataFrames into one
    combined_features = pd.concat(all_features, ignore_index=True)
    print(f"Total rows after combining: {len(combined_features)}")

    # Save combined feature set
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined_features.to_csv(output_path, index=False)

    print(f"Saved engineered features to {output_path}")
    print(combined_features.head())


