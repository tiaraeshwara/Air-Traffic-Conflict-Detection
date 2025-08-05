import pandas as pd
import os

def assign_si_label(df,
                    horiz_sep_col='horizontal_sep_NM',
                    vert_sep_col='vertical_sep_ft',
                    horiz_threshold=10.0,   # nautical miles
                    vert_threshold=1000.0): # feet
    """
    Assign binary Situation of Interest (SI) label to each aircraft pair based on
    separation thresholds:
        SI = 1 if horizontal separation < horiz_threshold AND vertical separation < vert_threshold
        Else, 0.

    Args:
        df (pd.DataFrame): DataFrame containing relative feature columns.
        horiz_sep_col (str): Column name for horizontal separation in NM.
        vert_sep_col (str): Column name for vertical separation in feet.
        horiz_threshold (float): Horizontal threshold for SI in NM.
        vert_threshold (float): Vertical threshold for SI in feet.

    Returns:
        pd.DataFrame: Same DataFrame with an added 'SI' column.
    """

    # Verify required columns exist
    if horiz_sep_col not in df.columns or vert_sep_col not in df.columns:
        raise ValueError(f"Columns {horiz_sep_col} and {vert_sep_col} must be present in DataFrame")

    df['SI'] = ((df[horiz_sep_col] < horiz_threshold) & (df[vert_sep_col] < vert_threshold)).astype(int)
    print(f"Assigned SI labels: {df['SI'].value_counts().to_dict()}")
    return df

def main():
    INPUT_FEATURES = '../../data/processed/features/features_segment_0.csv'  # Adjust path
    OUTPUT_LABELED = '../../data/processed/labeled/labeled_segment_0.csv'

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_LABELED), exist_ok=True)

    # Load features dataset
    df_features = pd.read_csv(INPUT_FEATURES)
    print(f"Loaded features data with shape: {df_features.shape}")

    # Assign SI labels
    df_labeled = assign_si_label(df_features)

    # Save labeled dataset for ML training
    df_labeled.to_csv(OUTPUT_LABELED, index=False)
    print(f"Labeled dataset saved to {OUTPUT_LABELED}")

    # Preview
    print(df_labeled.head())

if __name__ == "__main__":
    main()
