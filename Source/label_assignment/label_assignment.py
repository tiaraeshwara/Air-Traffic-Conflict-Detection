import pandas as pd
import os

def assign_si_label(df,
                    horiz_sep_col='horizontal_sep_NM',
                    vert_sep_col='vertical_sep_ft',
                    horiz_threshold=10.0,   # nautical miles
                    vert_threshold=1000.0): # feet

    # Verify required columns exist
    if horiz_sep_col not in df.columns or vert_sep_col not in df.columns:
        raise ValueError(f"Columns {horiz_sep_col} and {vert_sep_col} must be present in DataFrame")

    df['SI'] = ((df[horiz_sep_col] < horiz_threshold) & (df[vert_sep_col] < vert_threshold)).astype(int)
    print(f"Assigned SI labels: {df['SI'].value_counts().to_dict()}")
    return df

def label_and_save(OUTPUT_PATH,OUTPUT_LABELED):

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_LABELED), exist_ok=True)

    # Load features dataset
    df_features = pd.read_csv(OUTPUT_PATH)
    print(f"Loaded features data with shape: {df_features.shape}")

    # Assign SI labels
    df_labeled = assign_si_label(df_features)

    # Save labeled dataset for ML training
    df_labeled.to_csv(OUTPUT_LABELED, index=False)
    print(f"Labeled dataset saved to {OUTPUT_LABELED}")

    # Preview
    print(df_labeled.head())

