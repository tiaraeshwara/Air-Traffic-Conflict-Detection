import os
import pandas as pd

raw_data_dir = os.path.join(os.path.dirname(__file__), "../../Data/raw_data/")
processed_data_dir = os.path.join(os.path.dirname(__file__), '../../Data/processed_data/')
output_file = os.path.join(processed_data_dir, "combined_cleaned.csv")

columns_to_keep = ['time', 'icao24', 'callsign', 'lat', 'lon',
                   'baroaltitude', 'velocity', 'vertrate', 'heading']

def clean_and_load_file(file_path):
    print(f"Processing {file_path}")
    df = pd.read_csv(file_path)

    # Keep only relevant columns
    cols_in_file = [col for col in columns_to_keep if col in df.columns]
    df = df[cols_in_file].copy()

    # Drop rows with missing values
    df.dropna(subset=cols_in_file, inplace=True)

    # Remove negative velocity rows
    if 'velocity' in df.columns:
        df = df[df['velocity'] > 0]

    # Remove duplicates by icao24 and timestamp
    df.drop_duplicates(subset=['icao24', 'time'], inplace=True)

    # Remove invalid coordinates
    df = df[(df['lat'].between(-90, 90)) & (df['lon'].between(-180, 180))]

    print(f"Cleaned shape for {os.path.basename(file_path)}: {df.shape}")
    return df

def process_all_and_save():
    os.makedirs(processed_data_dir, exist_ok=True)
    csv_files = [os.path.join(raw_data_dir, f) for f in os.listdir(raw_data_dir) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_data_dir}")

    first_file = True
    total_rows = 0
    for file in csv_files:
        cleaned_df = clean_and_load_file(file)
        total_rows += len(cleaned_df)
        # Write header only for the first file
        cleaned_df.to_csv(output_file, mode='a', header=first_file, index=False)
        first_file = False

    print(f"Total combined cleaned data rows: {total_rows}")
    print(f"Saved combined cleaned data to {output_file}")

if __name__ == "__main__":
    process_all_and_save()