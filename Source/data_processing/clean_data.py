import os
import pandas as pd

raw_data_dir = os.path.join(os.path.dirname(__file__), "../../Data/raw_data/")
processed_data_dir = os.path.join(os.path.dirname(__file__), '../../Data/processed_data/')

#columns to keep save

columns_to_keep = ['time', 'icao24', 'callsign', 'lat', 'lon',
                        'baroaltitude', 'velocity', 'vertrate', 'heading']

#clean and load single file

def clean_and_load_file(file_path):

    print(f"Processing {file_path}")
    df = pd.read_csv(file_path)

    #save only relevant columns
    cols_in_file = [col for col in columns_to_keep if col in df.columns]
    df = df[cols_in_file].copy()

    #drop rows with missing values
    df.dropna(subset=cols_in_file, inplace = True)

    #remove negative velocity rows
    if 'velocity' in df.columns:
        df = df[df['velocity'] > 0]

    # Remove duplicates by icao24 and timestamp
    df.drop_duplicates(subset=['icao24', 'time'], inplace=True)

    # Remove invalid coordinates of latitudes and longitudes
    df = df[(df['lat'].between(-90, 90)) & (df['lon'].between(-180, 180))]

    print(f"Cleaned shape for {os.path.basename(file_path)}: {df.shape}")
    return df

#Load, clean, and combine all CSV files in the raw_data directory.
def load_all_and_combine():

    csv_files = [os.path.join(raw_data_dir, f) for f in os.listdir(raw_data_dir) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_data_dir}")

    all_dfs = []
    for file in csv_files:
        cleaned_df = clean_and_load_file(file)
        all_dfs.append(cleaned_df)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Total combined cleaned data shape: {combined_df.shape}")
    return combined_df

def save_combined(df):
    os.makedirs(os.path.dirname(processed_data_dir), exist_ok=True)
    df.to_csv(processed_data_dir, index=False)
    print(f"Saved combined cleaned data to {processed_data_dir}")

if __name__ == "__main__":
    combined_cleaned_df = load_all_and_combine()
    save_combined(combined_cleaned_df)
