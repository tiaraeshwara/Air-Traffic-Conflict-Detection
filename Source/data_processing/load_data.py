import os
from fileinput import filename

import numpy as np
import pandas as pd

# define the paths to raw_data directory and processed_data directory

raw_data_dir = os.path.join(os.path.dirname(__file__), '../../Data/raw_data')
processed_data_dir = os.path.join(os.path.dirname(__file__), '../../Data/processed_data')

#create a function to load a single raw ADS-B file into a pandas data frame

def load_single_file(filename):

    #define the relevant file path
    file_path = os.path.join(raw_data_dir,filename)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File {filename} not found in {raw_data_dir}")

    df = pd.read_csv(file_path)

    #check expected columns exist or not
    expected_columns = ['time', 'icao24', 'callsign', 'lat', 'lon',
                        'baroaltitude', 'velocity', 'vertrate', 'heading']
    #groundspeed unavailable velocity available
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns in {filename}: {missing_cols}")

    return df

# load all csv file and load into a single data frame
def load_all_file(out_filename):
    csv_files  = [f for f in os.listdir(raw_data_dir) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"File not found in {raw_data_dir}")
    if not os.path.exists(processed_data_dir):
        os.path.mkdir(processed_data_dir)
    out_path = os.path.join(processed_data_dir, out_filename)
    for file in csv_files:
        print(f"Loading {file}...")
        df_temp = load_single_file(file)

        df_list.append(df_temp)

    combined_df = pd.concat(df_list, ignore_index=True)



    combined_df.to_csv(out_path, index=False)
    print(f"Saved data to {out_path}")

    return combined_df


# save the combined data frame as csv to processed_data directory

def save_processed_data(df,out_filename):
    processed_data_dir = os.path.join(os.path.dirname(__file__), '../../Data/processed_data')

    if not os.path.exists(processed_data_dir):
        os.path.mkdir(processed_data_dir)

    out_path = os.path.join(processed_data_dir, out_filename)
    df.to_csv(out_path, index=False)
    print(f"Saved data to {out_path}")







