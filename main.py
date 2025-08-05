from Source.data_processing import *
from Source.data_processing.load_data import load_all_file, save_processed_data

df_all = load_all_file('processed_data.csv')
print(df_all.head())

#save_processed_data(df_all, 'processed_data.csv')
