from Source.data_processing import *
# from Source.data_processing.load_data import load_all_file, save_processed_data
from Source.data_processing.process_and_combine import process_all_and_save

# df_all = load_all_file('processed_data.csv')
# print(df_all.head())

from Source.data_processing.process_and_combine import process_all_and_save
# Run the cleaning and combining process
process_all_and_save()
