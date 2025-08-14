from Source.data_processing.process_and_combine import process_all_and_save
from Source.data_processing.segment_data import segment_adsb_data
from Source.data_processing.pair_generate import proceed_pairgeneration
from Source.feature_engineering.feature_engineering import load_and_process_all_pairs
from Source.label_assignment.label_assignment import label_and_save
from Source.ml_training.ml_dataset_prep import prepare_dataset
from Source.ml_training.train_model import train_rf_classifier
from Source.ml_training.evaluate import evaluate_model,hyperparameter_tuning

import os
import pandas as pd
import joblib



# Run the cleaning and combining process
process_all_and_save()

# Path to the cleaned combined data
cleaned_file_path = os.path.join(os.path.dirname(__file__), 'Data/processed_data/combined_cleaned.csv')
if not os.path.exists(cleaned_file_path):
    raise FileNotFoundError(f"Cleaned combined data file not found at {cleaned_file_path}")

# Load the cleaned data
df_cleaned = pd.read_csv(cleaned_file_path)
print(f"Loaded cleaned data shape: {df_cleaned.shape}")

# Segment data into 2-hour chunks and save CSV files for each segment
segments = segment_adsb_data(df_cleaned)

# Pair generation
proceed_pairgeneration()

#feature engineering and label assign
if __name__ == "__main__":
    INPUT_PAIRS = 'Data/processed_data/pairs/'
    OUTPUT_PATH = 'Data/processed_data/features/features_eng.csv'
    OUTPUT_LABELED ='Data/processed_data/labeled/labeled_data.csv'

    output_dir_ml = 'Data/processed_data/ml_prepared/'

    train_path = 'Data/processed_data/ml_prepared/train_data.csv'
    test_path = 'Data/processed_data/ml_prepared/test_data.csv'
    model_output_path = 'Model/random_forest_si_model.joblib'

    load_and_process_all_pairs(INPUT_PAIRS, OUTPUT_PATH)
    label_and_save(OUTPUT_PATH, OUTPUT_LABELED)
    prepare_dataset(OUTPUT_LABELED, output_dir_ml)
    train_rf_classifier(train_path, test_path, model_output_path)



    TUNED_MODEL_PATH = 'Model/random_forest_si_model_tuned.joblib'

    # Load datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Separate features and labels
    X_train = train_df.drop(columns=['SI'])
    y_train = train_df['SI']

    X_test = test_df.drop(columns=['SI'])
    y_test = test_df['SI']

    # Load trained model
    if not os.path.exists(model_output_path):
        raise FileNotFoundError(f"Trained model not found at {model_output_path}")
    model = joblib.load(model_output_path)

    # Evaluate the loaded model
    evaluate_model(model, X_test, y_test)

    # Optional: Perform hyperparameter tuning and save new model
    do_tuning = False  # Change to True to enable tuning
    if do_tuning:
        best_model = hyperparameter_tuning(X_train, y_train)
        joblib.dump(best_model, TUNED_MODEL_PATH)
        print(f"Tuned model saved at {TUNED_MODEL_PATH}")

        # Evaluate tuned model
        evaluate_model(best_model, X_test, y_test)