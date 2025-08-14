import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_dataset(OUTPUT_LABELED, output_dir_ml, target_col='SI', test_size=0.3, random_state=42):
    """
    Load labeled dataset, split into train/test, scale features, and save splits.

    Args:
        input_path (str): Path to the labeled CSV dataset.
        output_dir (str): Directory to save output train/test files.
        target_col (str): Name of the label column.
        test_size (float): Fraction of data used as test set.
        random_state (int): Random seed for reproducibility.

    Returns:
        None; saves train/test CSV files and scaler object if needed.
    """
    # Load dataset
    df = pd.read_csv(OUTPUT_LABELED)
    print(f"Loaded dataset with shape: {df.shape}")

    # Separate features and label
    id_cols = ['time', 'A_icao24', 'B_icao24']
    raw_state_cols = ['A_lat', 'A_lon', 'B_lat', 'B_lon', 'A_velocity', 'B_velocity',
        'A_baroaltitude', 'B_baroaltitude','A_heading', 'B_heading', 'A_vertrate', 'B_vertrate', ]

    X = df.drop(columns=id_cols + raw_state_cols + [target_col])
    y = df[target_col]

    print(f"Features shape after dropping IDs: {X.shape}")

    # Train-test split (stratify on target to keep class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Training set size: {X_train.shape[0]}, Testing set size: {X_test.shape[0]}")

    # Scale features (fit scaler on training data, transform both train and test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert scaled arrays back to DataFrames for easier export
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    # Combine scaled features and labels back for saving
    train_df = X_train_scaled.copy()
    train_df[target_col] = y_train.values

    test_df = X_test_scaled.copy()
    test_df[target_col] = y_test.values

    # Create output directory if not exist
    os.makedirs(output_dir_ml, exist_ok=True)

    # Save datasets
    train_path = os.path.join(output_dir_ml, 'train_data.csv')
    test_path = os.path.join(output_dir_ml, 'test_data.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved training set to {train_path}")
    print(f"Saved testing set to {test_path}")

    # Optionally save the scaler using joblib if you want to reuse it later
    # import joblib
    # scaler_path = os.path.join(output_dir, 'scaler.joblib')
    # joblib.dump(scaler, scaler_path)
    # print(f"Saved scaler to {scaler_path}")


