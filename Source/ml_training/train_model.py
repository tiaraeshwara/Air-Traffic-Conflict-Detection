import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

def train_rf_classifier(train_path, test_path, model_output_path, target_col='SI', random_state=42):
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

    # Initialize Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)

    # Train
    print("Training Random Forest classifier...")
    clf.fit(X_train, y_train)

    # Predict on test
    y_pred = clf.predict(X_test)

    # Evaluation
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Save the model to disk
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(clf, model_output_path)
    print(f"Model saved to {model_output_path}")

if __name__ == "__main__":
    TRAIN_PATH = '../../data/ml_prepared/train_data.csv'  # Adjust paths as per your project structure
    TEST_PATH = '../../data/ml_prepared/test_data.csv'
    MODEL_OUTPUT_PATH = '../../models/random_forest_si_model.joblib'

    train_rf_classifier(TRAIN_PATH, TEST_PATH, MODEL_OUTPUT_PATH)
