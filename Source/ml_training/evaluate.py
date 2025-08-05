import os
import pandas as pd
import joblib
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Evaluation metrics on Test Set:")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC:   {roc_auc_score(y_test, y_proba):.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(y_test, y_proba):.4f})")
    plt.plot([0,1], [0,1], 'k--', label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

def hyperparameter_tuning(X_train, y_train):
    print("Starting hyperparameter tuning for Random Forest...")

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['auto', 'sqrt']
    }

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1', verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print("Best parameters found:")
    print(grid_search.best_params_)
    print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_

if __name__ == "__main__":
    TRAIN_PATH = '../../data/ml_prepared/train_data.csv'
    TEST_PATH = '../../data/ml_prepared/test_data.csv'
    MODEL_PATH = '../../models/random_forest_si_model.joblib'
    TUNED_MODEL_PATH = '../../models/random_forest_si_model_tuned.joblib'

    # Load datasets
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Separate features and labels
    X_train = train_df.drop(columns=['SI'])
    y_train = train_df['SI']

    X_test = test_df.drop(columns=['SI'])
    y_test = test_df['SI']

    # Load trained model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

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
