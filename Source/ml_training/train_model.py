import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,precision_score, recall_score, f1_score
import joblib
from imblearn.over_sampling import SMOTE

def train_rf_classifier(train_path, test_path, model_output_path, target_col='SI', random_state=42):
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

    smote = SMOTE(random_state=random_state)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"SMOTE-applied training data shape: {X_train.shape}")

    # Initialize classifiers
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'XGBoost': XGBClassifier(eval_metric='logloss'),
        'Decision Tree': DecisionTreeClassifier(),
        'SVM': SVC(),
        'Naive Bayes': GaussianNB(),
        'KNN': KNeighborsClassifier(),
        'Logistic Regression': LogisticRegression(),

    }

    for name,clf in classifiers.items():
        #train data set
        clf.fit(X_train, y_train)
        #predict data set
        y_pred = clf.predict(X_test)
        print(f"\n{name} Classification Report:")
        print(classification_report(y_test, y_pred,digits=4))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred):.4f}")
        print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
        print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")


    # Save the model to disk
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(clf, model_output_path)
    print(f"Model saved to {model_output_path}")

