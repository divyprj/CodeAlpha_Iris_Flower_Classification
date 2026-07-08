import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def load_data(file_path="Iris.csv"):
    """Loads the Iris dataset and drops the 'Id' column if present."""
    df = pd.read_csv(file_path)
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
    return df

def train_model(df, model_type, hyperparams, test_size=0.2, random_state=42):
    """
    Trains a classification model on the Iris dataset.
    
    Parameters:
    - df: DataFrame containing features and target column 'Species'
    - model_type: str, type of classifier
    - hyperparams: dict, dictionary of hyperparameters for the model
    - test_size: float, proportion of dataset for testing
    - random_state: int, random seed for reproducibility
    
    Returns:
    - model: trained scikit-learn model
    - metrics: dict, accuracy, precision, recall, f1
    - X_test: DataFrame, test features
    - y_test: Series, test target (encoded)
    - y_pred: ndarray, predicted targets
    - le: LabelEncoder instance
    """
    # Split features and target
    X = df.drop('Species', axis=1)
    y = df['Species']
    
    # Encode target labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Train-test split with stratification to maintain class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    # Initialize the selected model
    if model_type == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=hyperparams.get("n_estimators", 100),
            max_depth=hyperparams.get("max_depth", None),
            random_state=random_state
        )
    elif model_type == "Support Vector Machine (SVM)":
        model = SVC(
            C=hyperparams.get("C", 1.0),
            kernel=hyperparams.get("kernel", "rbf"),
            probability=True,
            random_state=random_state
        )
    elif model_type == "K-Nearest Neighbors (KNN)":
        model = KNeighborsClassifier(
            n_neighbors=hyperparams.get("n_neighbors", 5),
            weights=hyperparams.get("weights", "uniform")
        )
    elif model_type == "Logistic Regression":
        model = LogisticRegression(
            C=hyperparams.get("C", 1.0),
            max_iter=hyperparams.get("max_iter", 200),
            random_state=random_state
        )
    elif model_type == "Decision Tree":
        model = DecisionTreeClassifier(
            max_depth=hyperparams.get("max_depth", None),
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    # Fit model
    model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    
    return model, metrics, X_test, y_test, y_pred, le

def predict_species(model, le, features_df):
    """
    Predicts the Iris species and returns the class and probabilities.
    """
    pred_encoded = model.predict(features_df)[0]
    pred_label = le.inverse_transform([pred_encoded])[0]
    
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features_df)[0]
        
    return pred_label, probabilities, le.classes_
