import pandas as pd
from sklearn.neural_network import MLPClassifier  # Add this import
from sklearn.metrics import accuracy_score
from zenml import step
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)

@step
def evaluate_model(trained_model: MLPClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluates the trained model on the test data.

    Args:
        trained_model (MLPClassifier): The trained classifier model.
        X_test (pd.DataFrame): Features of the test set.
        y_test (pd.Series): Labels of the test set.

    Returns:
        Dict[str, float]: Dictionary containing the evaluation metric(s).
    """
    logging.info("Evaluating model...")
    y_pred = trained_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info("Model evaluation complete. Accuracy: %f", accuracy)
    print(f"Model accuracy: {accuracy}")
    return {"accuracy": accuracy}
