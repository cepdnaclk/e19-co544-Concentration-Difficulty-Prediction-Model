import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from zenml import step
import logging

logging.basicConfig(level=logging.INFO)

@step
def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> MLPClassifier:
    """
    Train an MLPClassifier model on the provided training data with hyperparameter tuning.

    Args:
        X_train (pd.DataFrame): Features of the training set.
        y_train (pd.Series): Labels of the training set.

    Returns:
        MLPClassifier: Trained MLPClassifier model.
    """
    logging.info("Training model...")
    
    # Define parameter grid for GridSearchCV
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100), (200, 100, 50)],  
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'lbfgs'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
    }
    
    # Initialize MLPClassifier
    mlp = MLPClassifier(max_iter=100)
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(mlp, param_grid, cv=5, verbose=1, n_jobs=-1)
    
    # Perform GridSearchCV to find best parameters
    grid_search.fit(X_train, y_train)
    
    # Get the best model from GridSearchCV
    best_model = grid_search.best_estimator_
    
    logging.info("Model training complete.")
    
    return best_model
