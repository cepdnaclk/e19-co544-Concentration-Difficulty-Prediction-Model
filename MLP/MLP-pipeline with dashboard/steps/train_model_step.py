import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from zenml import step
import logging
import joblib  # Add this import

logging.basicConfig(level=logging.INFO)

@step
def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> MLPClassifier:
    logging.info("Training model...")
    
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100), (200, 100, 50)],  
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'lbfgs'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
    }
    
    mlp = MLPClassifier(max_iter=100)
    grid_search = GridSearchCV(mlp, param_grid, cv=5, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # Save the best model
    model_path = "best_model.joblib"
    joblib.dump(best_model, model_path)
    logging.info("Model saved to %s", model_path)
    
    logging.info("Model training complete.")
    
    return best_model
