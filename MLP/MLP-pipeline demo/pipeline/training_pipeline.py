from zenml import pipeline
from steps.preprocess_data_step import preprocess_data
from steps.train_model_step import train_model
from steps.evaluate_model_step import evaluate_model

@pipeline
def training_pipeline():
    # Step 1: Preprocess Data (includes data loading)
    X_train, y_train, X_test, y_test = preprocess_data()
    
    # Step 2: Train Model
    trained_model = train_model(X_train=X_train, y_train=y_train)
    
    # Step 3: Evaluate Model
    evaluate_model(trained_model=trained_model, X_test=X_test, y_test=y_test)
