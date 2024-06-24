import pandas as pd
from zenml import step
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)

train_path = "./data/train_set.csv"
test_path = "./data/test_set.csv"

@step
def preprocess_data(train_path: str = train_path, test_path: str = test_path) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    logging.info("Loading train data from: %s", train_path)
    train_data = pd.read_csv(train_path)
    logging.info("Loading test data from: %s", test_path)
    test_data = pd.read_csv(test_path)
    
    logging.info("Preprocessing data...")
    X_train = train_data.drop(columns=["Difficulty_level"])
    y_train = train_data["Difficulty_level"]
    X_test = test_data.drop(columns=["Difficulty_level"])
    y_test = test_data["Difficulty_level"]
    
    logging.info("Data preprocessing complete.")
    return X_train, y_train, X_test, y_test

# @step
# def preprocess_data(train_path: str = train_path, test_path: str = test_path) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
#     """
#     Load and preprocess training and test datasets.

#     Args:
#         train_path (str): File path to the training dataset.
#         test_path (str): File path to the test dataset.

#     Returns:
#         Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: Tuple containing X_train, y_train, X_test, y_test.
#     """
#     logging.info("Loading train data from: %s", train_path)
#     train_data = pd.read_csv(train_path)
    
#     logging.info("Loading test data from: %s", test_path)
#     test_data = pd.read_csv(test_path)
    
#     logging.info("Preprocessing data...")
#     X_train = train_data.drop(columns=["Difficulty_level"])
#     y_train = train_data["Difficulty_level"]
#     X_test = test_data.drop(columns=["Difficulty_level"])
#     y_test = test_data["Difficulty_level"]
    
    logging.info("Data preprocessing complete.")
    return X_train, y_train, X_test, y_test
