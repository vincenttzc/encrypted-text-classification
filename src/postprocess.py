from typing import Dict

import numpy as np
import pandas as pd


def add_prediction_to_test_data(
    y_pred: np.ndarray, test_data: pd.DataFrame, label_mapping: Dict
) -> pd.DataFrame:
    """Add model prediction to test data in the required submission format.
    Adds predicted label and prediction probability

    Args:
        y_pred (np.ndarray): model prediction
        test_data (pd.DataFrame): test data
        label_mapping (Dict): mapping of labels to index

    Returns:
        pd.DataFrame: dataframe
    """
    y_pred_label = np.argmax(y_pred, axis=1)
    y_pred_prob = np.max(y_pred, axis=1)

    reverse_mapping = {v: k for k, v in label_mapping.items()}

    test_data["label"] = y_pred_label
    test_data["label"] = test_data["label"].map(reverse_mapping)
    test_data["confidence_value"] = y_pred_prob
    test_data = test_data.rename({"Unnamed: 0": "id"}, axis=1)
    test_data = test_data.drop(labels="feature", axis=1)

    return test_data


def add_prediction_to_val_data(
    y_pred: np.ndarray, val_data: pd.DataFrame, label_mapping: Dict
) -> pd.DataFrame:
    """Add model prediction to validation data for error analysis purpose
    Adds predicted label and prediction probability

    Args:
        y_pred (np.ndarray): model prediction
        test_data (pd.DataFrame): test data
        label_mapping (Dict): mapping of labels to index

    Returns:
        pd.DataFrame: dataframe
    """
    y_pred_label = np.argmax(y_pred, axis=1)
    y_pred_prob = np.max(y_pred, axis=1)

    reverse_mapping = {v: k for k, v in label_mapping.items()}

    val_data["pred_label"] = y_pred_label
    val_data["pred_label"] = val_data["pred_label"].map(reverse_mapping)
    val_data["pred_confidence_value"] = y_pred_prob

    return val_data
