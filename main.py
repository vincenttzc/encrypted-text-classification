import json
import pandas as pd

from src.datapipeline import Datapipeline
from src.model_selector import ModelSelector
from src.postprocess import add_prediction_to_test_data

# Load config
with open("config/config.json", "r") as f:
    config = json.load(f)

# Read data
train_df = pd.read_csv(config["train_data_path"])
test_df = pd.read_csv(config["test_data_path"])

# Preprocess data
datapipeline = Datapipeline(config["max_sequence_len"], config["label_mapping"])
_, _ = datapipeline.transform_train_data(train_df)
X_test, _ = datapipeline.transform_test_data(test_df, is_validation=False)

# Build model and load trained weights
model = ModelSelector(config["model"], config["model_params"])
model = model.build_model()
model.load_weights(config["model_weights_path"])

# Predict and save file
y_pred = model.predict(X_test)
test_df_with_pred = add_prediction_to_test_data(
    y_pred, test_df, config["label_mapping"]
)
test_df_with_pred.to_csv(config["output_path"], index=False)
