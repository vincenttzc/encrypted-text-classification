from typing import Dict

from tensorflow.keras.models import Model

from src.bilstmmodel import BiLSTMModel
from src.conv1dmodel import Conv1DModel


class ModelSelector:
    """Model selector class to build selected model"""

    def __init__(self, model: str, model_config: Dict):
        """Constructor method

        Args:
            model (str): selected model class
            model_config (Dict): selected model class's config
        """
        self.model = model
        self.model_config = model_config

    def build_model(self) -> Model:
        """Build selected model with the specified model config

        Returns:
            Model: selected model class
        """
        if self.model == "conv1D":
            model = Conv1DModel(**self.model_config)
        elif self.model == "biLSTM":
            model = BiLSTMModel(**self.model_config)
        model = model.build_model()

        return model
