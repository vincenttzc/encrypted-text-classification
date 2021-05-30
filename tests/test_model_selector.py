import unittest

from src.model_selector import ModelSelector


class ModelSelectorTest(unittest.TestCase):
    def setUp(self):
        self.model_selector = ModelSelector
        self.model = "conv1D"
        self.model_config = {
            "input_size": 25,
            "num_classes": 3,
            "embedding_size": 50,
            "dropout_p": 0.5,
            "vocab_size": 38,
            "batch_normalisation": True,
            "conv_layers": [[32, 3, -1], [32, 3, -1], [32, 3, -1]],
            "dense_layers": [1024],
        }

    def test_model_selector_attr(self):
        """Test if attributes and methods are present"""
        model = ModelSelector(self.model, self.model_config)

        self.assertTrue(hasattr(model, "model"))
        self.assertTrue(hasattr(model, "model_config"))
        self.assertTrue(hasattr(model, "build_model"))

    def test_model_selector_build_model(self):
        """Test if model selector class can build model"""
        model = ModelSelector(self.model, self.model_config)
        model = model.build_model()

        self.assertIsNotNone(model)
