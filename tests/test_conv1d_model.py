import unittest

import numpy as np
from tensorflow.keras.layers import Embedding
import tensorflow as tf

from src.conv1dmodel import Conv1DModel


class Conv1DModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = Conv1DModel
        self.config = {
            "input_size": 25,
            "num_classes": 3,
            "embedding_size": 50,
            "dropout_p": 0.5,
            "vocab_size": 38,
            "batch_normalisation": True,
            "conv_layers": [[32, 3, -1], [32, 3, -1], [32, 3, -1]],
            "dense_layers": [1024],
        }

    def test_attr(self):
        """Test if attributes and methods are present"""
        model = self.model(**self.config)

        attributes = [
            "input_size",
            "num_classes",
            "embedding_size",
            "dropout_p",
            "vocab_size",
            "batch_normalisation",
            "conv_layers",
            "dense_layers",
        ]
        methods = ["build_model", "process_embedding_layer"]

        for attr in attributes:
            self.assertTrue(hasattr(model, attr))
        for method in methods:
            self.assertTrue(hasattr(model, method))

    def test_build_model(self):
        """Test if class can build model"""
        model = self.model(**self.config)
        model = model.build_model()

        self.assertIsNotNone(model)

    def test_forward_pass(self):
        """Test if model can return the right output based on the model architecture"""
        model = self.model(**self.config)
        model = model.build_model()

        sample_x = np.ones((1, self.config["input_size"], 1))
        output = model(sample_x)

        self.assertTrue(tf.is_tensor(output))
        self.assertEquals(output.shape, [1, self.config["num_classes"]])

    def test_process_embedding_layer(self):
        """Test if the embedding layer can be created"""
        model = self.model(**self.config)
        embeddinglayer = model.process_embedding_layer()

        self.assertIsInstance(embeddinglayer, Embedding)
