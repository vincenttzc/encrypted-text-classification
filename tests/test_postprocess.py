import unittest

import numpy as np
import pandas as pd

from src.postprocess import add_prediction_to_test_data, add_prediction_to_val_data


class PostProcessTest(unittest.TestCase):
    def setUp(self):
        self.train_data_path = "tests/data/train_data.csv"
        self.test_data_path = "tests/data/test_data.csv"
        self.label_mapping = {"class_1": 0, "class_2": 1, "class_3": 2}

    def test_add_prediction_to_test_data(self):
        """Test if function can add prediction to test data correctly"""
        data = pd.read_csv(self.test_data_path)
        test_pred = np.ones((len(data), 3))

        output = add_prediction_to_test_data(test_pred, data, self.label_mapping)
        columns = output.columns

        self.assertIsInstance(output, pd.DataFrame)
        self.assertTrue("label" in columns)
        self.assertTrue("confidence_value" in columns)
        self.assertTrue("id" in columns)
        self.assertEquals(output.shape[1], 3)
        self.assertEquals(len(output), len(data))

    def test_add_prediction_to_val_data(self):
        """Test if function can add prediction to validation data correctly"""
        data = pd.read_csv(self.train_data_path)
        test_pred = np.ones((len(data), 3))

        output = add_prediction_to_val_data(test_pred, data, self.label_mapping)
        columns = output.columns

        self.assertIsInstance(output, pd.DataFrame)
        self.assertTrue("pred_label" in columns)
        self.assertTrue("pred_confidence_value" in columns)
        self.assertEquals(output.shape[1], 4)
        self.assertEquals(len(output), len(data))
