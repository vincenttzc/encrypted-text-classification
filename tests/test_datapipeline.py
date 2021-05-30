import unittest

import numpy as np
import pandas as pd

from src.datapipeline import Datapipeline


class DataPipelineTest(unittest.TestCase):
    def setUp(self):
        self.datapipeline = Datapipeline
        self.max_sequence_len = 250
        self.label_mapping = {"class_1": 0, "class_2": 1, "class_3": 2}
        self.train_data_path = "tests/data/train_data.csv"

    def test_datapipeline_attr(self):
        """Test if datapipeline has the required attributes and the key methods"""
        datapipeline = self.datapipeline(self.max_sequence_len, self.label_mapping)

        self.assertTrue(hasattr(datapipeline, "max_sequence_len"))
        self.assertTrue(hasattr(datapipeline, "label_mapping"))
        self.assertTrue(hasattr(datapipeline, "transform_train_data"))
        self.assertTrue(hasattr(datapipeline, "transform_test_data"))

    def test_encode_target(self):
        """Test if datapipeline can correctly encode the target label"""
        data = pd.read_csv(self.train_data_path)
        datapipeline = self.datapipeline(self.max_sequence_len, self.label_mapping)
        output = datapipeline.encode_target(data)

        unique_labels = set(np.unique(output))
        print("u", unique_labels)

        self.assertIsInstance(output, np.ndarray)
        self.assertTrue(unique_labels == set([0, 1, 2]))

    def test_fit_transform_character_sequences(self):
        """Test if datapipeline can correctly fit and transform text into character sequences"""
        datapipeline = self.datapipeline(self.max_sequence_len, self.label_mapping)
        data = pd.read_csv(self.train_data_path)
        output = datapipeline.fit_transform_character_sequences(data)

        self.assertIsInstance(output, np.ndarray)
        self.assertTrue(output.shape[1] == self.max_sequence_len)
        self.assertTrue(output.shape[0] == len(data))

    def test_transform_character_sequences(self):
        """Test if datapipeline can correctly transform text into character sequences (after
        it has been fitted)"""
        datapipeline = self.datapipeline(self.max_sequence_len, self.label_mapping)
        data = pd.read_csv(self.train_data_path)
        _ = datapipeline.fit_transform_character_sequences(data)
        output = datapipeline.transform_character_sequences(data)

        self.assertIsInstance(output, np.ndarray)
        self.assertTrue(output.shape[1] == self.max_sequence_len)
        self.assertTrue(output.shape[0] == len(data))

    def test_transform_train_data(self):
        """Test if datapipeline can correctly transform train data into character sequences and
        encoded labels"""
        datapipeline = self.datapipeline(self.max_sequence_len, self.label_mapping)
        data = pd.read_csv(self.train_data_path)
        char_embedding, target_label = datapipeline.transform_train_data(data)

        self.assertIsInstance(char_embedding, np.ndarray)
        self.assertTrue(char_embedding.shape[1] == self.max_sequence_len)
        self.assertTrue(char_embedding.shape[0] == len(data))

        self.assertIsInstance(target_label, np.ndarray)
        self.assertTrue(len(target_label) == len(data))

        unique_labels = set(np.unique(target_label))
        self.assertTrue(unique_labels == set([0, 1, 2]))

    def test_transform_test_data(self):
        """Test if datapipeline can correctly transform validation/ test data into
        character sequences and encoded labels (after it has been fitted)"""
        datapipeline = self.datapipeline(self.max_sequence_len, self.label_mapping)
        data = pd.read_csv(self.train_data_path)
        _ = datapipeline.fit_transform_character_sequences(data)
        char_embedding, target_label = datapipeline.transform_test_data(
            data, is_validation=True
        )

        self.assertIsInstance(char_embedding, np.ndarray)
        self.assertTrue(char_embedding.shape[1] == self.max_sequence_len)
        self.assertTrue(char_embedding.shape[0] == len(data))

        self.assertIsInstance(target_label, np.ndarray)
        self.assertTrue(len(target_label) == len(data))

        unique_labels = set(np.unique(target_label))
        self.assertTrue(unique_labels == set([0, 1, 2]))

    def test_get_vocab_size(self):
        """Test if datapipeline can get the number of character vocab"""
        datapipeline = self.datapipeline(self.max_sequence_len, self.label_mapping)
        data = pd.read_csv(self.train_data_path)
        _ = datapipeline.fit_transform_character_sequences(data)

        vocab_size = datapipeline.get_vocab_size()

        self.assertIsInstance(vocab_size, int)
