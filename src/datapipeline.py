from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Datapipeline:
    """Datapipeline that preprocesses pandas dataframe into numpy array that
    can be used by model"""

    def __init__(self, max_sequence_len: int, label_mapping: Dict):
        """Constructor method for Datapipeline class

        Args:
            max_sequence_len (int): set max length of sequence to truncate
        """
        self.max_sequence_len = max_sequence_len
        self.label_mapping = label_mapping

    def transform_train_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess pandas dataframe into numpy array of X and y so it can be
        fed into model. Also fits the pipeline so same transformation can be
        performed on val/ test set

        Args:
            data (pd.DataFrame): dataframe with feature and label column

        Returns:
            Tuple[np.ndarray, np.ndarray]: Returns preprocessed X value and y value
        """
        char_embedding = self.fit_transform_character_sequences(data)
        target_label = self.encode_target(data)

        return char_embedding, target_label

    def transform_test_data(
        self, data: pd.DataFrame, is_validation: bool
    ) -> Tuple[np.ndarray, Union[None, np.ndarray]]:
        """Preprocess pandas dataframe into numpy array of X and y so it can be
        fed into model. Will only return y if it is used for validation set

        Args:
            data (pd.DataFrame): dataframe with feature and label column
            is_validation (bool): True if is validation set, else False if test set

        Returns:
            Tuple[np.ndarray, Union[None, np.ndarray]]:
            Returns preprocessed X value. Returns preprocessed y value if is
            validation set, else return None.
        """
        if is_validation:
            target_label = self.encode_target(data)
        else:
            target_label = None
        char_embedding = self.transform_character_sequences(data)

        return char_embedding, target_label

    def get_vocab_size(self) -> int:
        """Returns vocab size learnt from train data. Add 1 for unknown token.

        Returns:
            int: vocab size
        """
        return len(self.tokenizer.word_index) + 1

    def encode_target(self, data: pd.DataFrame) -> np.ndarray:
        """Encode target labesl from string to int

        Args:
            data (pd.DataFrame): dataframe with label column

        Returns:
            np.ndarray: np array of encoded label
        """
        encoded_target = data["label"].map(self.label_mapping)

        return encoded_target.values

    def fit_transform_character_sequences(self, data: pd.DataFrame) -> np.ndarray:
        """Fits character encoding tokenizer and transforms feature column into
        sequence of character index

        Args:
            data (pd.DataFrame): dataframe with 'feature' column

        Returns:
            np.ndarray: numpy array of sequence of character index
        """
        self.tokenizer = Tokenizer(char_level=True, oov_token="UNK")
        self.tokenizer.fit_on_texts(data["feature"])
        train_sequences = self.tokenizer.texts_to_sequences(data["feature"])
        train_data = pad_sequences(
            train_sequences, maxlen=self.max_sequence_len, padding="post"
        )
        train_data = np.array(train_data, dtype="float32")

        return train_data

    def transform_character_sequences(self, data: pd.DataFrame) -> np.ndarray:
        """Transforms feature column into sequence of character index.
        self.fit_transform_character_sequences needs to be called first to fit
        the tokenizer

        Args:
            data (pd.DataFrame): dataframe with 'feature' column

        Returns:
            np.ndarray: numpy array of sequence of character index
        """
        test_sequences = self.tokenizer.texts_to_sequences(data["feature"])
        test_data = pad_sequences(
            test_sequences, maxlen=self.max_sequence_len, padding="post"
        )
        test_data = np.array(test_data, dtype="float32")

        return test_data


if __name__ == "__main__":
    train_path = "data/train.csv"
    test_path = "data/test.csv"
    max_sequence_len = 400

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    datapipeline = Datapipeline(max_sequence_len)
    x_train, y_train = datapipeline.transform_train_data(train_data)
    x_test, _ = datapipeline.transform_train_data(test_data, is_validation=False)
