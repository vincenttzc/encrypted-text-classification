from typing import List

from tensorflow.keras.layers import Input, Embedding, Flatten, Dense
from tensorflow.keras.layers import (
    Dropout,
    BatchNormalization,
    Bidirectional,
    LSTM,
)
from tensorflow.keras.models import Model


class BiLSTMModel:
    def __init__(
        self,
        input_size: int,
        embedding_size: int,
        num_classes: int,
        vocab_size: int,
        dropout_p: float,
        batch_normalisation: bool,
        bilstm_layers: List[List],
        dense_layers: List,
    ):
        """Constructor method

        Args:
            input_size (int): input size of model (length of sequence)
            embedding_size (int): embedding dimension
            num_classes (int): num of target classes
            vocab_size (int): vocab size of character sequences
            dropout_p (float): dropout probability
            batch_normalisation (bool): set batch_normalisation
            bilstm_layers (List[List]): list of list of bilstm layer. [lstm units, lstm dropout]
            dense_layers (List): list of units for dense layer
        """
        self.input_size = input_size
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.dropout_p = dropout_p
        self.batch_normalisation = batch_normalisation
        self.bilstm_layers = bilstm_layers
        self.dense_layers = dense_layers

    def build_model(self) -> Model:
        """Builds keras model with the specified config when intiating class

        Returns:
            Model: initialised keras Model
        """
        embedding_layer = self.process_embedding_layer()

        inputs = Input(shape=(self.input_size,), dtype="int64")
        x = embedding_layer(inputs)

        # build bilstm layers
        for i, (lstm_units, lstm_dropout_p) in enumerate(self.bilstm_layers):
            x = Bidirectional(
                LSTM(lstm_units, return_sequences=True, dropout=lstm_dropout_p)
            )(x)
            if i == len(self.bilstm_layers) - 1:
                x = Bidirectional(LSTM(lstm_units, dropout=lstm_dropout_p))(x)

        if self.batch_normalisation:
            x = BatchNormalization()(x)

        # build dense layers
        for dense_units in self.dense_layers:
            x = Dense(dense_units, activation="relu")(x)
            x = Dropout(self.dropout_p)(x)
        outputs = Dense(self.num_classes, activation="softmax")(x)
        model = Model(inputs=inputs, outputs=outputs)

        return model

    def process_embedding_layer(self):
        """Creates embedding layer with specified config

        Returns:
            [Embedding]: embedding layer with specified config
        """
        embedding_layer = Embedding(
            self.vocab_size, self.embedding_size, input_length=self.input_size
        )
        return embedding_layer
