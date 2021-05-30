from typing import List

from tensorflow.keras.layers import Input, Embedding, Flatten, Dense
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, BatchNormalization
from tensorflow.keras.models import Model


class Conv1DModel:
    def __init__(
        self,
        input_size: int,
        embedding_size: int,
        num_classes: int,
        vocab_size: int,
        dropout_p: float,
        batch_normalisation: bool,
        conv_layers: List[List],
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
            conv_layers (List[List]): list of list of convolutional layer.
            [filters, kernel_size, maxpooling_size]
            dense_layers (List): list of units for dense layer
        """
        self.input_size = input_size
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.dropout_p = dropout_p
        self.batch_normalisation = batch_normalisation
        self.conv_layers = conv_layers
        self.dense_layers = dense_layers

    def build_model(self) -> Model:
        """Builds keras model with the specified config when intiating class

        Returns:
            Model: initialised keras Model
        """
        embedding_layer = self.process_embedding_layer()

        inputs = Input(shape=(self.input_size,), dtype="int64")
        x = embedding_layer(inputs)

        # build conv layers
        for (filter_num, filter_size, pooling_size) in self.conv_layers:
            x = Conv1D(filter_num, filter_size, activation="relu")(x)
            if pooling_size != -1:
                x = MaxPooling1D(pool_size=pooling_size)(x)
        if self.batch_normalisation:
            x = BatchNormalization()(x)
        x = Flatten()(x)
        # build dense layers
        for dense_units in self.dense_layers:
            x = Dense(dense_units, activation="relu")(x)
            x = Dropout(self.dropout_p)(x)
        outputs = Dense(self.num_classes, activation="softmax")(x)
        model = Model(inputs=inputs, outputs=outputs)

        return model

    def process_embedding_layer(self) -> Embedding:
        """Creates embedding layer with specified config

        Returns:
            [Embedding]: embedding layer with specified config
        """
        embedding_layer = Embedding(
            self.vocab_size, self.embedding_size, input_length=self.input_size
        )
        return embedding_layer
