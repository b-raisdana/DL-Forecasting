import numpy as np
from tensorflow.keras.layers import Layer, BatchNormalization, Dropout
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.layers.convolutional import Conv1D
from tensorflow.python.keras.layers.core import Dense, Reshape
from tensorflow.python.keras.layers.merge import Concatenate
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.models import Model


class CNNLSTMModel(Model):
    def __init__(self, y_shape: tuple, filters=64, lstm_units_list=None, dense_units=64, cnn_count=2,
                 cnn_kernel_growing_steps=2, dropout_rate=0.3):
        super(CNNLSTMModel, self).__init__(name="CNNLSTM_Model")
        self.shape_of_input = None
        self.submodels = {
            key: CNNLSTMLayer(model_prefix=f"{key}_cnn_lstm_layer", dropout_rate=dropout_rate,
                              cnn_filters=filters, lstm_units_list=lstm_units_list, dense_units=dense_units,
                              cnn_count=cnn_count, cnn_kernel_growing_steps=cnn_kernel_growing_steps,
                              output_shape=y_shape)
            for key in ['structure', 'pattern', 'trigger', 'double', 'structure_indicators', 'pattern_indicators',
                        'trigger_indicators', 'double_indicators']
        }
        self.concat = Concatenate()
        self.combined_dense = Dense(256)
        self.leaky_relu = LeakyReLU()
        self.final_output = Dense(np.prod(y_shape), activation='linear')
        self.reshape_output = Reshape(y_shape)

    def call(self, inputs):
        sub_outputs = [self.submodels[key](inputs[
                                               f"indicators" if 'indicators' in key else key
                                           ]) for key in self.submodels.keys()]
        x = self.concat(sub_outputs)
        x = self.combined_dense(x)
        x = self.leaky_relu(x)
        x = self.final_output(x)
        return x  # self.reshape_output(x)

    def build(self, input_shape):
        for key in self.submodels:
            shape_key = 'indicators' if 'indicators' in key else key
            self.submodels[key].build(input_shape[shape_key])
        self.shape_of_input = input_shape
        super().build(input_shape)

    def summary(self):
        for key, submodel in self.submodels.items():
            print(f"submodel: {key}")
            submodel.summary()
            print("-" * 50)
        super().summary()
        total_params = self.count_params()
        trainable_params = np.sum([np.prod(p.shape) for p in self.trainable_weights])
        non_trainable_params = np.sum([np.prod(p.shape) for p in self.non_trainable_weights])

        bytes_per_param = 4
        total_memory_MB = (total_params * bytes_per_param) / (1024 ** 2)

        print(f"Model: {self.name}")
        print(f"+Total Parameters: {total_params:,}")
        print(f"+Trainable Parameters: {trainable_params:,}")
        print(f"+Non-Trainable Parameters: {non_trainable_params:,}")
        print(f"+Estimated GPU Memory Usage: {total_memory_MB:.2f} MB")
        print("-" * 50)

        return total_params, trainable_params, non_trainable_params, total_memory_MB

class CNNLSTMLayer(Layer):

    def __init__(self, model_prefix, output_shape, cnn_filters=64, lstm_units_list=None, dense_units=64,
                 cnn_count=2,
                 cnn_kernel_growing_steps=2, dropout_rate=0.05):
        super(CNNLSTMLayer, self).__init__(name=f"{model_prefix}_layer")
        if lstm_units_list is None:
            lstm_units_list = [64, ]
        self.target_shape = output_shape
        self.shape_of_input = None
        # CNN Layers
        self.conv_layers = []
        self.bn_layers = []
        self.dropout_layers = []

        for i in range(cnn_count):
            self.conv_layers.append(Conv1D(filters=cnn_filters * (i + 1),
                                           kernel_size=3 + i * cnn_kernel_growing_steps,
                                           padding='same',
                                           activation='relu',
                                           name=f'{model_prefix}_conv{i + 1}',
                                           ))
            self.bn_layers.append(
                BatchNormalization(name=f'{model_prefix}_batch_norm_conv{i + 1}'))
            self.dropout_layers.append(
                Dropout(dropout_rate, name=f'{model_prefix}_dropout_conv{i + 1}'))
        # LSTM Layers
        self.lstm_layers = []
        for i, lstm_units in enumerate(lstm_units_list):
            return_seq = i < len(lstm_units_list) - 1
            self.lstm_layers.append(LSTM(lstm_units, return_sequences=return_seq,
                                         name=f'{model_prefix}_lstm{i + 1}'))
        # Dense Layers
        self.dense1 = Dense(dense_units, activation='relu', name=f'{model_prefix}_dense1')
        self.dropout_dense1 = Dropout(dropout_rate, name=f'{model_prefix}_dropout_dense1')
        self.output_layer = Dense(np.prod(output_shape), activation='linear', name=f'{model_prefix}_output')

    def call(self, inputs):
        x = inputs
        for conv, bn, dropout in zip(self.conv_layers, self.bn_layers, self.dropout_layers):
            x = conv(x)
            x = dropout(x)
            x = bn(x)
        # x = Reshape((-1, self.conv_layers[-1].filters))(x)
        for lstm in self.lstm_layers:
            x = lstm(x)
        x = self.dense1(x)
        x = self.dropout_dense1(x)
        x = self.output_layer(x)
        x = Reshape(self.target_shape)(x)
        return x

    def build(self, input_shape):
        for layer in self.conv_layers:
            layer.build(input_shape)
        self.shape_of_input = input_shape
        super().build(input_shape)
