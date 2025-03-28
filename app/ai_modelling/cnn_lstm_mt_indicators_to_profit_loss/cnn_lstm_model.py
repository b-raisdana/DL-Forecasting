import numpy as np
from tensorflow import keras as tf_keras

from br_py.do_log import log_d, log_e


class CNNLSTMModel(tf_keras.models.Model):
    def __init__(self, y_shape: tuple, cnn_filters=64, lstm_units_list=None, dense_units=64, cnn_count=2,
                 cnn_kernel_growing_steps=2, dropout_rate=0.3, **kwargs):
        super(CNNLSTMModel, self).__init__(name="CNNLSTM_Model", **kwargs)
        self.y_shape = y_shape
        self.cnn_filters = cnn_filters
        self.lstm_units_list = lstm_units_list
        self.dense_units = dense_units
        self.cnn_count = cnn_count
        self.cnn_kernel_growing_steps = cnn_kernel_growing_steps
        self.dropout_rate = dropout_rate

        self.shape_of_input = None
        self.submodels = {
            key: CNNLSTMLayer(model_prefix=f"{key}_cnn_lstm_layer", dropout_rate=dropout_rate,
                              cnn_filters=cnn_filters, lstm_units_list=lstm_units_list, dense_units=dense_units,
                              cnn_count=cnn_count, cnn_kernel_growing_steps=cnn_kernel_growing_steps,
                              output_shape=y_shape)
            for key in ['structure', 'pattern', 'trigger', 'double', 'structure_indicators', 'pattern_indicators',
                        'trigger_indicators', 'double_indicators']
        }
        self.concat = tf_keras.layers.Concatenate()
        self.combined_dense = tf_keras.layers.Dense(256)
        self.leaky_relu = tf_keras.layers.LeakyReLU()
        self.final_output = tf_keras.layers.Dense(np.prod(y_shape), activation='linear')
        self.reshape_output = tf_keras.layers.Reshape(y_shape)

    def call(self, inputs):
        missing_keys = [key for key in self.submodels if key not in inputs]
        if missing_keys:
            log_e(f"Missing input keys: {missing_keys}")
            raise ValueError(f"Missing input keys: {missing_keys}")
        try:
            sub_outputs = [self.submodels[key](inputs[
                                                   key
                                               ]) for key in self.submodels.keys()]
            x = self.concat(sub_outputs)
            x = self.combined_dense(x)
            x = self.leaky_relu(x)
            x = self.final_output(x)
            return self.reshape_output(x)
        except Exception as e:
            log_e(f"Expected keys:{list(self.submodels.keys())}")
            log_e(f"Received keys:{list(inputs.keys())}")
            raise e

    def build(self, input_shape):
        for key in self.submodels:
            shape_key = 'indicators' if 'indicators' in key else key
            if shape_key in input_shape:
                self.submodels[key].build(input_shape[shape_key])
            else:
                raise Exception(f"Shape key {shape_key} not found in input_shape")
        self.shape_of_input = input_shape
        super().build(input_shape)

    def get_config(self):
        config = super(CNNLSTMModel, self).get_config()
        config.update({
            "y_shape": self.y_shape,  # Include all parameters needed for construction
            "cnn_filters": self.cnn_filters,
            "lstm_units_list": self.lstm_units_list,
            "dense_units": self.dense_units,
            "cnn_count": self.cnn_count,
            "cnn_kernel_growing_steps": self.cnn_kernel_growing_steps,
            "dropout_rate": self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Ensure y_shape is passed correctly to the constructor
        return cls(
            y_shape=config["y_shape"],
            cnn_filters=config["cnn_filters"],
            lstm_units_list=config["lstm_units_list"],
            dense_units=config["dense_units"],
            cnn_count=config["cnn_count"],
            cnn_kernel_growing_steps=config["cnn_kernel_growing_steps"],
            dropout_rate=config["dropout_rate"]
        )


class CNNLSTMLayer(tf_keras.layers.Layer):

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
            self.conv_layers.append(tf_keras.layers.Convolution1D(filters=cnn_filters * (i + 1),
                                                                  kernel_size=3 + i * cnn_kernel_growing_steps,
                                                                  padding='same',
                                                                  activation='relu',
                                                                  name=f'{model_prefix}_conv{i + 1}',
                                                                  ))
            self.bn_layers.append(
                tf_keras.layers.BatchNormalization(name=f'{model_prefix}_batch_norm_conv{i + 1}'))
            self.dropout_layers.append(
                tf_keras.layers.Dropout(dropout_rate, name=f'{model_prefix}_dropout_conv{i + 1}'))
        # LSTM Layers
        self.lstm_layers = []
        for i, lstm_units in enumerate(lstm_units_list):
            return_seq = i < len(lstm_units_list) - 1
            self.lstm_layers.append(tf_keras.layers.LSTM(lstm_units, return_sequences=return_seq,
                                                         name=f'{model_prefix}_lstm{i + 1}'))
        # Dense Layers
        self.dense1 = tf_keras.layers.Dense(dense_units, activation='relu', name=f'{model_prefix}_dense1')
        self.dropout_dense1 = tf_keras.layers.Dropout(dropout_rate, name=f'{model_prefix}_dropout_dense1')
        self.output_layer = tf_keras.layers.Dense(np.prod(output_shape), activation='linear',
                                                  name=f'{model_prefix}_output')

    def call(self, inputs):
        x = inputs
        for conv, bn, dropout in zip(self.conv_layers, self.bn_layers, self.dropout_layers):
            x = conv(x)
            x = bn(x)
            x = dropout(x)
        # x =  tf_keras.layers.core.Reshape((-1, self.conv_layers[-1].filters))(x)
        for lstm in self.lstm_layers:
            x = lstm(x)
        x = self.dense1(x)
        x = self.dropout_dense1(x)
        x = self.output_layer(x)
        x = tf_keras.layers.Reshape(self.target_shape)(x)
        return x

    def build(self, input_shape):
        # for layer in self.conv_layers:
        self.conv_layers[0].build(input_shape)
        self.shape_of_input = input_shape
        super().build(input_shape)

    def get_config(self):
        config = super(CNNLSTMLayer, self).get_config()
        config.update({
            "model_prefix": self.model_prefix,
            "output_shape": self.target_shape,
            "cnn_filters": self.conv_layers[0].filters,
            "lstm_units_list": [lstm.units for lstm in self.lstm_layers],
            "dense_units": self.dense1.units,
            "cnn_count": len(self.conv_layers),
            "cnn_kernel_growing_steps": self.cnn_kernel_growing_steps,
            "dropout_rate": self.dropout_layers[0].rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            model_prefix=config["model_prefix"],
            output_shape=config["output_shape"],
            cnn_filters=config["cnn_filters"],
            lstm_units_list=config["lstm_units_list"],
            dense_units=config["dense_units"],
            cnn_count=config["cnn_count"],
            cnn_kernel_growing_steps=config["cnn_kernel_growing_steps"],
            dropout_rate=config["dropout_rate"]
        )
