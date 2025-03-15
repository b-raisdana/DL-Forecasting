import numpy as np
from tensorflow import keras as tf_keras
from br_py.do_log import log_d, log_e

# Ensure CNNLSTMLayer is imported properly
# from some_module import CNNLSTMLayer

class CNNLSTMModel(tf_keras.models.Model):
    def __init__(self, y_shape: tuple, cnn_filters=64, lstm_units_list=None, dense_units=64, cnn_count=2,
                 cnn_kernel_growing_steps=2, dropout_rate=0.3, **kwargs):
        super(CNNLSTMModel, self).__init__(name="CNNLSTM_Model", **kwargs)
        self.y_shape = y_shape
        self.shape_of_input = None

        if lstm_units_list is None:
            lstm_units_list = [64]

        self.submodels = {
            key: CNNLSTMLayer(model_prefix=f"{key}_cnn_lstm_layer", dropout_rate=dropout_rate,
                              cnn_filters=cnn_filters, lstm_units_list=lstm_units_list, dense_units=dense_units,
                              cnn_count=cnn_count, cnn_kernel_growing_steps=cnn_kernel_growing_steps,
                              output_shape=y_shape)
            for key in ['structure', 'pattern', 'trigger', 'double', 'structure_indicators', 'pattern_indicators',
                        'trigger_indicators', 'double_indicators']
        }

        self.concat = tf_keras.layers.Concatenate()
        self.combined_dense = tf_keras.layers.Dense(256, activation='relu')
        self.final_output = tf_keras.layers.Dense(np.prod(y_shape), activation='linear')
        self.reshape_output = tf_keras.layers.Reshape(y_shape)

    def call(self, inputs):
        try:
            sub_outputs = [self.submodels[key](inputs[key]) for key in self.submodels.keys()]
            x = self.concat(sub_outputs)
            x = self.combined_dense(x)
            x = self.final_output(x)
            return self.reshape_output(x)
        except KeyError as e:
            log_e(f"Missing expected input key: {str(e)}")
            log_e(f"Expected keys: {list(self.submodels.keys())}")
            log_e(f"Received keys: {list(inputs.keys())}")
            raise e
        except Exception as e:
            log_e(f"Error in CNNLSTMModel call: {str(e)}")
            raise e

    def build(self, input_shape):
        for key in self.submodels:
            shape_key = key if key in input_shape else 'indicators' if 'indicators' in key else None
            if shape_key:
                self.submodels[key].build(input_shape[shape_key])

        self.shape_of_input = input_shape
        super().build(input_shape)

    def summary(self, line_length=None, positions=None, print_fn=log_d):
        super().summary(line_length=line_length, positions=positions, print_fn=print_fn)

    def get_config(self):
        config = super(CNNLSTMModel, self).get_config()
        config.update({
            "y_shape": self.y_shape,
            "cnn_filters": self.submodels['structure'].conv_layers[0].filters if self.submodels['structure'].conv_layers else 64,
            "lstm_units_list": [l.units for l in self.submodels['structure'].lstm_layers],
            "dense_units": self.combined_dense.units,
            "cnn_count": len(self.submodels['structure'].conv_layers),
            "cnn_kernel_growing_steps": 2,
            "dropout_rate": 0.3
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CNNLSTMLayer(tf_keras.layers.Layer):
    def __init__(self, model_prefix, output_shape, cnn_filters=64, lstm_units_list=None, dense_units=64,
                 cnn_count=2, cnn_kernel_growing_steps=2, dropout_rate=0.05):
        super(CNNLSTMLayer, self).__init__(name=f"{model_prefix}_layer")
        if lstm_units_list is None:
            lstm_units_list = [64]

        self.target_shape = output_shape
        self.conv_layers = [
            tf_keras.layers.Convolution1D(filters=cnn_filters * (i + 1),
                                          kernel_size=3 + i * cnn_kernel_growing_steps,
                                          padding='same',
                                          activation='relu',
                                          name=f'{model_prefix}_conv{i + 1}')
            for i in range(cnn_count)
        ]
        self.bn_layers = [tf_keras.layers.BatchNormalization(name=f'{model_prefix}_batch_norm{i + 1}') for i in range(cnn_count)]
        self.dropout_layers = [tf_keras.layers.Dropout(dropout_rate, name=f'{model_prefix}_dropout{i + 1}') for i in range(cnn_count)]

        self.lstm_layers = [
            tf_keras.layers.LSTM(units, return_sequences=(i < len(lstm_units_list) - 1), name=f'{model_prefix}_lstm{i + 1}')
            for i, units in enumerate(lstm_units_list)
        ]

        self.dense1 = tf_keras.layers.Dense(dense_units, activation='relu', name=f'{model_prefix}_dense1')
        self.dropout_dense1 = tf_keras.layers.Dropout(dropout_rate, name=f'{model_prefix}_dropout_dense1')
        self.output_layer = tf_keras.layers.Dense(np.prod(output_shape), activation='linear', name=f'{model_prefix}_output')

    def call(self, inputs):
        x = inputs
        for conv, bn, dropout in zip(self.conv_layers, self.bn_layers, self.dropout_layers):
            x = conv(x)
            x = bn(x)
            x = dropout(x)

        for lstm in self.lstm_layers:
            x = lstm(x)

        x = self.dense1(x)
        x = self.dropout_dense1(x)
        x = self.output_layer(x)
        return tf_keras.layers.Reshape(self.target_shape)(x)

    def build(self, input_shape):
        self.conv_layers[0].build(input_shape)
        super().build(input_shape)

    def get_config(self):
        return {
            "model_prefix": self.name,
            "output_shape": self.target_shape,
            "cnn_filters": self.conv_layers[0].filters if self.conv_layers else 64,
            "lstm_units_list": [l.units for l in self.lstm_layers],
            "dense_units": self.dense1.units,
            "cnn_count": len(self.conv_layers),
            "cnn_kernel_growing_steps": 2,
            "dropout_rate": self.dropout_layers[0].rate if self.dropout_layers else 0.05,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
