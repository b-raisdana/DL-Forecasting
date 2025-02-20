import numpy as np
from tensorflow import keras as tf_keras

from helper.br_py.logging import log_d, log_e


class CNNLSTMModel(tf_keras.models.Model):
    def __init__(self, y_shape: tuple, cnn_filters=64, lstm_units_list=None, dense_units=64, cnn_count=2,
                 cnn_kernel_growing_steps=2, dropout_rate=0.3):
        super(CNNLSTMModel, self).__init__(name="CNNLSTM_Model")
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
        try:
            sub_outputs = [self.submodels[key](inputs[
                                                   key
                                                   # f"indicators" if 'indicators' in key else key
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
            self.submodels[key].build(input_shape[shape_key])
        self.shape_of_input = input_shape
        super().build(input_shape)

    def summary(self, line_length=None, positions=None, print_fn=log_d):
        # print_fn("=" * 60)
        # print_fn(f"🔹 Summary of {self.name}")
        #
        # total_params, trainable_params, non_trainable_params = (0,) * 3
        #
        # for key, submodel in self.submodels.items():
        #     print_fn(f"\n📌 Submodel: {key}")
        #     print_fn("-" * 50)
        #     t_total_params, t_trainable_params, t_non_trainable_params = submodel.summary()
        #     total_params += t_total_params
        #     trainable_params += t_trainable_params
        #     non_trainable_params += t_non_trainable_params
        #
        # print_fn("=" * 60)
        # print_fn("\n🔹 Summary of Main Model")
        # print_fn("=" * 60)

        t = super().summary(line_length=line_length, positions=positions, print_fn=log_d)

        # total_params += self.count_params()
        # trainable_params += np.sum([np.prod(p.shape) for p in self.trainable_weights])
        # non_trainable_params += np.sum([np.prod(p.shape) for p in self.non_trainable_weights])
        #
        # bytes_per_param = 4
        # total_memory_MB = (total_params * bytes_per_param) / (1024 ** 2)
        #
        # print_fn("\n🔹 Parameter Count & GPU Memory Usage")
        # print_fn("=" * 60)
        # print_fn(f"✅ Total Parameters: {total_params:,}")
        # print_fn(f"✅ Trainable Parameters: {trainable_params:,}")
        # print_fn(f"✅ Non-Trainable Parameters: {non_trainable_params:,}")
        # print_fn(f"💾 Estimated GPU Memory Usage: {total_memory_MB:.2f} MB")
        # print_fn("=" * 60)

        # return total_params, trainable_params, non_trainable_params, total_memory_MB


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
            x = dropout(x)
            x = bn(x)
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

    def summary(self, print_fn: callable = log_d):
        for layer in self.conv_layers:
            if ~layer.trainable:
                # print_fn(f"CONV layer {layer.name} is not trainable!")
                layer.trainable = True
        for layer in self.lstm_layers:
            if ~layer.trainable:
                # print_fn(f"LSTM layer {layer.name} is not trainable: {layer.trainable}")
                layer.trainable = True
        trainable_params = np.sum([np.prod(p.shape) for p in self.trainable_weights])
        non_trainable_params = np.sum([np.prod(p.shape) for p in self.non_trainable_weights])
        total_params = trainable_params + non_trainable_params
        print_fn(f"🔹 Layer: {self.name}")
        print_fn(f"  - Total Parameters: {total_params:,}")
        print_fn(f"  - Trainable Parameters: {trainable_params:,}")
        print_fn(f"  - Non-Trainable Parameters: {non_trainable_params:,}")
        print_fn("-" * 50)

        return total_params, trainable_params, non_trainable_params
