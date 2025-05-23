import tensorflow as tf
from keras.layers import Dense, Lambda, Softmax, Permute, Reshape
from keras.layers import GlobalAveragePooling1D
from keras.layers import Input, Conv1D, BatchNormalization, Dropout, LSTM
from keras.layers import LeakyReLU, Concatenate, Layer
from keras.models import Model


# def multi_head_attention(inputs, num_heads, key_dim, name=None):
#     total_dim = num_heads * key_dim
#
#     # Dense projections for Q, K, V
#     query = Dense(total_dim, name=f"{name}_query")(inputs)
#     key = Dense(total_dim, name=f"{name}_key")(inputs)
#     value = Dense(total_dim, name=f"{name}_value")(inputs)
#
#     def split_heads(x):
#         # x shape: (batch, seq_len, total_dim)
#         # reshape to (batch, seq_len, num_heads, key_dim)
#         # permute to (batch, num_heads, seq_len, key_dim)
#         x = Reshape((-1, num_heads, key_dim))(x)
#         return Permute((2, 1, 3))(x)
#
#     query_heads = split_heads(query)
#     key_heads = split_heads(key)
#     value_heads = split_heads(value)
#
#     def scaled_dot_product_attention(q, k, v):
#         matmul_qk = tf.matmul(q, k, transpose_b=True)  # (batch, heads, seq_len, seq_len)
#         dk = tf.cast(tf.shape(k)[-1], matmul_qk.dtype)  # match dtype here
#         scaled_logits = matmul_qk / tf.math.sqrt(dk)
#         attention_weights = Softmax(axis=-1)(scaled_logits)
#         output = tf.matmul(attention_weights, v)  # (batch, heads, seq_len, dim)
#         return output
#
#     def compute_attention_output_shape(input_shapes):
#         # input_shapes is a list of 3 shapes: (batch, heads, seq_len, key_dim)
#         batch_size, num_heads, seq_len, key_dim = input_shapes[0]
#         # Output shape same as input value shape: (batch, heads, seq_len, key_dim)
#         return (batch_size, num_heads, seq_len, key_dim)
#
#     attention = Lambda(
#         lambda x: scaled_dot_product_attention(x[0], x[1], x[2]),
#         output_shape=compute_attention_output_shape,
#         name=f"{name}_scaled_attention"
#     )([query_heads, key_heads, value_heads])
#
#     # Rearrange back to (batch, seq_len, total_dim)
#     attention = Permute((2, 1, 3), name=f"{name}_permute_back")(attention)
#
#     # Use -1 for dynamic reshape on seq_len dimension
#     attention = Reshape((-1, total_dim), name=f"{name}_reshape_back")(attention)
#
#     output = Dense(total_dim, name=f"{name}_output_dense")(attention)
#
#     return output


# class MultiHeadSelfAttention(Layer):
#     def __init__(self, num_heads, key_dim, **kwargs):
#         super().__init__(**kwargs)
#         self.num_heads = num_heads
#         self.key_dim = key_dim
#         self.total_dim = key_dim * num_heads
#
#     def build(self, input_shape):
#         self.query_dense = Dense(self.total_dim)
#         self.key_dense = Dense(self.total_dim)
#         self.value_dense = Dense(self.total_dim)
#         self.output_dense = Dense(self.total_dim)
#
#     def call(self, inputs):
#         batch_size = tf.shape(inputs)[0]
#
#         query = self.query_dense(inputs)
#         key = self.key_dense(inputs)
#         value = self.value_dense(inputs)
#
#         def split_heads(x):
#             x = tf.reshape(x, (batch_size, -1, self.num_heads, self.key_dim))
#             return tf.transpose(x, perm=[0, 2, 1, 3])
#
#         query = split_heads(query)
#         key = split_heads(key)
#         value = split_heads(value)
#
#         score = tf.matmul(query, key, transpose_b=True)
#         score /= tf.math.sqrt(tf.cast(self.key_dim, tf.float32))
#         weights = tf.nn.softmax(score, axis=-1)
#         attention = tf.matmul(weights, value)
#
#         attention = tf.transpose(attention, perm=[0, 2, 1, 3])
#         concat_attention = tf.reshape(attention, (batch_size, -1, self.total_dim))
#         return self.output_dense(concat_attention)


class MultiHeadAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.total_dim = num_heads * key_dim

    def build(self, input_shape):
        self.query_dense = Dense(self.total_dim, name="query_dense")
        self.key_dense = Dense(self.total_dim, name="key_dense")
        self.value_dense = Dense(self.total_dim, name="value_dense")
        self.output_dense = Dense(self.total_dim, name="output_dense")

    def call(self, inputs):
        # inputs: (batch, seq_len, features)
        batch_size = tf.shape(inputs)[0]

        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        # split into heads
        def split_heads(x):
            x = tf.reshape(x, (batch_size, -1, self.num_heads, self.key_dim))
            return tf.transpose(x, perm=[0, 2, 1, 3])  # (batch, heads, seq_len, key_dim)

        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)

        # Scaled Dot-Product Attention
        score = tf.matmul(query, key, transpose_b=True)
        score = score / tf.math.sqrt(tf.cast(self.key_dim, score.dtype))
        weights = tf.nn.softmax(score, axis=-1)
        attention = tf.matmul(weights, value)

        # Concatenate heads
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.total_dim))

        return self.output_dense(concat_attention)


def cnn_lstm_block(name, input_shape, cnn_filters, lstm_units, cnn_count, kernel_step, dropout_rate, num_heads, key_dim):
    inp = Input(shape=input_shape, name=f"{name}")
    x = inp

    for i in range(cnn_count):
        filters = cnn_filters * (i + 1)
        kernel = 3 + i * kernel_step
        x = Conv1D(filters, kernel_size=kernel, padding='same', activation='relu', name=f"{name}_conv{i}")(x)
        x = BatchNormalization(name=f"{name}_bn{i}")(x)
        x = Dropout(dropout_rate, name=f"{name}_dropout{i}")(x)

    for i, units in enumerate(lstm_units):
        #return_seq = i < len(lstm_units) - 1
        return_seq = True
        x = LSTM(units, return_sequences=return_seq, name=f"{name}_lstm{i}")(x)

    # Replace custom Layer with Lambda-based multi-head attention
    x = MultiHeadAttentionLayer(num_heads, key_dim, name=f"{name}_attention")(x)

    x = BatchNormalization(name=f"{name}_post_attn_bn")(x)
    x = GlobalAveragePooling1D(name=f"{name}_gap")(x)  # 👈 NEW
    x = Dense(64, activation='relu', name=f"{name}_dense")(x)
    x = Dropout(dropout_rate, name=f"{name}_dropout_dense")(x)
    x = Dense(128, activation='linear', name=f"{name}_output")(x)

    return inp, x


def build_cnn_lstm_attention_model(
        y_len,
        input_shapes,
        cnn_filters,  # =64,
        lstm_units,  # =[64, 32],
        cnn_count,  # =2,
        kernel_step,  # =2,
        dropout_rate,  # =0.3
        num_heads,  # = 4
        key_dim, # =16
):
    branches = []
    inputs = []

    for key in ['structure', 'pattern', 'trigger', 'double']:
        inp, out = cnn_lstm_block(
            name=key,
            input_shape=input_shapes[key],
            cnn_filters=cnn_filters,
            lstm_units=lstm_units,
            cnn_count=cnn_count,
            kernel_step=kernel_step,
            dropout_rate=dropout_rate,
            num_heads=num_heads, key_dim=key_dim,
        )
        inputs.append(inp)
        branches.append(out)

    merged = Concatenate()(branches)
    x = Dense(256)(merged)
    x = LeakyReLU()(x)
    x = Dense(y_len, activation='linear', dtype='float32')(x)
    out = Reshape((y_len,), name='output')(x)

    return Model(inputs=inputs, outputs=out)
