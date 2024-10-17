import os

import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Conv1D, LeakyReLU, Flatten, Dense, Concatenate, LSTM
from tensorflow.python.keras.models import Model, load_model

cnn_lstd_model_input_lengths = {
    'structure': 128,
    'pattern': 256,
    'trigger': 256,
    'double': 256,
}


def train_model(structure_data, pattern_data, trigger_data, double_data, target_data, input_shapes, model=None):
    '''
    Check if the model is already trained or partially trained. If not, build a new model.
    Continue training the model and save the trained model to 'cnn_lstm_model.h5' after each session.

    Args:
        structure_data: Data for the structure timeframe.
        pattern_data: Data for the pattern timeframe.
        trigger_data: Data for the trigger timeframe.
        double_data: Data for the double timeframe.
        target_data: The labels or target values for training.
        input_shapes: A dictionary containing the input shapes for structure, pattern, trigger, and double timeframe data.
    Returns:
        The trained model.
    '''
    # Check if the model already exists, load if it does
    model_path = 'cnn_lstm_model.h5'

    if model is None:
        if os.path.exists(model_path):
            print("Loading existing model from disk...")
            model = load_model(model_path)
        else:
            print("Building new model...")
            model = build_model(input_shapes)

    # Train the model
    history = model.fit([structure_data, pattern_data, trigger_data, double_data],
                        target_data,
                        epochs=10,
                        batch_size=32)
    print(history)
    # Save the model after each training session to avoid losing progress
    model.save(model_path)
    print("Model saved to disk.")

    return model


def create_cnn_lstm(input_shape, name_prefix):
    input_layer = Input(shape=input_shape)

    # CNN Layer with ReLU activation
    conv = Conv1D(filters=64, kernel_size=3, padding='same')(input_layer)
    conv = LeakyReLU()(conv)
    conv = Conv1D(filters=64, kernel_size=3, padding='same')(conv)
    conv = LeakyReLU()(conv)

    # Flatten the CNN output
    flatten = Flatten()(conv)

    # LSTM Layer (LSTM has built-in activations)
    lstm = LSTM(64, return_sequences=False)(tf.expand_dims(flatten, axis=1))

    # Fully connected layer with ReLU activation
    dense = Dense(64)(lstm)
    dense = LeakyReLU()(dense)

    return Model(inputs=input_layer, outputs=dense)


def build_model(input_shapes):
    structure_model = create_cnn_lstm((cnn_lstd_model_input_lengths['structure'], 5), 'structure_model')
    pattern_model = create_cnn_lstm((cnn_lstd_model_input_lengths['pattern'], 5), 'pattern_model')
    trigger_model = create_cnn_lstm((cnn_lstd_model_input_lengths['trigger'], 5), 'trigger_model')
    double_model = create_cnn_lstm((cnn_lstd_model_input_lengths['double'], 5), 'double_model')

    combined_output = Concatenate()(
        [structure_model.output, pattern_model.output, trigger_model.output, double_model.output])

    # Add an additional Dense layer with ReLU activation
    combined_dense = Dense(128)(combined_output)
    combined_dense = LeakyReLU()(combined_dense)

    # Final output layer (for regression tasks, use linear activation; for classification, consider sigmoid/softmax)
    final_output = Dense(1, activation='linear')(combined_dense)

    # Define the final model
    model = Model(inputs=[structure_model.input, pattern_model.input, trigger_model.input, double_model.input],
                  outputs=final_output)

    # Compile the model with mean squared error loss for regression tasks
    model.compile(optimizer='adam', loss='mse')

    # Model summary
    model.summary()

    return model

# config.processing_date_range = date_range_to_string(start=pd.to_datetime('03-01-24'),
#                                                     end=pd.to_datetime('09-01-24'))
# # devided by rolling mean, std
# n_mt_ohlcv = pd.read_csv(
#     os.path.join(r"C:\Code\dl-forcasting\data\Kucoin\Spot\BTCUSDT",
#                  f"n_mt_ohlcv.{config.processing_date_range}.csv.zip"), compression='zip')
# n_mt_ohlcv
