import os
import random

from tensorflow import keras as tf_keras

from Config import app_config
from ai_modelling.cnn_lstm_mt_indicators_to_profit_loss.base import dataset_folder, load_batch_zip, master_x_shape, \
    load_validators_zip
from ai_modelling.cnn_lstm_mt_indicators_to_profit_loss.cnn_lstm_model import CNNLSTMModel, CNNLSTMLayer


def model_compile(model):
    pass


def predict_it(x_shape, ):
    """
    load model from model_path_keras
    load
    Returns:

    """
    model_name = (f"cnn_lstm.mt_pnl_n_ind.cnn_f48c3k2.lstm_u256-128.dense_u128.drop_r0.3.keras")
    model_path_keras = os.path.join(app_config.path_of_data, model_name)

    model = tf_keras.models.load_model(model_path_keras,
                                       custom_objects={'CNNLSTMModel': CNNLSTMModel,
                                                       'CNNLSTMLayer': CNNLSTMLayer})
    model_compile(model)

    folder_name = dataset_folder(x_shape, 400)
    folder_path: str = os.path.join(app_config.path_of_data, folder_name)

    files = [f for f in os.listdir(folder_path) if f.startswith('dataset-') and f.endswith('.zip')]
    if not files or len(files) == 0:
        raise ValueError("No dataset files found!")

    selected_batch_file = random.randint(0, len(files) - 1)

    Xs, ys = load_batch_zip(x_shape=x_shape, batch_size=400, n = selected_batch_file)
    X_dfs, y_dfs, y_timeframe, y_tester_dfs = load_validators_zip(x_shape=x_shape, batch_size=400, n = selected_batch_file)

    selected_batch_row = random.randint(0, len(ys) - 1)
    predicted_y = model.predict(x=Xs)

    nop = 1


if __name__ == '__main__':
    predict_it(master_x_shape, )