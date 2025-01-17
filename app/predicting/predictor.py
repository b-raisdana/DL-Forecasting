import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from app.Config import app_config
from app.FigurePlotter.plotter import show_and_save_plot
from app.PreProcessing.encoding.rolling_mean_std import read_multi_timeframe_rolling_mean_std_ohlcv
from app.ai_modelling.cnn_lstm import cnn_lstd_model_x_lengths
from app.data_processing.fragmented_data import symbol_data_path
from app.data_processing.ohlcv import read_multi_timeframe_ohlcv
from app.helper.data_preparation import single_timeframe, trigger_timeframe
from app.helper.helper import profile_it
from app.helper.importer import go
from app.training.trainer import mt_train_n_test, plot_mt_train_n_test


@profile_it
def load_and_predict(input_x):
    # Load the model from disk
    model_path = os.path.join(app_config.path_of_data, 'cnn_lstm_model.keras')
    if os.path.exists(model_path):
        print("Loading model from disk...")
        model = load_model(model_path)
    else:
        raise FileNotFoundError(f"The model file does not exist at {model_path}")

    # Make predictions
    predictions = model.predict({'structure_model_input': input_x['structure'],
           'pattern_model_input': input_x['pattern'],
           'trigger_model_input': input_x['trigger'],
           'double_model_input': input_x['double']})
    return predictions


def prediction_reconstructor(row, prev_row):
    """
    Reconstruct high and low values based on previous row's statistics and current row's predictions.

    Args:
        row: The current row in the rolling window (contains current n_high, n_low, mean_high, mean_low, std_high, std_low).
        prev_row: The previous row's statistics (mean_high, mean_low, std_high, std_low).

    Returns:
        Reconstructed high, low, mean_high, mean_low, std_high, std_low.
    """
    # Calculate high and low using the previous row's mean and std
    high = prev_row['mean_high'] + prev_row['std_high'] * row['n_high']
    low = prev_row['mean_low'] + prev_row['std_low'] * row['n_low']

    # Update mean and standard deviation
    mean_high = (prev_row['mean_high'] * 255 + high) / 256
    mean_low = (prev_row['mean_low'] * 255 + low) / 256

    std_high = np.sqrt(((255 * prev_row['std_high'] ** 2) + (high - mean_high) ** 2) / 256)
    std_low = np.sqrt(((255 * prev_row['std_low'] ** 2) + (low - mean_low) ** 2) / 256)

    return pd.Series([high, low, mean_high, mean_low, std_high, std_low],
                     index=['high', 'low', 'mean_high', 'mean_low', 'std_high', 'std_low'])


def reconstruct_from_prediction(last_x_stats, last_x_double_datetime, predictions: np.array, predictions_tf,
                                forecast_horizon: int = 20):
    training_y_columns = ['n_high', 'n_low']

    # Create DataFrame from predictions
    reconstructed = pd.DataFrame(predictions, columns=training_y_columns,
                                 index=pd.date_range(last_x_double_datetime + pd.to_timedelta(predictions_tf),
                                                     periods=predictions.shape[0], freq=predictions_tf))
    reconstructed.index.name = 'date'

    # Initialize mean and std columns using the last known statistics
    reconstructed['mean_high'] = np.nan
    reconstructed['mean_low'] = np.nan
    reconstructed['std_high'] = np.nan
    reconstructed['std_low'] = np.nan
    reconstructed['high'] = np.nan
    reconstructed['low'] = np.nan

    # Set the first row's mean and std values based on last known statistics
    reconstructed.iloc[0, reconstructed.columns.get_loc('mean_high')] = last_x_stats[
        'mean_high']
    reconstructed.iloc[0, reconstructed.columns.get_loc('mean_low')] = last_x_stats['mean_low']
    reconstructed.iloc[0, reconstructed.columns.get_loc('std_high')] = last_x_stats['std_high']
    reconstructed.iloc[0, reconstructed.columns.get_loc('std_low')] = last_x_stats['std_low']
    reconstructed.iloc[0, reconstructed.columns.get_loc('high')] = last_x_stats['high']
    reconstructed.iloc[0, reconstructed.columns.get_loc('low')] = last_x_stats['low']

    # Iterate over the forecast horizon to compute reconstructed values
    for i in range(1, forecast_horizon):
        prev_row = reconstructed.iloc[i - 1]
        current_row = reconstructed.iloc[i]
        reconstructed.loc[reconstructed.index[i], ['high', 'low', 'mean_high', 'mean_low', 'std_high', 'std_low']] = (
            prediction_reconstructor(current_row, prev_row))[['high', 'low', 'mean_high', 'mean_low', 'std_high', 'std_low']]

    return reconstructed


def plot_mt_predict(x_df: dict[str, pd.DataFrame], y_df: pd.DataFrame, predictions: np.array, predictions_tf: str,
                    n: int,
                    base_ohlcv, show=True):
    # fig = plot_mt_train_n_test(x, y, n, base_ohlcv, show=False)
    # prediction
    fig = go.Figure()
    reconstructed = reconstruct_from_prediction(
        last_x_stats={
            'mean_high': x_df['double'][n].iloc[-1]['mean_high'],
            'mean_low': x_df['double'][n].iloc[-1]['mean_low'],
            'std_high': x_df['double'][n].iloc[-1]['std_high'],
            'std_low': x_df['double'][n].iloc[-1]['std_low'],
            'high': x_df['double'][n].iloc[-1]['high'],
            'low': x_df['double'][n].iloc[-1]['low'],
        },
        last_x_double_datetime=x_df['double'][n].index.get_level_values(level='date').max(),
        predictions=predictions[n],
        predictions_tf=predictions_tf,
    )  # last_x_stats, last_x_datetime, predictions: np.array, timeframe,
    fig.add_trace(go.Candlestick(
        x=reconstructed.index.get_level_values('date'),
        open=reconstructed['low'],
        high=reconstructed['high'],
        low=reconstructed['low'],
        close=reconstructed['high'],
        name='Prediction'
    ))
    # x structure
    ohlcv = x_df['structure'][n]
    fig.add_trace(go.Candlestick(
        x=ohlcv.index.get_level_values('date'),
        open=ohlcv['low'],
        high=ohlcv['high'],
        low=ohlcv['low'],
        close=ohlcv['high'],
        name='Structure'
    ))
    # x pattern
    ohlcv = x_df['pattern'][n]
    fig.add_trace(go.Candlestick(
        x=ohlcv.index.get_level_values('date'),
        open=ohlcv['low'],
        high=ohlcv['high'],
        low=ohlcv['low'],
        close=ohlcv['high'],
        name='Pattern'
    ))
    # x trigger
    ohlcv = x_df['trigger'][n]
    fig.add_trace(go.Candlestick(
        x=ohlcv.index.get_level_values('date'),
        open=ohlcv['low'],
        high=ohlcv['high'],
        low=ohlcv['low'],
        close=ohlcv['high'],
        name='Trigger'
    ))
    # x double
    ohlcv = x_df['double'][n]
    fig.add_trace(go.Candlestick(
        x=ohlcv.index.get_level_values('date'),
        open=ohlcv['low'],
        high=ohlcv['high'],
        low=ohlcv['low'],
        close=ohlcv['high'],
        name='Double'
    ))
    # y
    ohlcv = y_df[n]
    fig.add_trace(go.Candlestick(
        x=ohlcv.index.get_level_values('date'),
        close=ohlcv['low'],
        high=ohlcv['high'],
        low=ohlcv['low'],
        open=ohlcv['high'],
        name='Y'
    ))
    show_and_save_plot(fig.update_yaxes(fixedrange=False))


if __name__ == "__main__":
    # Define the path to your saved model

    # # Prepare your input data (assuming you have already preprocessed it)
    # # This should be a list of numpy arrays, one for each input branch of your model
    # input_structure = np.random.rand(1, 5)  # Example shape (1, 5)
    # input_pattern = np.random.rand(1, 5)  # Example shape (1, 5)
    # input_trigger = np.random.rand(1, 5)  # Example shape (1, 5)
    # input_double = np.random.rand(1, 5)  # Example shape (1, 5)
    symbol = 'BTCUSDT'
    app_config.under_process_symbol = symbol
    app_config.processing_date_range = "24-08-20.00-00T24-10-31.00-00"
    n_mt_ohlcv = read_multi_timeframe_rolling_mean_std_ohlcv(app_config.processing_date_range)
    mt_ohlcv = read_multi_timeframe_ohlcv(app_config.processing_date_range)
    structure_tf = '4h'
    base_ohlcv = single_timeframe(mt_ohlcv, '15min')
    batch_size = 1
    x, y, x_df, y_df, y_timeframe = mt_train_n_test(structure_tf, n_mt_ohlcv, cnn_lstd_model_x_lengths, batch_size)

    # Load the model and make predictions
    # try:
    t_predictions = load_and_predict(x)
    print("Predictions:", t_predictions)
    plot_mt_predict(x_df, y_df, t_predictions, y_timeframe, 0, base_ohlcv)
    # except Exception as e:
    #     print("An error occurred:", e)
