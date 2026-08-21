import random
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, jensenshannon
from scipy.stats import wasserstein_distance, ks_2samp

from Config import app_config
from ai_modelling.base import overlapped_quarters, master_x_shape
from ai_modelling.dataset_generator.training_datasets import train_data_of_mt_n_profit
from data_processing.ohlcv import read_multi_timeframe_ohlcv
from helper.br_py.br_py.do_log import log_i, log_d
from helper.functions import date_range_to_string


def compare_columns_similarity(X_df: pd.DataFrame):
    results = []
    features = [col for col in X_df.columns if col not in ['symbol', 'timeframe']]
    symbols = X_df['symbol'].unique()

    # Precompute describe stats per symbol
    stat_df = (
        X_df.groupby('symbol')[features]
        .describe(percentiles=[0.25, 0.5, 0.75])
        .unstack()
    )
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None, 'display.precision', 3, 'display.float_format', '{:.3f}'.format):
        print(stat_df)
    for feat in features:
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i + 1:]:
                data1 = X_df[X_df['symbol'] == sym1][feat].dropna()
                data2 = X_df[X_df['symbol'] == sym2][feat].dropna()

                if len(data1) < 50 or len(data2) < 50:
                    continue  # Skip small samples

                # Histograms for JS and cosine
                hist1, _ = np.histogram(data1, bins=50, density=True)
                hist2, _ = np.histogram(data2, bins=50, density=True)

                hist1 = hist1 / hist1.sum()
                hist2 = hist2 / hist2.sum()

                js_div = jensenshannon(hist1, hist2)
                cos_sim = 1 - cosine(hist1, hist2)
                w_dist = wasserstein_distance(data1, data2)
                ks_stat, ks_pval = ks_2samp(data1, data2)

                # Compute absolute diff of stats
                stat_diff = {
                    stat: abs(stat_df.loc[(feat, stat, sym1)] - stat_df.loc[(feat, stat, sym2)])
                    for stat in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
                }

                results.append({
                    'feature': feat,
                    'symbol_1': sym1,
                    'symbol_2': sym2,
                    'js_divergence': js_div,
                    'cosine_similarity': cos_sim,
                    'wasserstein_distance': w_dist,
                    'ks_statistic': ks_stat,
                    'ks_pvalue': ks_pval,
                    **{f'diff_{k}': v for k, v in stat_diff.items()}
                })

    df_results = pd.DataFrame(results)
    print(df_results.sort_values(by='js_divergence'))
    return df_results

def summarize_feature_similarity(comparison: pd.DataFrame):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None, 'display.precision', 3, 'display.float_format', '{:.3f}'.format):
        # Sort by Jensen-Shannon divergence (lower means more similar)
        print("\n=== Most Similar Features (Lowest JS Divergence) ===")
        print(comparison.sort_values(by='js_divergence').head(20))

        # Sort by Jensen-Shannon divergence descending (highest means least similar)
        print("\n=== Least Similar Features (Highest JS Divergence) ===")
        print(comparison.sort_values(by='js_divergence', ascending=False).head(20))

        # You can also do the same for Wasserstein distance if you want:
        print("\n=== Most Similar Features (Lowest Wasserstein Distance) ===")
        print(comparison.sort_values(by='wasserstein_distance').head(20))

        print("\n=== Least Similar Features (Highest Wasserstein Distance) ===")
        print(comparison.sort_values(by='wasserstein_distance', ascending=False).head(20))

        # And similarly for cosine similarity (higher means more similar)
        print("\n=== Most Similar Features (Highest Cosine Similarity) ===")
        print(comparison.sort_values(by='cosine_similarity', ascending=False).head(20))

        print("\n=== Least Similar Features (Lowest Cosine Similarity) ===")
        print(comparison.sort_values(by='cosine_similarity').head(20))

def summarize_symbol_similarity(df_results: pd.DataFrame, top_n=5):
    sim_summary = []

    grouped = df_results.groupby(['symbol_1', 'symbol_2'])[
        ['js_divergence', 'cosine_similarity', 'wasserstein_distance', 'ks_statistic']].mean().reset_index()
    grouped['similarity_score'] = (
            (1 - grouped['js_divergence']) * 0.4 +
            grouped['cosine_similarity'] * 0.3 +
            (1 - grouped['wasserstein_distance'] / grouped['wasserstein_distance'].max()) * 0.2 +
            (1 - grouped['ks_statistic']) * 0.1
    )

    top_similar = grouped.sort_values(by='similarity_score', ascending=False).head(top_n)
    least_similar = grouped.sort_values(by='similarity_score', ascending=True).head(top_n)

    print("\n💚 Most Similar Symbol Pairs:")
    print(top_similar[['symbol_1', 'symbol_2', 'similarity_score']])

    print("\n💔 Least Similar Symbol Pairs:")
    print(least_similar[['symbol_1', 'symbol_2', 'similarity_score']])


def pairs_stat_compare(
        start: datetime, end: datetime,
        batch_size: int = 100,
        number_of_batches= 100,
        forecast_trigger_bars: int = 3 * 4 * 4 * 4 * 1,
        verbose: bool = True):
    X_df = None
    y_df = None
    pd.set_option('display.precision', 3)

    quarters = overlapped_quarters(date_range_to_string(start=start, end=end))
    log_i("pairs_stat_compare started")

    random.shuffle(quarters)
    for q_start, q_end in quarters:
        if verbose: log_d(f'quarter {q_start} → {q_end}')
        app_config.processing_date_range = date_range_to_string(start=q_start, end=q_end)
        symbol_list = [
            'BNBUSDT',
            'BTCUSDT',
            'EOSUSDT',
            'ETHUSDT',
            'SOLUSDT',
            'TRXUSDT']
        # random.shuffle(symbol_list)
        for symbol in symbol_list:
            if verbose: log_d(f'Symbol {symbol}')
            app_config.under_process_symbol = symbol
            mt_ohlcv = read_multi_timeframe_ohlcv(app_config.processing_date_range)
            for _ in range(number_of_batches):

                _, _, x_dfs, y_dfs, *_ = train_data_of_mt_n_profit(
                    structure_tf='4h',
                    mt_ohlcv=mt_ohlcv,
                    x_shape=master_x_shape,
                    batch_size=batch_size,
                    dataset_batches=1,
                    forecast_trigger_bars=forecast_trigger_bars,
                    verbose=False)
                for k in x_dfs.keys():
                    x_dfs[k]['timeframe'] = k
                x_dfs = pd.concat(x_dfs)
                x_dfs['symbol'] = symbol
                y_dfs = pd.concat(y_dfs)
                y_dfs['symbol'] = symbol
                if X_df is None:
                    X_df = x_dfs
                    y_df = y_dfs
                else:
                    X_df = pd.concat([X_df, x_dfs])
                    y_df = pd.concat([y_df, y_dfs])
                if verbose: log_d(
                    f'put {symbol} batch for {app_config.processing_date_range} (size={len(y_df)})')
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None, 'display.precision', 3, 'display.float_format', '{:.3f}'.format):
        print("X_df.describe():", X_df.describe())
        print("\ny_df.describe():", y_df.describe())
        comparison = compare_columns_similarity(X_df)
        pd.set_option('display.precision', 3)
        print("Comparison:", comparison)
    summarize_symbol_similarity(comparison)
    summarize_feature_similarity(comparison)


if __name__ == '__main__':
    for i in range(5):
        pairs_stat_compare(
            start=pd.to_datetime('2024-03-01'),
            end=pd.to_datetime('2024-09-01'),
            batch_size=10,
            number_of_batches=10,
            verbose=False
        )
