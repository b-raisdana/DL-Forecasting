#
# def active_tops(multi_timeframe_peaks_n_valleys: pt.DataFrame[PeaksValleys]):
#     _levels = pd.DataFrame(columns=['end', 'hits', 'atr', 'breakout', 'margin', 'width'], index=['timeframe', 'date'])
#     for timeframe in config.timeframes[2::-1]:
#         timeframe_tops = major_peaks_n_valleys(multi_timeframe_peaks_n_valleys, timeframe)
#         overlapping_peaks_n_valleys = []
#         for _index, _top in timeframe_tops.iterrows():
#             coverage_start = to_timeframe(_index[1], timeframe)
#             coverage_end = coverage_start + pd.to_timedelta(                    timeframe)
#             _indexes_mask = \
#                 (multi_timeframe_peaks_n_valleys.index.get_level_values('date') >= coverage_start) \
#                 & (multi_timeframe_peaks_n_valleys.index.get_level_values('date') <= coverage_end)
#             _indexes = multi_timeframe_peaks_n_valleys[_indexes_mask]
#             overlapping_peaks_n_valleys.append(_indexes)
#             # if level_index != to_timeframe(level_index[1], timeframe):
#             #     raise Exception(f'{level_index[1]} expected to be aligned with {timeframe} timeframe!')
#         multi_timeframe_peaks_n_valleys.drop(list(set(overlapping_peaks_n_valleys)))
