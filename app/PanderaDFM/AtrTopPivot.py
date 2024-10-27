from datetime import datetime
from typing import Annotated

import pandas as pd
import pytz
from pandas import Timestamp
from pandera import typing as pt

from app.PanderaDFM.ExtendedDf import ExtendedDf
from app.PanderaDFM.Pivot2 import MultiTimeframePivot2DFM, Pivot2DFM


class AtrMovementPivotDFM(Pivot2DFM):
    movement_start_time: pt.Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]]
    return_end_time: pt.Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]]
    movement_start_value: pt.Series[float]
    return_end_value: pt.Series[float]


class AtrMovementPivotDf(ExtendedDf):
    schema_data_frame_model = AtrMovementPivotDFM


_sample_df = pd.DataFrame({
    'date': [Timestamp(datetime(year=1980, month=1, day=1, hour=1, minute=1, second=1).replace(tzinfo=pytz.UTC))],
    'movement_start_time': \
        [Timestamp(datetime(year=1980, month=1, day=1, hour=1, minute=1, second=1).replace(tzinfo=pytz.UTC))],
    'return_end_time': \
        [Timestamp(datetime(year=1980, month=1, day=1, hour=1, minute=1, second=1).replace(tzinfo=pytz.UTC))],
    'movement_start_value': [0.0],
    'is_resistance': [False],
    'return_end_value': [0.0],
    'level': [0.0],
    'internal_margin': [0.0],
    'external_margin': [0.0],
    'original_start': \
        [Timestamp(datetime(year=1980, month=1, day=1, hour=1, minute=1, second=1).replace(tzinfo=pytz.UTC))],
    'ttl': [Timestamp(datetime(year=1980, month=1, day=1, hour=1, minute=1, second=1).replace(tzinfo=pytz.UTC))],
    'major_timeframe': [False],
})
AtrMovementPivotDf._sample_df = _sample_df.set_index(['date', 'original_start'])


class MultiTimeframeAtrMovementPivotDFM(AtrMovementPivotDFM, MultiTimeframePivot2DFM):
    pass


class MultiTimeframeAtrMovementPivotDf(ExtendedDf):
    schema_data_frame_model = MultiTimeframeAtrMovementPivotDFM

_sample_df = pd.DataFrame({
    'date': [Timestamp(datetime(year=1980, month=1, day=1, hour=1, minute=1, second=1).replace(tzinfo=pytz.UTC))],
    'timeframe': ['sample'],
    'movement_start_time': \
        [Timestamp(datetime(year=1980, month=1, day=1, hour=1, minute=1, second=1).replace(tzinfo=pytz.UTC))],
    'return_end_time': \
        [Timestamp(datetime(year=1980, month=1, day=1, hour=1, minute=1, second=1).replace(tzinfo=pytz.UTC))],
    'movement_start_value': [0.0],
    'is_resistance': [False],
    'return_end_value': [0.0],
    'level': [0.0],
    'internal_margin': [0.0],
    'external_margin': [0.0],
    'original_start': \
        [Timestamp(datetime(year=1980, month=1, day=1, hour=1, minute=1, second=1).replace(tzinfo=pytz.UTC))],
    'ttl': [Timestamp(datetime(year=1980, month=1, day=1, hour=1, minute=1, second=1).replace(tzinfo=pytz.UTC))],
    'major_timeframe': [False],
})
MultiTimeframeAtrMovementPivotDf._sample_df = _sample_df.set_index(['timeframe', 'date', 'original_start'])

