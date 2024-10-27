from datetime import datetime
from typing import Annotated

import pandas as pd
import pandera
import pytz
from pandas import Timestamp
from pandera import typing as pt

from app.PanderaDFM.ExtendedDf import ExtendedDf, BaseDFM
from app.PanderaDFM.MultiTimeframe import MultiTimeframe


class PivotDFM(BaseDFM):
    date: pt.Index[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]] = pandera.Field(
        check_name=True)  # the original time of creating pivot
    original_start: pt.Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]]  # this part activated at (passing time)

    level: pt.Series[float]
    is_resistance: pt.Series[bool]
    internal_margin: pt.Series[float]
    external_margin: pt.Series[float]
    ttl: pt.Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]]  # = pandera.Field(nullable=True)
    passing_time: pt.Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]] = pandera.Field(nullable=True)
    deactivated_at: pt.Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]] = pandera.Field(nullable=True)
    # hit: pt.Series[int] = pandera.Field(nullable=True)
    hit: pt.Series[pd.Int8Dtype] = pandera.Field(nullable=True)
    # the master pivot which this pivot is overlapping with
    master_pivot_timeframe: pt.Series[str] = pandera.Field(nullable=True)
    master_pivot_date: pt.Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]] = pandera.Field(nullable=True)

    # ftc_base_pattern_timeframes: pt.Series[List[str]] = pandera.Field(nullable=True)
    # ftc_base_pattern_dates: pt.Series[List[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]]] = pandera.Field(nullable=True)

    @staticmethod
    def description(start_time: datetime, pivot_timeframe: str, pivot_info) -> str:
        output = (f"Pivot {pivot_info['level']:.0f}={pivot_timeframe}@{start_time.strftime('%m/%d.%H:%M')}"
                  f"[{pivot_info['internal_margin']:.0f}-{pivot_info['external_margin']:.0f}](")

        if hasattr(pivot_info, 'movement_start_value'):
            output += f"M{abs(pivot_info['movement_start_value'] - pivot_info['level']):.0f}"
        if hasattr(pivot_info, 'return_end_value'):
            output += f"R{abs(pivot_info['return_end_value'] - pivot_info['level']):.0f}"
        output += f"H{pivot_info['hit']}"
        if pd.notna(pivot_info['master_pivot_timeframe']):
            output += f"O{pivot_info['master_pivot_timeframe']}/{pivot_info['master_pivot_date']}"
        return output

    @staticmethod
    def name(start_time: datetime, pivot_timeframe: str, pivot_info) -> str:
        output = f"Pivot {pivot_info['level']:.0f}={pivot_timeframe}@{start_time.strftime('%m/%d.%H:%M')}"
        if pd.notna(pivot_info['master_pivot_timeframe']):
            output += f"O{pivot_info['master_pivot_timeframe']}/{pivot_info['master_pivot_date']}"
        return output


class PivotDf(ExtendedDf):
    schema_data_frame_model = PivotDFM


_sample_df = pd.DataFrame({
    'date': [Timestamp(datetime(year=1980, month=1, day=1, hour=1, minute=1, second=1).replace(tzinfo=pytz.UTC))],
    'level': [0.0],
    'is_resistance': [False],
    'internal_margin': [0.0],
    'external_margin': [0.0],
    'original_start': [
        Timestamp(datetime(year=1980, month=1, day=1, hour=1, minute=1, second=1).replace(tzinfo=pytz.UTC))],
    'ttl': [Timestamp(datetime(year=1980, month=1, day=1, hour=1, minute=1, second=1).replace(tzinfo=pytz.UTC))],
})
PivotDf._sample_df = _sample_df.set_index(['date'])


class MultiTimeframePivotDFM(PivotDFM, MultiTimeframe):
    pass


class MultiTimeframePivotDf(ExtendedDf):
    schema_data_frame_model = MultiTimeframePivotDFM


_sample_df = pd.DataFrame({
    'date': [Timestamp(datetime(year=1980, month=1, day=1, hour=1, minute=1, second=1).replace(tzinfo=pytz.UTC))],
    'timeframe': ['test'],
    'level': [0.0],
    'is_resistance': [False],
    'internal_margin': [0.0],
    'external_margin': [0.0],
    'original_start': [
        Timestamp(datetime(year=1980, month=1, day=1, hour=1, minute=1, second=1).replace(tzinfo=pytz.UTC))],
    'ttl': [Timestamp(datetime(year=1980, month=1, day=1, hour=1, minute=1, second=1).replace(tzinfo=pytz.UTC))],
})
MultiTimeframePivotDf._sample_df = _sample_df.set_index(['date', 'timeframe'])
