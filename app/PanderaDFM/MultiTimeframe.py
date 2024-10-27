from typing import TypeVar

import pandera
from pandera import typing as pt


class MultiTimeframe(pandera.DataFrameModel):
    timeframe: pt.Index[str]


MultiTimeframe_Type = TypeVar('MultiTimeframe_Type', bound=MultiTimeframe)
