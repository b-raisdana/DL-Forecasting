from typing import Annotated

import pandas as pd
import pandera
from archive_not_used_trash.domain.schemas.market_structure.Pivot import PivotDFM
from pandera import typing as pt


class BullBearSidePivot(PivotDFM):
    movement_start_time: pt.Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]] = pandera.Field(nullable=True)
    movement_start_value: pt.Series[float] = pandera.Field(nullable=True)
    return_end_time: pt.Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]] = pandera.Field(nullable=True)
    return_end_value: pt.Series[float] = pandera.Field(nullable=True)
