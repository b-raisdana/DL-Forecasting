import json
import base64
import datetime
import uuid
import numpy as np
import pandas as pd
from decimal import Decimal
from fractions import Fraction
from collections import deque, Counter


def serialize_it(obj):
    """Convert complex Python objects to JSONB-compatible format."""
    if isinstance(obj, set):
        return {"__set__": list(obj)}
    if isinstance(obj, tuple):
        return {"__tuple__": list(obj)}
    if isinstance(obj, frozenset):
        return {"__frozenset__": list(obj)}
    if isinstance(obj, pd.DataFrame):
        return {"__dataframe__": obj.to_dict()}
    if isinstance(obj, pd.Series):
        return {"__series__": obj.to_dict()}
    if isinstance(obj, pd.Index):
        return {"__index__": list(obj)}
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": obj.tolist()}
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.complex128):
        return {"__complex__": [obj.real, obj.imag]}
    if isinstance(obj, datetime.datetime):
        return {"__datetime__": obj.isoformat()}
    if isinstance(obj, datetime.date):
        return {"__date__": obj.isoformat()}
    if isinstance(obj, datetime.time):
        return {"__time__": obj.isoformat()}
    if isinstance(obj, datetime.timedelta):
        return {"__timedelta__": obj.total_seconds()}
    if isinstance(obj, uuid.UUID):
        return {"__uuid__": str(obj)}
    if isinstance(obj, Decimal):
        return {"__decimal__": str(obj)}
    if isinstance(obj, Fraction):
        return {"__fraction__": [obj.numerator, obj.denominator]}
    if isinstance(obj, deque):
        return {"__deque__": list(obj)}
    if isinstance(obj, Counter):
        return {"__counter__": obj.most_common()}
    if isinstance(obj, bytes):
        return {"__bytes__": base64.b64encode(obj).decode()}

    return obj  # Default case