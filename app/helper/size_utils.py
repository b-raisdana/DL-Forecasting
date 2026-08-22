import sys

import numpy as np


def _children_size(obj: object, seen: set[int]) -> int:
    if isinstance(obj, dict):
        return sum(get_size(k, seen) + get_size(v, seen) for k, v in obj.items())
    if isinstance(obj, (list, tuple, set)):
        return sum(get_size(i, seen) for i in obj)
    if hasattr(obj, "__dict__"):
        return get_size(obj.__dict__, seen)
    return 0


def get_size(obj: object, seen: set[int] | None = None) -> int:
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    try:
        size = sys.getsizeof(obj)
    except TypeError:
        return 0
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    size += _children_size(obj, seen)
    if obj in [None, Ellipsis, NotImplemented]:
        return sys.getsizeof(obj)
    return size
