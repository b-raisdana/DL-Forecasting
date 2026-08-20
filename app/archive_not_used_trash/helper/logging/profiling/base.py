import inspect
import sys
import time
from collections.abc import Callable
from functools import wraps

import numpy as np
import pandas as pd
from colorama import Fore

from archive_not_used_trash.helper.logging.profiling.serialization import serialize_it

from ..do_log.log_it import log_d


def profile_to_db(func):
    @wraps(func)
    def _to_db_profiling(*args, **kwargs):
        start_time = time.time()
        function_parameters = {
            "args": serialize_it(args),
            "kwargs": serialize_it(kwargs),
        }
        log_d(f"{func.__name__}({function_parameters}) started", stack_offset=1)
        result = func(*args, **kwargs)
        return result

    return _to_db_profiling


__global_profile_to_db = False


def init_global_profile_to_db():
    global __global_profile_to_db
    __global_profile_to_db = True
    sys.setprofile(profile_func)


def profile_func(frame, event, arg):
    if event == "call":
        func_name = frame.f_code.co_name
        func = frame.f_globals.get(func_name)  # Get the function object

        if func and callable(func):  # Ensure it's a function
            try:
                sig = inspect.signature(func)
                bound_args = sig.bind(*frame.f_locals.values())  # Full binding of args and kwargs
                bound_args.apply_defaults()  # Ensure default values are included

                print(f"Calling {func_name} with arguments:")
                for name, value in bound_args.arguments.items():
                    print(f"  {name}: {value}")

            except TypeError:
                # If function signature binding fails, still print arguments manually
                print(f"Calling {func_name} with raw arguments: {frame.f_locals}")

    elif event == "return":
        print(f"Function {frame.f_code.co_name} returned: {arg}")


# def new_profile_it(func):
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         profiler = cProfile.Profile()
#         profiler.enable()
#         result = func(*args, **kwargs)
#         profiler.disable()
#
#         s = StringIO()
#         ps = pstats.Stats(profiler, stream=s).sort_stats('cumtime')
#         ps.print_stats()
#         print(s.getvalue())  # Or log the output
#         return result
#
#     return wrapper


