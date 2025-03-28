import inspect
import sys
import time
from functools import wraps

import numpy as np
import pandas as pd
from colorama import Fore

from .serialization import serialize_it
from ..do_log.log_it import log_d


def profile_it(func):
    @wraps(func)
    def _measure_time(*args, **kwargs):
        start_time = time.time()
        function_parameters = parameters_to_str(args, kwargs)
        log_d(f"{func.__name__}({function_parameters}) started", stack_offset=1)

        # try:
        result = func(*args, **kwargs)
        # except OSError as e:
        #     log_e(f"Current directory is {os.path.abspath(os.path.curdir)}", stack_trace=False)
        #     log_e(f"Error in {func.__name__}({function_parameters}): {str(e)}", stack_trace=True)
        #     raise e
        # except Exception as e:
        #     log(f"Error in {func.__name__}({function_parameters}): {str(e)}", stack_trace=True)
        #     raise

        end_time = time.time()
        execution_time = end_time - start_time
        execution_time_color = Fore.BLUE if execution_time < 0.01 \
            else Fore.GREEN if execution_time < 0.1 \
            else Fore.YELLOW if execution_time < 1 \
            else Fore.RED
        log_d(f"{func.__name__}({function_parameters}) "
              f"executed in {execution_time_color}{execution_time:.3f} seconds", stack_offset=1)
        return result

    return _measure_time


def profile_to_db(func):
    @wraps(func)
    def _to_db_profiling(*args, **kwargs):
        start_time = time.time()
        function_parameters = {
            'args': serialize_it(args),
            'kwargs': serialize_it(kwargs),
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


def parameters_to_str(args, kwargs):
    def process_item(item):
        if isinstance(item, pd.DataFrame):
            return f'{len(item)}*{item.columns}'
        elif isinstance(item, list):
            # return f'list{np.array(item).shape}'
            try:
                return f'list{np.array(item).shape}'
            except ValueError as e:
                nop = 1
                raise e
            except Exception as e:
                nop = 1
                raise e
        elif isinstance(item, np.ndarray):
            return f'ndarray{item.shape}'
        elif isinstance(item, dict):
            # Handle dictionaries
            return process_dict(item)
        else:
            return str(item)

    def process_dict(d):
        t_parameters = []
        for key, value in d.items():
            if isinstance(value, list):
                try:
                    t_parameters.append(f'{key}: list{np.array(value).shape}')
                except ValueError as e:
                    nop = 1
                    raise e
                except Exception as e:
                    nop = 1
                    raise e
            elif isinstance(value, pd.DataFrame):
                t_parameters.append(f'{key}: {len(value)}*{value.columns}')
            elif isinstance(value, np.ndarray):
                t_parameters.append(f'{key}: ndarray{value.shape}')
            elif isinstance(value, dict):
                t_parameters.append(f'{key}: {{ {process_dict(value)} }}')
            else:
                t_parameters.append(f'{key}: {value}')
        return ", ".join(t_parameters)

    parameters = [process_item(arg) for arg in args]
    parameters += [f'{k}: {process_item(kwargs[k])}' for k in kwargs.keys()]

    return ", ".join(parameters)
