import numpy as np
import pandas as pd


def RMA(values: pd.DataFrame, length):
    # alpha = 1 / length
    # rma = np.zeros_like(values)
    # rma[0] = values[0]
    # for i in range(1, len(values)):
    #     rma[i] = alpha * values[i] + np.nan_to_num((1 - alpha) * rma[i - 1])
    #     pass
    # return rma
    alpha = 1 / length
    rma = pd.DataFrame(values.index, np.nan)  # Initialize with NaN

    # Find the first non-NaN value in the series
    first_valid_index = values.first_valid_index()
    if first_valid_index is None:
        return rma  # Return as all NaN if no valid values

    rma[first_valid_index] = values[first_valid_index]  # Start with the first valid value

    for i in range(first_valid_index + 1, len(values)):
        rma[i] = alpha * values[i] + (1 - alpha) * rma[i - 1]

    return rma
