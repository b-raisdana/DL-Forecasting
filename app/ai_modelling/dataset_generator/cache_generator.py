import logging
import multiprocessing as mp

import pandas as pd

from ai_modelling.dataset_generator.npz_batch import npz_cache_generator


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [CacheGen-%(process)d] %(message)s")


    # Define worker wrapper
    def worker():
        start = pd.to_datetime('03-01-24')
        end = pd.to_datetime('09-01-24')
        npz_cache_generator(start=start, end=end)


    num_workers = 10
    processes = []

    for i in range(num_workers):
        p = mp.Process(target=worker)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
