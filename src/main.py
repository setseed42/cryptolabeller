from model import train_model
from opt_model import arch_search_space, get_best_params
from import_data import binance_db, handle_trading_pair
from itertools import product
import os
from relib import f

if __name__ == "__main__":
    trading_pairs = sorted([
        collection['name']
        for collection in
        binance_db.list_collections()
    ])
    if not os.path.exists('./models'):
        os.mkdir('./models')
    import dask.bag as db
    from dask.diagnostics import ProgressBar

    trading_pairs = sorted([
        collection['name']
        for collection in
        binance_db.list_collections()
    ])

    with ProgressBar():
        db \
            .from_sequence(trading_pairs[:100]) \
            .map(handle_trading_pair) \
            .map(lambda x: None) \
            .compute()

    search_space = arch_search_space()
    get_best_params(search_space)
