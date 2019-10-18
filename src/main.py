from model import train_model
from opt_model import arch_search_space, get_best_params
from import_data import binance_db, handle_trading_pair, get_trading_pair_info
from itertools import product
import os
from relib import f

if __name__ == "__main__":
    if not os.path.exists('./models'):
        os.mkdir('./models')
    import dask.bag as db
    from dask.diagnostics import ProgressBar

    trading_pair_info = get_trading_pair_info()
    trading_pairs = sorted([
        collection['name']
        for collection in
        binance_db.list_collections()
        for collection['name']
        in trading_pair_info.keys()
    ])

    with ProgressBar():
        db \
            .from_sequence(trading_pairs[:100]) \
            .map(handle_trading_pair) \
            .map(lambda x: None) \
            .compute()

    # search_space = arch_search_space()
    # get_best_params(search_space)
    train_model({
        "dense_depth":1,
        "dense_dropout_p":0.383395377969626,
        "dense_hidden_size":9,
        "dense_reg_power":4,
        "embedding_size":5,
        "lookback":3,
        "lstm_depth":1,
        "lstm_dropout_p":0.06915426913928757,
        "lstm_hidden_size":4,
        "lstm_last_dropout_p":0.43206395773247525,
        "lstm_last_hidden_size":9,
        "lstm_last_reg_power":3,
        "lstm_reg_power":5,
        "output_reg_power":5,
    })
