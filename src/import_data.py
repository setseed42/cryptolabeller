import pymongo
import pandas as pd
from pprint import pprint
from binance.client import Client
from checkpointer import checkpoint
from tqdm import tqdm
import dask.bag as db
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from preprocessors import preprocessing_pipeline


client = pymongo.MongoClient('mongodb://localhost:27017/')
binance_db = client['binance-data']

def get_trading_pair_data(trading_pair):
    all_data = binance_db[trading_pair].find()
    return pd.DataFrame([
        d for d in all_data
    ])

@checkpoint
def handle_trading_pair(trading_pair):
    print(f'Doing {trading_pair}')
    df = get_trading_pair_data(trading_pair)
    trading_pair_info = get_trading_pair_info()[trading_pair]
    return preprocessing_pipeline(df, trading_pair_info)


@checkpoint
def get_trading_pair_info():
    symbols = Client().get_exchange_info()['symbols']
    return {
        symbol['symbol']: {
            'base_asset': symbol['baseAsset'],
            'quote_asset': symbol['quoteAsset'],
        }
        for symbol in symbols
    }


if __name__ == "__main__":
    trading_pairs = sorted([
        collection['name']
        for collection in
        binance_db.list_collections()
    ])
    trading_pair = 'BATETH'
    print(f'Doing {trading_pair}')
    df = handle_trading_pair(trading_pair)
    print(df.columns)
