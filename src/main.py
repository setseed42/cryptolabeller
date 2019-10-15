from model import train_model
from import_data import binance_db
from itertools import product
import os
from relib import f

def train_lookback_model(trading_pair, lookback, model):
    return [
        train_model(
            trading_pair,
            lookback=lookback,
            model_name=model,
            model_params={
                'hidden_size': 64, 'depth': 0
            }
        ),
        train_model(
            trading_pair,
            lookback=lookback,
            model_name=model,
            model_params={
                'hidden_size': 64, 'depth': 1
            }
        ),
        train_model(
            trading_pair,
            lookback=lookback,
            model_name=model,
            model_params={
                'hidden_size': 128, 'depth': 1
            }
        ),
        train_model(
            trading_pair,
            lookback=lookback,
            model_name=model,
            model_params={
                'hidden_size': 128, 'depth': 2
            }
        )
    ]

if __name__ == "__main__":
    trading_pairs = sorted([
        collection['name']
        for collection in
        binance_db.list_collections()
    ])
    if not os.path.exists('./models'):
        os.mkdir('./models')
    trading_pair = 'BATETH'
    print(f'Doing {trading_pair}')
    lookbacks = [1, 15]
    models = ['mlp', 'lstm']
    model_paths = f.flatten([
        train_lookback_model(trading_pair, lookback, model)
        for lookback, model
        in product(lookbacks, models)
        if not (lookback==1 and model=='lstm')
    ])
