import numpy as np
from tqdm import tqdm
from checkpointer import checkpoint
from preprocessors import get_x_features
from imblearn.under_sampling import RandomUnderSampler
from pprint import pprint
from collections import Counter

@checkpoint(format='bcolz')
def split_shard_dataset(df, lookback=1):
    x_features = get_x_features(df)

    def handle_y(which):
        y = shard_data(
            df[df['split'] == which]['label'].values,
            lookback, is_y=True
        )
        y = y.reshape(-1).astype(int) + 1
        y_new = np.zeros((len(y), 3))
        for (i, j) in enumerate(y):
            y_new[i, j] = 1
        return y_new

    model_x = shard_data(df[df['split'] == 'train'][x_features].values, lookback)
    model_y = handle_y('train')
    train_ix = int(len(model_x)*0.7)
    x_test = shard_data(df[df['split'] == 'test'][x_features].values, lookback)
    y_test = handle_y('test')
    pprint('Class balance before under-sample')
    pprint(get_class_dist(model_y))
    x_train, y_train = under_sample(
        model_x[:train_ix],
        model_y[:train_ix],
    )
    x_val, y_val = under_sample(
        model_x[train_ix:],
        model_y[train_ix:],
    )
    return {
        'x_train': x_train,
        'x_test': x_test,
        'x_val': x_val,
        'y_train': y_train,
        'y_test': y_test,
        'y_val': y_val,
    }

def shard_data(data, lookback, is_y=False):
    used_ixs = range(lookback, len(data))

    def get_shard(index):
        shard = data[index-lookback:index]
        if is_y:
            shard = shard[-1]
        return np.expand_dims(shard, axis=0)
    print('Sharding...')
    return np.concatenate([
        get_shard(index)
        for index in tqdm(used_ixs)
    ])

def under_sample(x, y):
    under_sampler = RandomUnderSampler(random_state=42, return_indices=True)
    under_sampler.fit_resample(
        np.concatenate([i[-1:] for i in x]),
        np.argmax(y, axis=1),
    )
    indexes = under_sampler.sample_indices_
    n = len(x)
    dropped_data = (n - len(indexes))*100/n
    print(f'Imbalanced learn -- Dropped {dropped_data} percent of data')
    return x[indexes], y[indexes]




def get_class_dist(y):
    counts = Counter(np.argmax(y, axis=1))
    return {
        key: value/len(y)
        for key, value
        in dict(counts).items()
    }


