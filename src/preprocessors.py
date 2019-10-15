import numpy as np
from pandas_flavor import register_dataframe_method as rdm
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

def preprocessing_pipeline(df, trading_pair_info):
    return df \
        .initial_preprocess() \
        .add_tp_data(trading_pair_info) \
        .reduce_mem_usage() \
        .filter_features() \
        .add_dt_index() \
        .label_data() \
        .split_preprocess_df() \
        .reduce_mem_usage()

@rdm
def initial_preprocess(df):
    df = df.drop('_id', axis=1)
    columns = df.columns
    already_formatted = [
        'opentime',
        'closetime',
        'tradenum'
    ]
    for col in columns:
        if col in already_formatted:
            continue
        df[col] = df[col].astype(float)
    return df

@rdm
def add_tp_data(df, trading_pair_info):
    for key in trading_pair_info:
        df[key] = trading_pair_info[key]
    return df


#UNUSED goes after add_tp_data
@rdm
def shotgun_analysis(df):
    import talib
    from talib import abstract
    keys = [
        'open', 'high', 'low', 'close', 'volume'
    ]
    df = df.sort_values('opentime')
    inputs = {
        key: df[key].values
        for key in keys
    }
    for func_name in talib.get_functions():
        function = abstract.Function(func_name)
        try:
            processed = function(inputs)
        except:
            processed = None
        if isinstance(processed, np.ndarray):
            df[func_name] = processed
        elif processed is None:
            print(f'Could not do {func_name}')
        else:
            assert isinstance(processed, list)
            for (i, l) in enumerate(processed):
                df[f'{func_name}_{i}'] = l
    return df

# https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
@rdm
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtype
        if pd.api.types.is_datetime64_any_dtype(col_type):
            continue
        if col_type not in [object, pd.Timestamp]:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
       # else:
        #    df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(
        100 * (start_mem - end_mem) / start_mem))
    return df


@rdm
def filter_features(df):
    ts_features = [
        col for col in df.columns
        if col.lower() == col
        and col not in [
            'base_asset', 'quote_asset', 'opentime', 'closetime'
        ]
    ]
    return df[['closetime', *ts_features]]


@rdm
def add_dt_index(df):
    dt_func = lambda x: datetime.fromtimestamp(x/1000)
    df.index = df['closetime'].apply(dt_func)
    return df

#https://towardsdatascience.com/financial-machine-learning-part-1-labels-7eeed050f32e
@rdm
def label_data(df):
    print('Getting vol')
    df = df.assign(threshold=get_vol(df.close)).dropna()
    print('Getting_horizons')
    df = df.assign(t1=get_horizons(df)).dropna()
    events = df[['t1', 'threshold']]
    events = events.assign(side=pd.Series(1., events.index))  # long only
    print('Getting touches')
    touches = get_touches(df.close, events, [1, 1])
    print('Getting labels')
    touches = get_labels(touches)
    return df.assign(label=touches.label.astype(int))

@rdm
def split_preprocess_df(df, train_size=0.7, stadardize=True):
    x_features = get_x_features(df)
    df['split'] = np.where(np.arange(len(df)) < int(
        len(df)*train_size), 'train', 'test')
    if stadardize:
        scaler = StandardScaler().fit(df[df['split'] == 'train'][x_features])
        df[x_features] = scaler.transform(df[x_features])
    return df

def get_x_features(df):
    return [
        col for col in df.columns
        if col not in ['closetime', 'threshold', 't1', 'label', 'split']
    ]

def get_vol(prices, span=100, delta=pd.Timedelta(hours=1)):
    df0 = prices.index.searchsorted(prices.index - delta)
    df0 = df0[df0 > 0]
    df0 = pd.Series(prices.index[df0-1],
                    index=prices.index[prices.shape[0]-df0.shape[0]:])
    df0 = prices.loc[df0.index] / prices.loc[df0.values].values - 1
    df0 = df0.ewm(span=span).std()
    return df0


def get_horizons(prices, delta=pd.Timedelta(minutes=15)):
    t1 = prices.index.searchsorted(prices.index + delta)
    t1 = t1[t1 < prices.shape[0]]
    t1 = prices.index[t1]
    t1 = pd.Series(t1, index=prices.index[:t1.shape[0]])
    return t1


def get_touches(prices, events, factors=[1, 1]):
    out = events[['t1']].copy(deep=True)
    if factors[0] > 0:
        thresh_uppr = factors[0] * events['threshold']
    else:
        thresh_uppr = pd.Series(index=events.index)  # no uppr thresh
    if factors[1] > 0:
        thresh_lwr = -factors[1] * events['threshold']
    else:
        thresh_lwr = pd.Series(index=events.index)  # no lwr thresh
    for loc, t1 in tqdm(events['t1'].iteritems()):
        df0 = prices[loc:t1]                              # path prices
        df0 = (df0 / prices[loc] - 1) * events.side[loc]  # path returns
        out.loc[loc, 'stop_loss'] = \
            df0[df0 < thresh_lwr[loc]].index.min()  # earliest stop loss
        out.loc[loc, 'take_profit'] = \
            df0[df0 > thresh_uppr[loc]].index.min()  # earliest take profit
    return out


def get_labels(touches):
    out = touches.copy(deep=True)
    first_touch = touches[['stop_loss', 'take_profit']].min(axis=1)
    for loc, t in first_touch.iteritems():
        if pd.isnull(t):
            out.loc[loc, 'label'] = 0
        elif t == touches.loc[loc, 'stop_loss']:
            out.loc[loc, 'label'] = -1
        else:
            out.loc[loc, 'label'] = 1
    return out





