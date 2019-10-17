import streamlit as st
from relib import hashing
import json
from tensorflow import keras
from modelify_dataset import get_all_data
from checkpointer import checkpoint
from sklearn.decomposition import PCA
import altair as alt
import pandas as pd
import numpy as np
from import_data import handle_trading_pair
from preprocessors import get_x_features
import matplotlib.pyplot as plt


def get_models():
    with open('./param_map.json') as f:
        return json.load(f)

def load_model(model_hash):
    return keras.models.load_model(f'./models/{model_hash}.hdf5')

# def get_layer_names(model_hash):
#     params = get_models()[model_hash]
#     layer_names = []
#     for i in range(params['lstm_depth']):


def predict_sub_model(model_hash, layer_name, data):
    model = load_model(model_hash)
    sub_model = keras.models.Model(
        model.input,
        model.get_layer(layer_name).output
    )
    return sub_model.predict((
        data[f'x_val'],
        data[f'quote_asset_val'],
        data[f'base_asset_val']
    ), verbose=1)


@checkpoint
def get_val_data(lookback):
    data, asset_map = get_all_data(lookback)
    ixs = np.random.choice(len(data['x_val']), 1000, replace=False)
    data = {
        key: value[ixs]
        for key, value
        in data.items()
        if 'val' in key
    }

    return data, asset_map

@checkpoint
def get_x_cols():
    data = handle_trading_pair('ADABNB')
    return get_x_features(data)

def decompose_embeds(embeds):
    return PCA(n_components=2, random_state=42, whiten=True).fit_transform(embeds)

def get_layer_names(model):
    layer_names = []
    i = 0
    while True:
        try:
            layer_name = model.get_layer(index = i).name
            is_flatten = layer_name == 'flatten'
            is_dense = 'dense' in layer_name
            is_last_lstm = layer_name == 'lstm_last_hidden'
            if np.any([is_flatten, is_dense, is_last_lstm]):
                layer_names.append(layer_name)
            i += 1
        except:
            break
    return layer_names


def get_transformed_params(model_hash):
    models = get_models()
    params = models[model_hash]
    for key in params.keys():
        if 'size' in key:
            params[key] = 2**params[key]
        if key == 'lookback':
            params[key] = 15 * (2**params[key])
        if 'reg_power' in key:
            params[key] = 10 ** (-1 * params[key])
    return params


@checkpoint
def get_plot_data(lookback, model_hash, layer_name):
    data, asset_map = get_val_data(params['lookback'])
    embeds = predict_sub_model(model_hash, layer_name, data)
    preds = predict_model(model_hash, data)
    pca_embeds = decompose_embeds(embeds)
    return pd.DataFrame({
        'pca_0': pca_embeds[:,0],
        'pca_1': pca_embeds[:,1],
        'class': data['y_val'],
        **{
            f'pred_{i}': preds[:,i]
            for i in range(preds.shape[1])
        },
    })
@checkpoint
def predict_model(model_hash, data):
    model = load_model(model_hash)
    return model.predict((
        data[f'x_val'],
        data[f'quote_asset_val'],
        data[f'base_asset_val']
    ), verbose=1)

if __name__ == "__main__":
    models = get_models()
    model_hash = st.selectbox(
        'Select model:',
        list(models.keys())
    )
    model = load_model(model_hash)
    params = get_transformed_params(model_hash)
    model.summary(print_fn=st.write)
    st.write(params)
    layer_names = get_layer_names(model)
    layer_name = st.selectbox(
        'Select layer:',
        layer_names
    )
    st.write(layer_name)
    pca_data = get_plot_data(params['lookback'], model_hash, layer_name)
    data, asset_map = get_val_data(params['lookback'])
    x_cols = get_x_cols()
    for c in pca_data['class'].unique():

        class_data = pca_data[pca_data['class']==c]
        plot = alt \
            .Chart(class_data) \
            .mark_circle() \
            .encode(
                x='pca_0',
                y='pca_1',
                color=f'pred_{c}'
            )
        st.altair_chart(plot)
        max_pca_0 = st.slider(
            label='max_pca_0',
            min_value=float(class_data['pca_0'].min()),
            max_value=float(class_data['pca_0'].max()),
            value = float(class_data['pca_0'].max())
        )
        min_pca_0 = st.slider(
            label='min_pca_0',
            min_value=float(class_data['pca_0'].min()),
            max_value=float(class_data['pca_0'].max()),
            value = float(class_data['pca_0'].min())
        )
        max_pca_1 = st.slider(
            label='max_pca_1',
            min_value=float(class_data['pca_1'].min()),
            max_value=float(class_data['pca_1'].max()),
            value = float(class_data['pca_1'].max())
        )
        min_pca_1 = st.slider(
            label='min_pca_1',
            min_value=float(class_data['pca_1'].min()),
            max_value=float(class_data['pca_1'].max()),
            value = float(class_data['pca_1'].min())
        )
        filtered = class_data[
            (class_data['pca_0']>min_pca_0) &
            (class_data['pca_0']<max_pca_0) &
            (class_data['pca_1']>min_pca_1) &
            (class_data['pca_1']<max_pca_1)
        ]
        ixes = np.random.choice(filtered.index, 4, replace=False)
        plt.figure(figsize=(10, 10))
        for (subplot_i, ix) in enumerate(ixes):
            plt.subplot(2, 2, subplot_i+1)
            flt_data = pd.DataFrame({
                'i': range(data['x_val'].shape[1]),
                **{
                    x_col: data['x_val'][ix][:,i]
                    for (i, x_col)
                    in enumerate(x_cols)
                }
            })
            for x_col in x_cols:
                plt.plot(
                    flt_data['i'],
                    flt_data[x_col],
                    label=x_col
                )
        plt.legend()
        st.pyplot()
        plt.close()
