from tensorflow import keras
from relib import hashing
from pprint import pprint
import datetime
import numpy as np
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from itertools import product
from import_data import handle_trading_pair
from modelify_dataset import get_all_data
from tensorboard.plugins.hparams import api as hp

def train_model(params):
    data, asset_map = get_all_data(15 * (2**params['lookback']))
    for split in ['train', 'test', 'val']:
        y_key = f'y_{split}'
        data[y_key] = keras.utils.to_categorical(data[y_key], num_classes=3).astype(int)
    model = model_arch(
        params,
        data['x_train'].shape[2],
        data['x_train'].shape[1],
        len(asset_map),
    )
    model.summary()
    model_metadata = '-'.join([
            f'{key}-{str(np.round(value,3))}'
            for key, value
            in sorted(params.items())
    ])
    model_hash = hashing.hash(model_metadata)
    log_dir = "logs/finance/{}-{}".format(
        model_hash,
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    model_path = f'./models/{model_hash}.hdf5'
    def get_x(split):
        return data[f'x_{split}'], data[f'quote_asset_{split}'], data[f'base_asset_{split}']
    history = model.fit(
        get_x('train'), data['y_train'],
        shuffle=True,
        validation_data=(
            get_x('val'),
            data['y_val']
        ),
        epochs=10**10,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                patience=5,
                factor=0.02
            ),
            keras.callbacks.ModelCheckpoint(
                monitor='val_loss',
                save_best_only=True,
                filepath=model_path
            ),
            keras.callbacks.TensorBoard(log_dir=log_dir),
            hp.KerasCallback(log_dir, params)
        ]
    )
    keras.backend.clear_session()
    return history.history

def model_arch(params, n_features, lookback, n_assets, n_labels=3):
    input_layer, quote_input, base_input = get_input_layers(n_features, lookback)
    asset_embeddings = get_embeddings(quote_input, base_input, n_assets, emb_size=2**params['embedding_size'])
    x = input_layer
    def get_reg_power(reg_power):
        return keras.regularizers.l2(10**(-1 * params[reg_power]))
    for i in range(params['lstm_depth']):
        x = keras.layers.LSTM(
            2**params['lstm_hidden_size'],
            return_sequences=True,
            name=f'lstm_hidden_{i}',
            kernel_regularizer=get_reg_power('lstm_reg_power')
        )(x)
        x = keras.layers.Dropout(params['lstm_dropout_p'])(x)
    x = keras.layers.LSTM(
        2**params['lstm_last_hidden_size'],
        name='lstm_last_hidden',
        kernel_regularizer=get_reg_power('lstm_last_reg_power')
    )(x)
    x = keras.layers.Dropout(params['lstm_last_dropout_p'])(x)
    x = keras.layers.Concatenate()([x, asset_embeddings])
    for i in range(params['dense_depth']):
        x = keras.layers.Dense(
                2**params[f'dense_hidden_size'],
                activation='relu',
                name=f'dense_hidden_{i}',
                kernel_regularizer=get_reg_power('dense_reg_power')
            )(x)
        x = keras.layers.Dropout(params['dense_dropout_p'])(x)
    output_layer = keras.layers.Dense(
        n_labels,
        activation='softmax',
        kernel_regularizer=get_reg_power('output_reg_power')
    )(x)
    model = keras.models.Model([input_layer, quote_input, base_input], output_layer)
    model.compile(
        optimizer=keras.optimizers.RMSprop(clipvalue=5),
        loss='categorical_crossentropy',
        metrics=[
            keras.metrics.AUC(),
            keras.metrics.Precision(),
            keras.metrics.Recall(),
            keras.metrics.CategoricalAccuracy()
        ]
    )
    return model

def get_input_layers(n_features, lookback):
    input_layer = keras.layers.Input(shape=(lookback, n_features,), name='ts_input')
    quote_input = keras.layers.Input(shape=(1, ), name='quote_input')
    base_input = keras.layers.Input(shape=(1, ), name='base_input')
    return input_layer, quote_input, base_input


def get_embeddings(quote_input, base_input, n_assets, emb_size=16):
    asset_embedding = keras.layers.Embedding(n_assets, emb_size)
    asset_embeddings = keras.layers.Concatenate()([
        asset_embedding(quote_input),
        asset_embedding(base_input)
    ])
    return keras.layers.Flatten()(asset_embeddings)


def get_hidden_output(model_path):
    model = keras.models.load_model(model_path)
    layer_name = 'last_hidden'
    intermediate_layer_model = keras.models.Model(
        inputs=model.input,
        outputs=model.get_layer(layer_name).output
    )
    return intermediate_layer_model

def get_hidden_by_name(name):
    models = {
        'mlp': mlp_arch,
        'lstm': lstm_arch,
    }
    return models[name]
