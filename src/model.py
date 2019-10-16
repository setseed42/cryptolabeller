from tensorflow import keras
from relib import hashing
from pprint import pprint
import datetime
import numpy as np
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from itertools import product
from import_data import handle_trading_pair
from modelify_dataset import split_shard_dataset

def train_model(trading_pair, lookback, model_name, model_params={}):
    print(f'Training {trading_pair} with {model_name}')
    print(f'Parameters: ')
    print(f'Lookback: {lookback}')
    pprint(model_params)
    data = handle_trading_pair(trading_pair)
    data = split_shard_dataset(data, lookback)
    model = get_model_by_name(model_name)(
        data['x_train'].shape[2],
        data['x_train'].shape[1],
        **model_params
    )
    model.summary()
    model_metadata = [
        trading_pair,
        model_name,
        str(lookback),
        '-'.join([
            f'{key}-{value}'
            for key, value
            in model_params.items()
        ])
    ]
    log_dir = "logs/finance/{}-{}".format(
        '-'.join(model_metadata),
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    model_hash = hashing.hash(model_metadata)
    model_path = f'./models/{model_hash}.hdf5'
    model.fit(
        data['x_train'], data['y_train'],
        shuffle=True,
        validation_data=(
            data['x_val'],
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
            keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1
            )
        ]
    )
    keras.backend.clear_session()
    return model_path


def mlp_arch(n_features, lookback, n_labels=3, hidden_size=64, depth=1, dropout_p=0.5):
    input_layer = keras.layers.Input(shape=(lookback, n_features,))
    x = keras.layers.Flatten()(input_layer)
    for i in range(depth):
        if i == depth-1:
            x = keras.layers.Dense(
                hidden_size,
                activation='relu',
                name='last_hidden',
                kernel_regularizer=keras.regularizers.l2(0.001)
            )(x)
        else:
            x = keras.layers.Dense(
                hidden_size,
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(0.001)
            )(x)
        x = keras.layers.Dropout(dropout_p)(x)
    output_layer = keras.layers.Dense(n_labels, activation='softmax')(x)
    model = keras.models.Model(input_layer, output_layer)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=[
            keras.metrics.AUC(),
            keras.metrics.Precision(),
            keras.metrics.Recall(),
            keras.metrics.CategoricalAccuracy()
        ]
    )
    return model


def get_model_by_name(name):
    models = {
        'mlp': mlp_arch,
        'lstm': lstm_arch,
    }
    return models[name]


def lstm_arch(n_features, lookback, n_labels=3, hidden_size=64, depth=0):
    input_layer = keras.layers.Input(shape=(lookback, n_features))
    x = input_layer
    for _ in range(depth):
        x = keras.layers.LSTM(hidden_size, return_sequences=True)(x)
    x = keras.layers.LSTM(hidden_size, name='last_hidden')(x)
    output_layer = keras.layers.Dense(n_labels, activation='softmax')(x)
    model = keras.models.Model(input_layer, output_layer)
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

def get_hidden_output(model_path):
    model = keras.models.load_model(model_path)
    layer_name = 'last_hidden'
    intermediate_layer_model = keras.models.Model(
        inputs=model.input,
        outputs=model.get_layer(layer_name).output
    )
    return intermediate_layer_model
