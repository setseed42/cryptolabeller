from hyperopt import STATUS_OK, hp, tpe, Trials, fmin
from hyperopt.pyll.stochastic import sample
import numpy as np
from model import train_model
from pprint import pprint
def perform_optimization(search_space):
    bayes_trials = Trials()
    MAX_EVALS = 100

    def objective(params):

        global ITERATION

        ITERATION += 1
        intable_params = [
            param for param in params
            if 'size' in param
            or 'depth' in param
            or param=='lookback'
        ]
        for param in intable_params:
            if param in params:
                params[param] = int(params[param])

        print(params)
        minloss = min(train_model(params)['val_loss'])
        return {
            'loss': minloss,
            'params': params,
            'iteration': ITERATION,
            'status': STATUS_OK
        }

    global ITERATION

    ITERATION = 0

    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=MAX_EVALS,
        trials=bayes_trials,
        rstate=np.random.RandomState(42)
    )
    return bayes_trials

def arch_search_space():
    return {
        'lstm_hidden_size': hp.quniform('lstm_hidden_size', 4, 6, 1),
        'lstm_last_hidden_size': hp.quniform('lstm_last_hidden_size', 4, 6, 1),
        'dense_hidden_size': hp.quniform('dense_hidden_size', 4, 6, 1),
        'lookback': hp.quniform('lookback', 0, 4, 1),
        'lstm_dropout_p': hp.uniform('lstm_dropout_p', 0.0, 0.5),
        'lstm_last_dropout_p': hp.uniform('lstm_last_dropout_p', 0.0, 0.5),
        'dense_dropout_p': hp.uniform('dense_dropout_p', 0.0, 0.5),
        'lstm_reg_power': hp.quniform('lstm_reg_power', 2, 5, 1),
        'lstm_last_reg_power': hp.quniform('lstm_last_reg_power', 2, 5, 1),
        'output_reg_power': hp.quniform('output_reg_power', 2, 5, 1),
        'dense_reg_power': hp.quniform('dense_reg_power', 2, 5, 1),
        'dense_depth': hp.quniform('dense_depth', 0, 3, 1),
        'lstm_depth': hp.quniform('lstm_depth', 0, 3, 1),
        'embedding_size': hp.quniform('embedding_size', 2, 5, 1),
    }



def get_best_params(search_space):
    bayes_trials = perform_optimization(search_space)
    bayes_trials_results = sorted(
        bayes_trials.results, key=lambda x: x['loss'])
    return bayes_trials_results

if __name__ == "__main__":
    search_space = arch_search_space()
    get_best_params(search_space)
