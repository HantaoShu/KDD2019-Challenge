from typing import Dict, List
from utility import Config, log, timeit, Timer
from lib import hyperopt
import lightgbm as lgb
import numpy as np
import pandas as pd
from lib.hyperopt import STATUS_OK, Trials, hp, space_eval, tpe
from sklearn.metrics import roc_auc_score
import CONSTANT
import time
import gc

MODEL = "model"


@timeit
def train(X_train, X_val, y_train, y_val, config, t_remain):
    log(f'remaining time {t_remain}')
    timer = Timer()
    start_time = time.time()
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "num_threads": 4
    }

    log('---------- Training Params ------------')
    log(params)

    rolling_first_length = int(np.ceil(len(X_train) / CONSTANT.FOLD_NUM))
    time_delta_rolling = 5
    # weight = [0] * time_delta_rolling + (rolling_first_length - time_delta_rolling) * [0.5] + (
    #             len(X_train) - rolling_first_length) * [1]
    weight = (rolling_first_length ) * [0.5] + (len(X_train) - rolling_first_length) * [1]
    X_train[CONSTANT.WEIGHT_NAME] = weight

    train_data = lgb.Dataset(np.array(X_train.drop(CONSTANT.WEIGHT_NAME, axis=1).values, dtype=np.float32), label=y_train,
                             weight=X_train[CONSTANT.WEIGHT_NAME])
    valid_data = lgb.Dataset(np.array(X_val.values, dtype=np.float32), label=y_val, reference=train_data)

    train_columns = list(X_train.columns)
    train_columns.remove(CONSTANT.WEIGHT_NAME)

    exec_time = time.time() - start_time
    t_remain -= exec_time

    hyperparams, all_trees, hyper_len = hyperopt_lgb(X_train, X_val, y_train, y_val, params, t_remain * 0.4)  # Modi
    hyperpara_tunning_exec_time = time.time() - start_time

    t_remain -= (hyperpara_tunning_exec_time)
    t_remain -= 30


    tree_per_time = hyperpara_tunning_exec_time / all_trees

    print(f'Important, only {t_remain} for training')

    all_data_len = len(X_train) + len(X_val)
    print(f'{all_data_len}, {hyper_len},{tree_per_time}')
    evaluate_time_per_tree = all_data_len / hyper_len * tree_per_time + 0.2
    max_iterations = int(t_remain / evaluate_time_per_tree)
    print(f'Important, only {max_iterations} for training')

    #del X_train, X_val, y_train, y_val
    gc.collect()


    train_file_name = 'train' + str(time.time()) + '.bin'
    valid_file_name = 'valid' + str(time.time()) + '.bin'

    train_data.save_binary(train_file_name)
    valid_data.save_binary(valid_file_name)
    train_data = lgb.Dataset(train_file_name)
    valid_data = lgb.Dataset(valid_file_name)

    config[MODEL] = lgb.train({**params, **hyperparams},
                              train_data,
                              max_iterations,
                              valid_data,
                              early_stopping_rounds=30,
                              verbose_eval=100)

    timer.check("fitting")

    # try:
    #     dataset_name = config.data.get('dataset_name', None)
    #     if dataset_name != None:
    #         train_score = roc_auc_score(y_train, config[MODEL].predict(X_train))
    #         validation_score = roc_auc_score(y_val, config[MODEL].predict(X_val))
    #
    #         print(validation_score)
    #         log_path = config.data.get('log_path', 'x.log')
    #         with open(log_path, 'a') as f:
    #             print('-' * 10, file=f)
    #             print(f'train_score " {train_score}', file=f)
    #             print(f'validation_score : {validation_score}', file=f)
    #     print('here')
    #     importance = config[MODEL].feature_importance(importance_type='split')
    #     feature_name = config[MODEL].feature_name()
    #     feature_importance = pd.DataFrame({'feature_name': feature_name, 'importance': importance})
    #     feature_importance.to_csv(f'{dataset_name}_feature_importance.csv', index=False)
    # except:
    #     # Nothing
    #     pass

    del train_data, valid_data
    return train_columns


@timeit
def predict(X, config):
    return config[MODEL].predict(np.array(X.values, dtype=np.float32))


@timeit
def hyperopt_lgb(X_train, X_val, y_train, y_val, params, time_left):
    start_time = time.time()
    num_sub_sample = 150000

    X_train, y_train = data_sample(X_train, y_train, num_sub_sample, CONSTANT.SEED)
    X_val, y_val = data_sample(X_val, y_val, num_sub_sample, CONSTANT.SEED)

    train_len = len(X_train) + len(X_val)

    exec_time = time.time() - start_time
    time_left = time_left - exec_time - 30

    space = {
        "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.05)),
        "num_leaves": hp.choice("num_leaves", np.linspace(16, 256, 24, dtype=int)), # 256
        "feature_fraction": hp.quniform("feature_fraction", 0.6, 0.9, 0.1),
        "bagging_fraction": hp.quniform("bagging_fraction", 0.6, 0.9, 0.1),
        "bagging_freq": hp.choice("bagging_freq", np.linspace(1, 30, 10, dtype=int)),
        #"min_data_in_leaf": hp.choice('min_data_in_leaf', []),
        "reg_alpha": hp.uniform("reg_alpha", 0, 2),
        "reg_lambda": hp.uniform("reg_lambda", 0, 10),
        "min_child_weight": hp.uniform('min_child_weight', 3, 20),
        "max_bin": hp.choice("max_bin", [255, 511]),
    }


    # train_data = lgb.Dataset(X_train.drop(CONSTANT.WEIGHT_NAME, axis=1), label=y_train, weight=X_train[CONSTANT.WEIGHT_NAME], free_raw_data=False)
    # valid_data = lgb.Dataset(X_val, label=y_val, free_raw_data=False, reference=train_data)

    train_data = lgb.Dataset(np.array(X_train.drop(CONSTANT.WEIGHT_NAME, axis=1).values, dtype=np.float32), label=y_train,
                             weight=X_train[CONSTANT.WEIGHT_NAME], free_raw_data=False)
    valid_data = lgb.Dataset(np.array(X_val.values, dtype=np.float32), label=y_val, reference=train_data, free_raw_data=False)


    del X_train, X_val, y_val, y_train
    gc.collect()


    def objective(hyperparams):


        model = lgb.train({**params, **hyperparams}, train_data, 400,
                          valid_data, early_stopping_rounds = 5, verbose_eval=0) # Modift row 30
        trees = model.best_iteration + 5

        score = model.best_score["valid_0"][params["metric"]]

        # in classification, less is better
        return {'loss': -score, 'status': STATUS_OK, 'trees': trees}

    trials = Trials()
    best = hyperopt.fmin(fn=objective, space=space, trials=trials,
                         algo=tpe.suggest, max_evals=60, verbose=1, time_left=time_left,  # modift row 10
                         rstate=np.random.RandomState(CONSTANT.SEED))

    trees_list = []
    for b in trials.results:
        trees_list.append(b['trees'])

    all_trees = np.sum(trees_list)


    hyperparams = space_eval(space, best)

    log(f"auc = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")

    print(hyperparams)
    del train_data, valid_data
    gc.collect()
    return hyperparams, all_trees, train_len


def time_split(X, y, test_size):
    sp = int(np.ceil(len(X) * (1 - test_size)))
    Xtrain, Xtest = X[:sp], X[sp:]
    ytrain, ytest = y[:sp], y[sp:]

    return Xtrain, Xtest, ytrain, ytest


def data_sample(X, y, nrows, rn):
    if len(X) > nrows:
        X_sample = X.sample(nrows, random_state=rn)
        y_sample = y[X_sample.index]
    else:
        X_sample = X
        y_sample = y

    return X_sample, y_sample
