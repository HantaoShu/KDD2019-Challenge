# pylint: disable=wrong-import-order, wrong-import-position, import-error
# pylint: disable=missing-docstring
import base64
from datetime import datetime
import os
from os.path import join
import sys
import argparse
from sklearn import metrics

#  os.system("pip3 install cryptography")
parser = argparse.ArgumentParser()
parser.add_argument('--m', type=str, default='offline')
parser.add_argument('--l', type=str, default='tmp.txt')
parser.add_argument('--code', type=str, default='mycode-5')
parser.add_argument('--sp', type=float, default=0.3, help="for split")
parser.add_argument('--d', type=str, default='data_public', help="dataset")
opt = parser.parse_args()

ROOT_DIR = os.getcwd()
# if opt.code != 'current':
# 	submission = join(ROOT_DIR, 'code')
# else:
submission = join(ROOT_DIR, opt.code)
print('-----'*10)
print(submission)


DIRS = {
    'input': join(ROOT_DIR, opt.d),
    'input_sub': join(ROOT_DIR, 'sample_data'),
    'output_show': join(ROOT_DIR, 'baselines_output'),
    'program': join(ROOT_DIR, 'ingestion_program'),
        'submission': submission,
    'ref': join(ROOT_DIR, 'sample_ref')

}

def mprint(msg):
    """info"""
    cur_time = datetime.now().strftime('%m-%d %H:%M:%S')
    print(f"INFO  [{cur_time}] {msg}")


sys.path.append(DIRS['submission'])

mprint("Import Model")
from model import Model
import json
import signal
import time
from contextlib import contextmanager
import numpy as np
import pandas as pd
import math

TYPE_MAP = {
    'time': str,
    'cat': str,
    'multi-cat': str,
    'num': np.float64
}


class TimeoutException(Exception):
    pass


class Timer:
    def __init__(self):
        self.duration = 0
        self.total = None
        self.remain = None
        self.exec = None

    def set(self, time_budget):
        self.total = time_budget
        self.remain = time_budget
        self.exec = 0

    @contextmanager
    def time_limit(self, pname):
        def signal_handler(signum, frame):
            raise TimeoutException("Timed out!")
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(self.remain)
        start_time = time.time()

        yield

        exec_time = time.time() - start_time
        signal.alarm(0)
        self.exec += exec_time
        self.duration += exec_time
        remain_time = math.ceil(self.total - self.exec)
        self.remain = remain_time

        mprint(f'{pname} success, time spent so far {self.exec} sec')


def read_train(datapath, info):
    train_data = {}
    for table_name, columns in info['tables'].items():
        mprint(f'Table name: {table_name}')

        table_dtype = {key: TYPE_MAP[val] for key, val in columns.items()}

        if table_name == 'main':
            table_path = join(datapath, 'train', 'main_train.data')
        else:
            table_path = join(datapath, 'train', f'{table_name}.data')

        date_list = [key for key, val in columns.items() if val == 'time']
        print(date_list)

        train_data[table_name] = pd.read_csv(
            table_path, sep='\t', dtype=table_dtype, parse_dates=date_list,
            date_parser=lambda millisecs: millisecs if np.isnan(
                float(millisecs)) else datetime.fromtimestamp(
                    float(millisecs)/1000))

    # get train label
    train_label = pd.read_csv(
        join(datapath, 'train', 'main_train.solution'))['label']
    return train_data, train_label


def read_info(datapath):
    mprint('Read info')
    with open(join(datapath, 'train', 'info.json'), 'r') as info_fp:
        info = json.load(info_fp)
    mprint(f'Time budget for this task is {info["time_budget"]} sec')
    return info


def read_test(datapath, info):
    # get test data
    main_columns = info['tables']['main']
    table_dtype = {key: TYPE_MAP[val] for key, val in main_columns.items()}

    table_path = join(datapath, 'test', 'main_test.data')

    date_list = [key for key, val in main_columns.items() if val == 'time']

    test_data = pd.read_csv(
        table_path, sep='\t', dtype=table_dtype, parse_dates=date_list,
        date_parser=lambda millisecs: millisecs if np.isnan(
            float(millisecs)) else datetime.fromtimestamp(
                float(millisecs) / 1000))
    return test_data


def write_predict(output_dir, dataname, prediction):
    os.makedirs(output_dir, exist_ok=True)
    prediction.rename('label', inplace=True)
    prediction.to_csv(
        join(output_dir, f'{dataname}.predict.{opt.m}'), index=False, header=True)

def get_auc(prediction, solution):
    # solution = pd.read_csv(
    # 	join(dirs['ref'], dataname, 'main_test.solution'))
    auc = metrics.roc_auc_score(solution, prediction)
    return auc

def data_split(X, y, test_size):
    split_bar = int(np.ceil(len(X) * (1 - test_size)))
    Xtrain, Xtest = X[:split_bar], X[split_bar:]
    ytrain, ytest = y[:split_bar], y[split_bar:]

    return Xtrain, Xtest, ytrain, ytest


def main():
    datanames = sorted(os.listdir(DIRS['input']))
    sub_datanames = sorted(os.listdir(DIRS['input_sub']))
    mprint(f'Datanames: {datanames}')
    timer = Timer()

    predictions = {}
    auc_dict = {}
    # datanames = ['A']
    for dataname in datanames:
    # if True:
        mprint(f'Read data: {dataname}')
        datapath = join(DIRS['input'], dataname)
        info = read_info(datapath)
        timer.set(info['time_budget'])
        train_data, train_label = read_train(datapath, info)
        print(train_data['main']['t_01'].min())
        if opt.m == 'offline':
            # DATA_RESORT
            main_table = train_data['main']
            main_table['#NO_USE_label'] = train_label
            main_table.sort_values(by=info['time_col'], inplace=True)
            main_table.reset_index(drop=True, inplace=True)
            train_label = main_table['#NO_USE_label']
            main_table.drop('#NO_USE_label', axis=1, inplace=True)

            # DATA_SPLIT
            sp = {'A':0.5,'B':0.33,'C':0.33,'D':0.5,'E':0.14}
            if dataname in ['A','B','C','D','E']:
                xtrain, xtest, ytrain, ytest = data_split(main_table, train_label, sp[dataname])
                xtrain['#NO_USE_label'] = ytrain
                xtrain = xtrain.sample(frac=1).reset_index(drop=True)
                train_label = xtrain['#NO_USE_label']
                xtrain.drop('#NO_USE_label', axis=1, inplace=True)
                train_data['main'] = xtrain
            else:
                xtrain, xtest, ytrain, ytest = data_split(main_table, train_label, opt.sp)
                train_data['main'] = xtrain
                train_label = ytrain

        mprint('Initalize model')
        with timer.time_limit('Initialization'):
            info['dataset_name'] = dataname
            cmodel = Model(info)


        mprint('Start fitting')
        with timer.time_limit('Fitting'):
            # cmodel.fit(train_data, train_label, 500)
            cmodel.fit(train_data, train_label, timer.remain)

        test_data = read_test(datapath, info)
        # user prediction
        mprint('Start prediction')
        with timer.time_limit('Prediction'):
            if opt.m == 'offline':
                predictions_result = cmodel.predict(xtest, timer.remain)
                predictions_result.rename('label', inplace=True)
                predictions[dataname] = predictions_result
                auc = get_auc(predictions[dataname], ytest)
                ytest.to_csv('Clabel.csv')
                with open(opt.l, 'a') as f:
                    print(f'{dataname} score : {auc}', file=f)
            else:
                predictions_result = cmodel.predict(test_data, timer.remain)
                predictions_result.rename('label', inplace=True)
                predictions[dataname] = predictions_result

        mprint(f'Done, exec_time={timer.exec}')

    # predict K, L
    # for dataname in sub_datanames:
    # 	mprint(f'Read data: {dataname}')
    # 	datapath = join(DIRS['input_sub'], dataname)
    # 	info = read_info(datapath)
    # 	timer.set(info['time_budget'])
    # 	train_data, train_label = read_train(datapath, info)
    #
    # 	mprint('Initalize model')
    # 	with timer.time_limit('Initialization'):
    # 		info['log_path'] = opt.l
    # 		info['dataset_name'] = dataname
    # 		cmodel = Model(info)
    #
    # 	mprint('Start fitting')
    # 	with timer.time_limit('Fitting'):
    # 		cmodel.fit(train_data, train_label, timer.remain)
    #
    # 	test_data = read_test(datapath, info)
    # 	# user prediction
    # 	mprint('Start prediction')
    # 	with timer.time_limit('Prediction'):
    # 		predictions_result = cmodel.predict(test_data, timer.remain)
    # 		predictions_result.rename('label', inplace=True)
    # 		predictions[dataname] = predictions_result
    #
    # 	solution = pd.read_csv(
    # 		join(DIRS['ref'], dataname, 'main_test.solution'))
    # 	auc = get_auc(predictions[dataname], solution)
    # 	with open(opt.l, 'a') as f:
    # 		print(f'{dataname} score : {auc}', file=f)
    #
    # 	mprint(f'Done, exec_time={timer.exec}')

    mprint(f'Write results')
    for dataname in datanames:
        write_predict(DIRS['output_show'], dataname, predictions[dataname])

    # mprint(f'Duration: {timer.duration}')
    # with open(join(DIRS['output'], 'duration.txt'), 'w') as out_f:
    # 	out_f.write(str(timer.duration))


if __name__ == '__main__':
    main()
