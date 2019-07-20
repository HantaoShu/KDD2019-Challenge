import os

os.system("pip3 install lightgbm")
os.system("pip3 install joblib")
os.system('pip3 install pandas==0.24.2')
os.system("pip3 install future")
os.system("pip3 install networkx")
os.system("pip3 install tqdm")
os.system("pip3 install holidays")

import numpy as np
import pandas as pd
from automl import predict, train
from CONSTANT import MAIN_TABLE_NAME, SEED, NUMERICAL_PREFIX, NULL_COUNT_PREFIX, TIME_PREFIX, TRAIN_DATA_LENGTH, \
    TEST_DATA_LENGTH, STAGE, TRAIN_LEN_OF_TRAIN_VAL, VAL_LEN_OF_TRAIN_VAL, TIME_COL
from merge import merge_table
from contextlib import contextmanager
from preprocess import clean_df, clean_tables, feature_engineer, \
    pre_process, pre_feature_extract, pre_tables_memory_cut
from utility import Config, timeit,  table_memory_cut
from Lib_preprocess.mvLableCnt import Mv_Label_Cnt_Func
from Lib_preprocess.catFeatureLabelCnt import cat_Lable_Cnt_Fun
from automl import data_sample
import signal
import time
import math
import gc
import warnings
import random as rn

warnings.filterwarnings('ignore')

os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
rn.seed(SEED)


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
        try:
            yield
        finally:
            exec_time = time.time() - start_time
            signal.alarm(0)
            self.exec += exec_time
            self.duration += exec_time
            remain_time = math.ceil(self.total - self.exec)
            self.remain = remain_time


class Model:
    def __init__(self, info):
        self.info = info
        self.config = Config(info)
        print(f'Info : {info}')
        self.map_config = {}
        self.tables = None
        try:
            self.evaluated_data_size = self._evaluate_data_size()
        except:
            self.evaluated_data_size = 1000000

    def _fectch_time_range(self, df):
        res = {}
        for c in [c for c in df.keys() if c.startswith(TIME_PREFIX)]:
            if df[c].nunique() == 0:
                df.drop(c, axis=1, inplace =True)
            else:
                time_range = (df[c].max() - df[c].min()).total_seconds()

                hours = math.floor(time_range / 3600)
                days = math.floor(hours / 24)
                months = math.floor(days / 30)
                print('-' * 10)
                print(f'Training time range {months} months, {days} days, {hours} hours for category {c}')
                res[c] = {'hour': hours, 'day': days, 'month': months}
        return res

    def _fetch_class_rate(self, y):
        pos_rate = len(y[y == 1]) / len(y)
        print(f"The class distribution is {pos_rate}")
        self.config['pos_rate'] = pos_rate

    def _fecth_table_cnt(self):
        table_cnt = {}
        for table in self.info['tables']:
            tmp_cnt = {}
            num = 0
            cat = 0
            multicat = 0
            time = 0
            for c in self.info['tables'][table].keys():
                if c.startswith('n_'):
                    num += 1
                if c.startswith('c_'):
                    cat += 1
                if c.startswith('m_'):
                    multicat += 1
                if c.startswith('t_'):
                    time += 1

            tmp_cnt['num'] = num
            tmp_cnt['cat'] = cat
            tmp_cnt['multi-cat'] = multicat
            tmp_cnt['time'] = time
            table_cnt[table] = tmp_cnt
        return table_cnt

    def get_multi_factor(self, dty, attach_t_name, join_type):
        if attach_t_name is None or join_type == 'one':
            if dty == 'num':
                factor = 1
            elif dty == 'multi-cat':
                factor = 8
            elif dty == 'time':
                factor = 9
            else:
                factor = 2
        else:
            if dty == 'num':
                factor = 3
            elif dty == 'multi-cat':
                factor = 17
            elif dty == 'time':
                factor = 9
            else:
                factor = 8
        if attach_t_name != 'main' and attach_t_name != None:
            factor *= 2

        return factor

    def _evaluate_data_size(self):
        table_rels = {}
        for rel in self.info['relations']:
            print(rel)
            type_ = rel['type'].split('_')[-1]
            table_name = rel['table_B']
            table_rels[table_name] = [rel['table_A'], type_]
        table_rels['main'] = [None, 'one']

        table_cnt = self._fecth_table_cnt()
        print(table_cnt)

        all_cnt = 0
        for name in table_cnt:
            cnt = table_cnt[name]
            table_num_count = 0
            for dty, num in cnt.items():
                factor = self.get_multi_factor(dty, *table_rels[name])
                table_num_count += factor * num
            all_cnt += table_num_count
            print(name, table_num_count)

        print(f'evaluate columns : {all_cnt}  {471 * 1110000 / all_cnt}')
        return int(471 * 1110000 / all_cnt)

    def null_count_sum(self, df, config):

        null_count_cols = [c for c in df if c.startswith(NULL_COUNT_PREFIX)]
        if len(null_count_cols) > 0:
            df[NUMERICAL_PREFIX + 'null_count'] = df[null_count_cols].sum(axis=1) + df.isnull().sum(axis=1)
            df.drop(null_count_cols, axis=1, inplace=True)

    def _sort_by_time(self, df, y, time_col):
        df['#NO_USE_label'] = y
        df.sort_values(by=time_col, inplace=True)
        df.reset_index(drop=True, inplace=True)
        y = df['#NO_USE_label']
        df.drop('#NO_USE_label', axis=1, inplace=True)

        return df, y

    @timeit
    def fit(self, Xs, y, time_ramain):
        # fetch information of dara
        self.config[STAGE] = 'train'
        print(self._fectch_time_range(Xs[MAIN_TABLE_NAME]))
        self._fetch_class_rate(y)

        # sample data
        main_table = Xs[MAIN_TABLE_NAME]
        print('-' * 10)
        print(f'Note sample data {len(main_table)} / {self.evaluated_data_size}')
        print('-'*10)
        main_table, y = data_sample(main_table, y, self.evaluated_data_size, SEED)
        print('sampled data size', main_table.shape)

        # sort data by time
        Xs[MAIN_TABLE_NAME], y = self._sort_by_time(main_table, y, self.config[TIME_COL])

        self.tables = Xs
        self.train_label = y
        self.config[TRAIN_DATA_LENGTH] = len(Xs[MAIN_TABLE_NAME])

    @timeit
    def predict(self, X_test, time_remain):
        timer = Timer()
        timer.set(time_remain)
        with timer.time_limit('ProProcess'):
            # fetch information of test dataset
            self.config[TEST_DATA_LENGTH] = len(X_test)
            self.config['test_time'] = self._fectch_time_range(X_test)
            self.config[STAGE] = 'test'

            Xs = self.tables
            main_table = pd.concat([Xs[MAIN_TABLE_NAME], X_test], axis=0, copy=False)
            main_table.reset_index(drop=True, inplace=True)

            del Xs[MAIN_TABLE_NAME]
            Xs[MAIN_TABLE_NAME] = main_table

            pre_process(Xs, self.config)
            clean_tables(Xs)
            pre_feature_extract(Xs)
            pre_tables_memory_cut(Xs)

            X = merge_table(Xs, self.config)
            # clean datas
            del self.tables, Xs
            gc.collect()

            self.null_count_sum(X, self.config)
            clean_df(X, fill_time=True)
            # compress data for memory problem
            X = table_memory_cut(X)

            # feature engineering
            print('overall X size', X.shape)
            X, add_feature = feature_engineer(X, self.config)

            # 内存问题 11G
            X = table_memory_cut(X)
            add_feature = table_memory_cut(add_feature)
            X = pd.concat([X, add_feature], axis=1, copy=False)
            del add_feature
            print(X.shape)
            # re compress data

            # 测试集分割
            X_train_val, y_train_val = X.iloc[:self.config[TRAIN_DATA_LENGTH]], self.train_label
            X_test = X.iloc[self.config[TRAIN_DATA_LENGTH]:]

            train_len = int(self.config[TRAIN_DATA_LENGTH] * 0.8)
            valid_len = self.config[TRAIN_DATA_LENGTH] - train_len
            self.config[TRAIN_LEN_OF_TRAIN_VAL] = train_len
            self.config[VAL_LEN_OF_TRAIN_VAL] = valid_len
            del X
            gc.collect()

            # 特征处理
            all_label_count_feature_list = cat_Lable_Cnt_Fun(X_train_val, y_train_val, X_test, self.config)
            all_mutlicat_feature_data_list = Mv_Label_Cnt_Func(X_train_val, y_train_val, X_test, self.config)

            if (all_label_count_feature_list is None) & (all_mutlicat_feature_data_list is None):
                X_train, y_train = X_train_val.iloc[:train_len], self.train_label[:train_len]
                X_val, y_val = X_train_val.iloc[train_len:], self.train_label[train_len:]
            else:
                all_feature_list = []
                if all_label_count_feature_list is not None:
                    all_feature_list += all_label_count_feature_list
                if all_mutlicat_feature_data_list is not None:
                    all_feature_list += all_mutlicat_feature_data_list

                add_feature_data = pd.concat(all_feature_list, axis=1, copy=False)
                add_feature_data.sort_index(inplace=True)

                del all_label_count_feature_list, all_mutlicat_feature_data_list, all_feature_list
                gc.collect()

                X_train = pd.concat([X_train_val[:train_len], add_feature_data[:train_len]], axis=1, copy=False)
                X_val = pd.concat([X_train_val[train_len:self.config[TRAIN_DATA_LENGTH]], add_feature_data[train_len:self.config[TRAIN_DATA_LENGTH]]], axis=1, copy=False)
                y_train = self.train_label[:train_len]
                y_val = self.train_label[train_len:]

                X_test = pd.concat([X_test, add_feature_data[self.config[TRAIN_DATA_LENGTH]:]], axis=1, copy=False)

                del X_train_val, y_train_val, add_feature_data, self.train_label
                gc.collect()



        train_columns = train(X_train, X_val, y_train, y_val, self.config, timer.remain)
        del X_train, X_val, y_train, y_val
        gc.collect()

        result = predict(X_test[train_columns], self.config)

        return pd.Series(result)
