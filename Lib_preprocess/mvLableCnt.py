import pandas as pd
import CONSTANT
import numpy as np
import gc
from utility import timeit, Timer
from concurrent.futures import ProcessPoolExecutor
from functools import reduce
from CONSTANT import NUMERICAL_PREFIX, LABEL_CTR_MIN, CTR_MULTIPLY,LABEL


def seperate(x):
    try:
        x = tuple(x.split(','))
    except AttributeError:
        x = ('-1', )
    return x

@timeit
def Mv_Label_Cnt_Func(train_data, y, test_data, config):
    timer = Timer()
    cat_feature_list = [c for c in train_data if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]
    if len(cat_feature_list) == 0: return None
    train_data_length = len(train_data)

    train_data[LABEL] = y
    row_splits = int(np.ceil((len(train_data) + len(test_data)) / 500000))
    column_splits = int(np.ceil(len(cat_feature_list) / 10))
    splits = row_splits * column_splits
    print(f' **** We should split it as {splits}, {column_splits}-{row_splits} splits to process! ****')
    cols_split = np.array_split(cat_feature_list, splits)
    data_list = []
    for i, cols in enumerate(cols_split):
        if len(cols) >= 1:
            pool = ProcessPoolExecutor(4)
            result_list = pool.map(Mv_Label_Cnt_Func_sub,
                                   [[train_data[[col, LABEL]], test_data[col], col, config['pos_rate'], config['train_len_of_trainVal']] for col
                                    in
                                    cols])

            pool.shutdown(wait=True)

            for i_data in result_list:
                if i_data is not None:
                    data_list += i_data

            print(f'{i} split successful')

    # feature_data = pd.concat(data_list, axis=1, copy=False)
    # feature_data.columns = name_list
    # timer.check("label count map done")
    #
    # del data_list
    # gc.collect()

    test_data.drop(cat_feature_list, axis=1, inplace=True)
    cat_feature_list+=[LABEL]
    train_data.drop(cat_feature_list, axis=1, inplace=True)
    timer.check("drop")

    return data_list


def Mv_Label_Cnt_Func_sub(params):

    train_df, test_df, c, prior, train_len_of_trainVal = params
    df_target_list = []

    label_cnt_data = MvLabelCntClass(prior, train_len_of_trainVal).fit_transform(train_df, test_df)
    col_name = NUMERICAL_PREFIX + c + LABEL_CTR_MIN
    df_target_list.append(label_cnt_data.apply(lambda x:x[0]).astype('float32').rename(col_name))

    col_name = NUMERICAL_PREFIX + c + CTR_MULTIPLY
    df_target_list.append(label_cnt_data.apply(lambda x: x[1]).astype('float32').rename(col_name))

    del label_cnt_data

    if len(df_target_list) == 0:
        return None

    return df_target_list

class MvLabelCntClass():
    def __init__(self, prior, train_len_of_trainVal):
        self.prior = prior
        self.train_len_of_trainVal = train_len_of_trainVal

    def encode(self, cats):
        min_label_cnt = 100
        label_cnt_list = []
        for c in set(cats):
            try:
                current_label_cnt = self.mapping[c]
            except:
                if self.prior < 0.5:
                    current_label_cnt = 1 - self.prior
                else:
                    current_label_cnt = self.prior
            label_cnt_list.append(current_label_cnt)

            if current_label_cnt < min_label_cnt:
                min_label_cnt = current_label_cnt

        return (min_label_cnt, reduce(lambda x, y: x * y, label_cnt_list))


    def count_label(self, X, col):
        label_cat_count = {}
        idf_count = {}

        for cats, label_cnt, cnt in zip(X[col], X['sum'], X['count']):
            for c in set(cats):
                try:
                    idf_count[c] += cnt
                except KeyError:
                    idf_count[c] = cnt
                try:
                    label_cat_count[c] += label_cnt
                except KeyError:
                    label_cat_count[c] = label_cnt
            del cats

        label_cnt_rate = {}
        for c in idf_count.keys():
            try:
                label_cnt_rate_tmp = (label_cat_count[c] + self.prior) / (idf_count[c] + 1)
            except KeyError:
                label_cnt_rate_tmp = (self.prior) / (idf_count[c] + 1)
            if self.prior < 0.5:
                label_cnt_rate[c] = 1 - label_cnt_rate_tmp
            else:
                label_cnt_rate[c] = label_cnt_rate_tmp

        del label_cat_count, idf_count

        return label_cnt_rate

    def fit_transform(self, X, test_X):
        col, label = X.columns[0], X.columns[1]

        X[col] = X[col].astype(str).map(seperate)

        test_X = test_X.astype(str).map(seperate)

        # 处理训练集数据
        X_label_cnt_list = []
        num_per_fold = int(np.ceil(self.train_len_of_trainVal / CONSTANT.FOLD_NUM))
        index_range = np.arange(self.train_len_of_trainVal)
        for i in range(CONSTANT.FOLD_NUM - 1):
            large_split = list(index_range[: (i + 1) * num_per_fold])
            small_split = list(index_range[(i + 1) * num_per_fold:(i + 2) * num_per_fold])
            if len(small_split) == 0: break

            map_X = X.iloc[large_split].groupby(col).agg({label:['count', 'sum']})
            map_X.columns = map_X.columns.get_level_values(1)
            map_X.reset_index(inplace=True)
            self.mapping = self.count_label(map_X, col)
            X_label_cnt_list.append(X.iloc[small_split][col].map(self.encode))
            del map_X

            if i == 0:
                # 处理最开始的数据
                first_roll_data = X.iloc[large_split]
                if self.prior < 0.5:
                    first_roll_data['label_cnt'] = [(1 - self.prior, 1 - self.prior)] * len(first_roll_data)
                else:
                    first_roll_data['label_cnt'] = [(self.prior, self.prior)] * len(first_roll_data)

                X_label_cnt_list.append(first_roll_data['label_cnt'])
                del first_roll_data

        # 处理验证集数据
        index_range = np.arange(len(X))
        large_split = list(index_range[:self.train_len_of_trainVal])
        small_split = list(index_range[self.train_len_of_trainVal:])

        map_X = X.iloc[large_split].groupby(col).agg({label: ['count', 'sum']})
        map_X.columns = map_X.columns.get_level_values(1)
        map_X.reset_index(inplace=True)
        self.mapping = self.count_label(map_X, col)
        X_label_cnt_list.append(X.iloc[small_split][col].map(self.encode))
        del map_X

        # 处理测试集数据
        map_X = X.groupby(col).agg({label: ['count', 'sum']})
        map_X.columns = map_X.columns.get_level_values(1)
        map_X.reset_index(inplace=True)
        self.mapping = self.count_label(map_X, col)
        X_label_cnt_list.append(test_X.map(self.encode))

        result = pd.concat(X_label_cnt_list, axis=0)
        # result.sort_index(inplace=True)

        del self.mapping, X_label_cnt_list, map_X


        return result


