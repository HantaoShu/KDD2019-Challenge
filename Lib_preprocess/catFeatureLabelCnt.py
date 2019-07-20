from utility import log, timeit, Timer
import CONSTANT
from concurrent.futures import ProcessPoolExecutor
import datetime
import numpy as np
import os
import pandas as pd
import CONSTANT
import gc
from CONSTANT import LABEL,NUMERICAL_PREFIX,LABEL_CNT_SUFFIX




@timeit
def cat_Lable_Cnt_Fun(train_data, y, test_data, config):
    timer = Timer()
    cat_feature_list = [c for c in train_data if c.startswith(CONSTANT.CATEGORY_PREFIX)]
    if len(cat_feature_list) == 0: return None

    # train_data_length = len(train_data)
    train_data[LABEL] = y

    row_sp = int(np.ceil((len(train_data) + len(test_data)) / 1000000))
    col_sp = int(np.ceil(len(cat_feature_list) / 20))
    sp= row_sp * col_sp
    print(f' **** We should split it as {sp}, {col_sp}-{row_sp} sp to process! ****')
    cols_split = np.array_split(cat_feature_list, sp)
    data_list = []
    for i, cols in enumerate(cols_split):
        if len(cols) >= 1:
            pool = ProcessPoolExecutor(4)
            result_list = pool.map(cat_Lable_Cnt_Fun_sub,
                                   [[train_data[[col, LABEL]], test_data[[col]], col, config['pos_rate'], config[CONSTANT.TRAIN_LEN_OF_TRAIN_VAL]] for col in
                                    cols])

            pool.shutdown(wait=True)

            for i_data in result_list:
                if i_data is not None:
                    data_list += i_data

            print(f'{i} split successful')

    # feature_data = pd.concat(data_list, axis=1, copy=False)
    # feature_data.columns = name_list
    # timer.check("label count map done")
    # del data_list
    # gc.collect()

    test_data.drop(cat_feature_list, axis=1, inplace=True)
    cat_feature_list+=[LABEL]
    train_data.drop(cat_feature_list, axis=1, inplace=True)
    timer.check("drop")

    return data_list

def cat_Lable_Cnt_Fun_sub(params):
    train_df, test_df, c, prior, train_len_of_trainVal = params
    df_target_list = []

    col_name = NUMERICAL_PREFIX + c + LABEL_CNT_SUFFIX

    df_target_list.append(CatLabelCntClass(prior, train_len_of_trainVal).fit_transform(train_df, test_df).rename(col_name))

    if len(df_target_list) == 0:
        return None
    return df_target_list

class CatLabelCntClass:

    def __init__(self, prior, train_len_of_trainVal):
        self.prior = prior
        self.train_len_of_trainVal = train_len_of_trainVal

    def fit_transform(self, X, test_X):
        col = X.columns[0]
        label = X.columns[1]
        # CONSTANT.ROLLING_FOLD_WINDOW

        X_label_cnt_list = []

        num_per_fold = int(np.ceil(self.train_len_of_trainVal / CONSTANT.FOLD_NUM))
        index_range = np.arange(self.train_len_of_trainVal)
        for i in range(CONSTANT.FOLD_NUM - 1):
            large_split = list(index_range[ : (i+1)*num_per_fold])
            small_split = list(index_range[(i+1)*num_per_fold : (i+2)*num_per_fold])
            if len(small_split)==0: break

            label_cnt = X.iloc[large_split].groupby(col).agg({label: ['count','sum']})
            label_cnt.columns = label_cnt.columns.get_level_values(1)
            label_cnt['label_cnt'] = (label_cnt['sum'] + self.prior)/ (label_cnt['count'] + 1)
            label_cnt_result = X.iloc[small_split].reset_index().merge(label_cnt, how='left', on=col).fillna(value={'label_cnt':self.prior}).set_index('index')
            X_label_cnt_list.append(label_cnt_result['label_cnt'])
            del label_cnt, label_cnt_result

            if i == 0:
                # 处理最开始的数据
                first_roll_data = X.iloc[large_split]
                first_roll_data['label_cnt'] = self.prior
                X_label_cnt_list.append(first_roll_data['label_cnt'])
                del first_roll_data

        # 处理验证集数据
        index_range = np.arange(len(X))
        large_split = list(index_range[: self.train_len_of_trainVal])
        small_split = list(index_range[ self.train_len_of_trainVal:])
        label_cnt = X.iloc[large_split].groupby(col).agg({label: ['count', 'sum']})
        label_cnt.columns = label_cnt.columns.get_level_values(1)
        label_cnt['label_cnt'] = (label_cnt['sum'] + self.prior) / (label_cnt['count'] + 1)
        label_cnt_result = X.iloc[small_split].reset_index().merge(label_cnt, how='left', on=col).fillna(
            value={'label_cnt': self.prior}).set_index('index')
        X_label_cnt_list.append(label_cnt_result['label_cnt'])
        del label_cnt, label_cnt_result


        # 处理测试集数据
        label_cnt = X.groupby(col).agg({label: ['count', 'sum']})
        label_cnt.columns = label_cnt.columns.get_level_values(1)
        label_cnt['label_cnt'] = (label_cnt['sum'] + self.prior) / (label_cnt['count'] + 1)
        label_cnt_result = test_X.reset_index().merge(label_cnt, how='left', on=col).fillna(value={'label_cnt':self.prior}).set_index('index')
        X_label_cnt_list.append(label_cnt_result['label_cnt'])


        result = pd.concat(X_label_cnt_list, axis = 0).astype('float32')
        # result.sort_index(inplace = True)
        del label_cnt, label_cnt_result

        return result