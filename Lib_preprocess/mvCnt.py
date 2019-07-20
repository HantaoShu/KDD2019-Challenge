from CONSTANT import MULTI_IN_MULTI, NUMERICAL_PREFIX, TABLE
from utility import log, timeit, Timer
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import CONSTANT
import gc


@timeit
def multi_feature_cnt(df, add_seri_list):
    df_columns = [c for c in list(df.columns) if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]
    add_df_columns = [c.name for c in add_seri_list if (c.name.startswith(CONSTANT.MULTI_CAT_PREFIX))]
    add_df_all_columns = [c.name for c in add_seri_list]

    mul_cat_feature_list = df_columns + add_df_columns
    if len(mul_cat_feature_list) == 0: return add_seri_list


    handle_multicat_list = []
    for c in mul_cat_feature_list:
        if (TABLE not in c) | ((TABLE in c) & (MULTI_IN_MULTI in c)):
            handle_multicat_list.append(c)

    if len(handle_multicat_list) == 0:
        # df.drop(df_columns, axis=1, inplace=True)
        #
        # for c in add_df_columns[::-1]:
        #     index = add_df_all_columns.index(c)
        #     print(add_df_all_columns[index])
        #     del add_seri_list[index]
        return add_seri_list
    print(f'{len(handle_multicat_list)} features to process')
    print(handle_multicat_list)
    row_splits = int(np.ceil(len(df) / 500000))
    column_splits = int(np.ceil(len(handle_multicat_list) / 10))
    splits = row_splits * column_splits
    print(f' **** We should split it as {splits}, {column_splits}-{row_splits} splits to process! ****')
    cols_split = np.array_split(handle_multicat_list, splits)
    data_list = []
    name_list = []
    for i, cols in enumerate(cols_split):
        if len(cols) >= 1:
            process_data_list = []
            pool = ProcessPoolExecutor(4)
            for col in cols:
                if col in df_columns:
                    process_data_list.append([df[[col]], col])
                else:
                    process_data_list.append([add_seri_list[add_df_all_columns.index(col)].to_frame(), col])
            result_list = pool.map(multi_feature_cnt_sub, process_data_list)
            pool.shutdown(wait=True)
            del process_data_list

            for i_data, i_name in result_list:
                if i_data is not None:
                    data_list += i_data
                    name_list += i_name

            print(f'{i} split successful')

    #删除multicat 特征
    # df.drop(df_columns, axis=1, inplace=True)
    #
    # for c in add_df_columns[::-1]:
    #     index = add_df_all_columns.index(c)
    #     print(add_df_all_columns[index])
    #     del add_seri_list[index]

    add_seri_list += data_list
    return add_seri_list

def multi_feature_cnt_sub(params):
    df, c = params
    df_target_list = []
    df_name_list = []

    # encode
    col_name = NUMERICAL_PREFIX + c + '_encoder'
    df_target_list.append(MVEncoder().fit_transform(df).astype('uint32').rename(col_name))
    df_name_list.append(col_name)

    if len(df_target_list) == 0:
        return None, None

    return df_target_list, df_name_list

class MVEncoder:
    @staticmethod
    def seperate(x):
        try:
            x = tuple(x.split(','))
        except AttributeError:
            x = ('-1',)
        return x

    def __init__(self, max_cat_num=1000):
        self.max_cat_num = max_cat_num

    def encode(self, cats):
        return max((self.mapping[c] for c in cats))

    def fit_transform(self, X):
        col = X.columns[0]
        tmp_X = X.groupby(col).size().reset_index().rename(columns={0: 'count'})

        X_multicat = tmp_X[col].astype(str).map(MVEncoder.seperate)

        cat_count = {}
        for cats, cnt in zip(X_multicat, tmp_X['count']):
            for c in cats:
                try:
                    cat_count[c] += cnt
                except KeyError:
                    cat_count[c] = cnt
        cat_list = np.array(list(cat_count.keys()))
        cat_num = np.array(list(cat_count.values()))
        idx = np.argsort(-cat_num)
        cat_list = cat_list[idx]

        self.mapping = {}
        for i, cat in enumerate(cat_list):
            self.mapping[cat] = min(i, self.max_cat_num)
        #self.mapping = cat_count
        del cat_count, cat_list, cat_num
        tmp_X['count'] = X_multicat.map(self.encode)
        return X.reset_index().merge(tmp_X, how='left', left_on=col, right_on=col).set_index('index')['count']