import pandas as pd
import re
from utility import get_mode, log
import numpy as np
import gc
import CONSTANT
from concurrent.futures import ProcessPoolExecutor
from utility import timeit, Timer
from CONSTANT import NUMERICAL_PREFIX, MULTI_CAT_PREFIX, LMU_SUFFIX, MULTI_TO_MULTI_FIRST_SUFFIX, \
    FROM_TIME, CATEGORY_PREFIX, C2M, M2M, LEN_SUFFIX, MULTI_TO_MULTI_MODE_SUFFIX

def find_n_sub_str(src, sub, ith):

    index = src.find(sub, 0)
    find_ith = 1
    while ((index != -1) & (find_ith != ith)):
        index = src.find(sub, index + 1)
        find_ith += 1

    if index == -1:
        return src
    return src[:index]


def multicat_sample(df,add_seri_list):
    df_multi_cat_cols = [c for c in list(df.columns) if (c.startswith(CONSTANT.MULTI_CAT_PREFIX))]
    add_df_columns = [c.name for c in add_seri_list if (c.name.startswith(CONSTANT.MULTI_CAT_PREFIX))]
    add_df_all_columns = [c.name for c in add_seri_list]

    mulcat_feature_list = df_multi_cat_cols + add_df_columns
    for c in mulcat_feature_list:
        if c in df_multi_cat_cols:
            df[c] = df[c].apply(lambda x: find_n_sub_str(x, ',', 10))
        else:
            col_index = add_df_all_columns.index(c)
            add_seri_list[col_index] = add_seri_list[col_index].apply(lambda x: find_n_sub_str(x, ',', 10))


@timeit
def MV_Combine_Divide(df):
    timer = Timer()
    mulcat_feature_list = [c for c in list(df.columns) if (c.startswith(CONSTANT.MULTI_CAT_PREFIX)) & ((C2M in c) | (M2M in c))]
    if len(mulcat_feature_list) == 0:
        print('NO columns in cat/multicat 2 multicat')
        return []

    print(f'{len(mulcat_feature_list)} features to process')
    row_splits = int(np.ceil(len(df)/1000000))
    column_splits = int(np.ceil(len(mulcat_feature_list) / 20))
    splits = row_splits * column_splits
    print(f' **** We should split it as {splits}, {column_splits}-{row_splits} splits to process! ****')
    cols_split = np.array_split(mulcat_feature_list, splits)
    data_list = []
    for i, cols in enumerate(cols_split):
        if len(cols) >= 1:
            pool = ProcessPoolExecutor(4)
            result_list = pool.map(MV_Combine_Divide_sub, [[df[col], col] for col in cols])

            pool.shutdown(wait=True)

            for data, _ in result_list:
                if data is not None:
                    data_list += data

            print(f'{i} split successful')

    timer.check("map done")

    df.drop(mulcat_feature_list, axis = 1, inplace = True)
    timer.check("drop")

    return data_list


def MV_Combine_Divide_sub(params):
    df, c = params

    df_target_list = []
    df_name_list = []

    def get_first(x, seq):
        pos = x.find(seq)
        if pos == -1:
            return x
        else:
            return x[:pos]

    # length - unique
    if M2M in c:
        col_name = NUMERICAL_PREFIX + c + LMU_SUFFIX
        def length_M_unique(x):
            d = re.split('[;,]', x)
            return int(len(d) - len(set(d)))
        df_target_list.append(df.apply(length_M_unique).astype('uint32').rename(col_name))
        df_name_list.append(col_name)
    else:
        col_name = NUMERICAL_PREFIX + c + LMU_SUFFIX
        def length_M_unique(x):
            d = x.split(',')
            return int(len(d) - len(set(d)))
        df_target_list.append(df.apply(length_M_unique).astype('uint32').rename(col_name))
        df_name_list.append(col_name)


    if np.percentile(df_target_list[-1], 50) > 0:
        col_name = CATEGORY_PREFIX + c + MULTI_TO_MULTI_MODE_SUFFIX
        if M2M in c:
            df_target_list.append(df.apply(lambda x: get_mode(re.split('[;,]', x))).rename(col_name))
        else:
            df_target_list.append(df.apply(lambda x: get_mode(x.split(','))).rename(col_name))
        df_name_list.append(col_name)
    elif FROM_TIME not in c:
        if (C2M in c):
            if  np.percentile(df.apply(lambda x: x.count(',')), 50) == 0:
                col_name = CATEGORY_PREFIX + c + MULTI_TO_MULTI_FIRST_SUFFIX
                df_target_list.append(df.apply(lambda x: get_first(x, ',')).rename(col_name))
                df_name_list.append(col_name)

        elif (M2M in c):
            if np.percentile(df.apply(lambda x: x.count(';')), 50) == 0:
                col_name = MULTI_CAT_PREFIX + c + MULTI_TO_MULTI_FIRST_SUFFIX
                df_target_list.append(df.apply(lambda x: get_first(x, ';')).rename(col_name))
                df_name_list.append(col_name)
    # first

    if FROM_TIME in c:
        if (C2M in c):
            col_name = CATEGORY_PREFIX + c + MULTI_TO_MULTI_FIRST_SUFFIX
            df_target_list.append(df.apply(lambda x: get_first(x, ',')).rename(col_name))
            df_name_list.append(col_name)
        elif (M2M in c):
            col_name = MULTI_CAT_PREFIX + c + MULTI_TO_MULTI_FIRST_SUFFIX
            df_target_list.append(df.apply(lambda x: get_first(x, ';')).rename(col_name))
            df_name_list.append(col_name)

    if len(df_target_list) == 0:
        return None, None
    return df_target_list, df_name_list


@timeit
def MV_Feature_Extract(df, add_seri_list):
    timer = Timer()
    df_columns = [c for c in list(df.columns) if (c.startswith(CONSTANT.MULTI_CAT_PREFIX))]
    add_df_all_columns = [c.name for c in add_seri_list]
    add_df_columns = [c.name for c in add_seri_list if (c.name.startswith(CONSTANT.MULTI_CAT_PREFIX))]

    mulcat_feature_list = df_columns + add_df_columns
    if len(mulcat_feature_list) == 0:
        print('NO columns in cat/multicat 2 multicat')
        return add_seri_list

    print(f'{len(mulcat_feature_list)} features to process')
    row_splits = int(np.ceil(len(df) / 1000000))
    column_splits = int(np.ceil(len(mulcat_feature_list) / 20))
    splits = row_splits * column_splits
    print(f' **** We should split it as {splits}, {column_splits}-{row_splits} splits to process! ****')
    cols_split = np.array_split(mulcat_feature_list, splits)
    data_list = []
    name_list = []
    for i, cols in enumerate(cols_split):
        log(cols_split)
        if len(cols) >= 1:
            pool = ProcessPoolExecutor(4)
            process_data_list = []
            for col in cols:
                if col in df_columns:
                    process_data_list.append([df[col], col])
                else:
                    process_data_list.append([add_seri_list[add_df_all_columns.index(col)], col])

            result_list = pool.map(MV_Feature_Extract_sub, process_data_list)
            del process_data_list
            pool.shutdown(wait=True)

            for i_data, i_name in result_list:
                if i_data is not None:
                    data_list += i_data
                    name_list += i_name

            print(f'{i} split successful')

    add_seri_list += data_list
    timer.check("map done")
    return add_seri_list


def MV_Feature_Extract_sub(params):
    df, c = params
    df_target_list = []
    df_name_list = []

    col_name = NUMERICAL_PREFIX + c + LEN_SUFFIX
    def count_num(x):
        if x == '0':
            return 0
        else:
            return x.count(',') + 1
    df_target_list.append(df.apply(count_num).astype('uint32').rename(col_name))

    col_name = NUMERICAL_PREFIX + c + LMU_SUFFIX
    def length_M_unique(x):
        d = x.split(',')[:100]
        return int(len(d) - len(set(d)))
    df_target_list.append(df.apply(length_M_unique).astype('uint32').rename(col_name))
    df_name_list.append(col_name)

    # mode
    if np.percentile(df_target_list[-1], 50) > 0:
        col_name = CATEGORY_PREFIX + c + MULTI_TO_MULTI_MODE_SUFFIX
        df_target_list.append(df.apply(lambda x: get_mode(x.split(','))).rename(col_name))
        df_name_list.append(col_name)

    if len(df_target_list) == 0:
        return None, None
    return df_target_list, df_name_list