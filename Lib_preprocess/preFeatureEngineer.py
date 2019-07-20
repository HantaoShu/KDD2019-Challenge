import gc
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import CONSTANT
from Lib_preprocess.catCnt import cat_feature_cnt_sub
from Lib_preprocess.mvCnt import multi_feature_cnt_sub
from utility import timeit, Timer


@timeit
def before_merge_mv_cnt(df):
    timer = Timer()
    mul_cat_feature_list = [c for c in df if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]
    if len(mul_cat_feature_list) == 0: return df

    handle_multicat_list = []
    for c in mul_cat_feature_list:
        all_sum = df[c].apply(lambda x: x.count(',') + 1).sum()

        if all_sum < 5000000:
            handle_multicat_list.append(c)
        else:
            print(f'{c} has too many cats!')

    if len(mul_cat_feature_list) >= 1:
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
                pool = ProcessPoolExecutor(4)
                result_list = pool.map(multi_feature_cnt_sub, [[df[[col]], col] for col in cols])

                pool.shutdown(wait=True)

                for i_data, i_name in result_list:
                    if i_data is not None:
                        data_list += i_data
                        name_list += i_name

                print(f'{i} split successful')

        feature_data = pd.concat(data_list, axis=1, copy=False)
        feature_data.columns = name_list

        timer.check("map done")

        df = pd.concat([df, feature_data], axis=1, copy=False)
        timer.check("concat")

        del data_list
        del feature_data
        gc.collect()

    return df

@timeit
def before_cat_cnt(df):
    timer = Timer()
    cat_feature_list = [c for c in df if (c.startswith(CONSTANT.CATEGORY_PREFIX))]
    if len(cat_feature_list) == 0: return df
    print(f'{len(cat_feature_list)} features to process')
    row_splits = int(np.ceil(len(df) / 1000000))
    column_splits = int(np.ceil(len(cat_feature_list) / 20))
    splits = row_splits * column_splits
    print(f' **** We should split it as {splits}, {column_splits}-{row_splits} splits to process! ****')
    cols_split = np.array_split(cat_feature_list, splits)
    data_list = []
    name_list = []
    for i, cols in enumerate(cols_split):
        if len(cols) >= 1:
            pool = ProcessPoolExecutor(4)
            result_list = pool.map(cat_feature_cnt_sub, [[df[[col]], col] for col in cols])

            pool.shutdown(wait=True)

            for i_data, i_name in result_list:
                if i_data is not None:
                    data_list += i_data
                    name_list += i_name

            print(f'{i} split successful')

    feature_data = pd.concat(data_list, axis=1, copy=False)
    feature_data.columns = name_list

    timer.check("map done")

    df = pd.concat([df, feature_data], axis=1, copy=False)
    timer.check("concat")
    del data_list
    del feature_data
    gc.collect()
    return df
