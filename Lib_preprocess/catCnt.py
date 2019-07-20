from CONSTANT import MULTI_IN_MULTI, TABLE, LABEL_COUNT, COUNT_FOR_MERGE
from utility import log, timeit, Timer
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import CONSTANT
import gc


@timeit
def cat_feature_cnt(df,add_seri_list):
    timer = Timer()
    df_columns = [c for c in list(df.columns) if (c.startswith(CONSTANT.CATEGORY_PREFIX))]
    add_df_all_columns = [c.name for c in add_seri_list]
    add_df_columns = [c.name for c in add_seri_list if (c.name.startswith(CONSTANT.CATEGORY_PREFIX))]
    target_list = df_columns + add_df_columns

    cat_feature_list = []
    for c in target_list:
        if (TABLE not in c) | ((TABLE in c) & (MULTI_IN_MULTI in c)) | (CONSTANT.TIME_SUFFIX in c)  :
            cat_feature_list.append(c)

    if len(cat_feature_list) == 0: return add_seri_list
    print(f'{len(cat_feature_list)} features to process')
    row_sp = int(np.ceil(len(df) / 1000000))
    col_sp = int(np.ceil(len(cat_feature_list) / 20))
    splits = row_sp * col_sp
    log(f' split it as {splits}, {col_sp}-{row_sp}')
    cols_split = np.array_split(cat_feature_list, splits)
    data = []
    for i, cols in enumerate(cols_split):
        if len(cols) >=1:
            process_data_list = []
            pool = ProcessPoolExecutor(4)
            for col in cols:
                if col in df_columns:
                    process_data_list.append([df[[col]], col])
                else:
                    process_data_list.append([add_seri_list[add_df_all_columns.index(col)].to_frame(), col])
            result_list = pool.map(cat_feature_cnt_sub, process_data_list)
            pool.shutdown(wait=True)
            del process_data_list
            for tmp_data, tmp_name in result_list:
                if tmp_data is not None:
                    data += tmp_data

            print(f'{i} split successful')

    timer.check("count map done")
    add_seri_list += data

    return add_seri_list


def cat_feature_cnt_sub(params):
    df, c = params
    df_target_list = []
    df_name_list = []
    col_name=CONSTANT.NUMERICAL_PREFIX + c + LABEL_COUNT
    df_target_list.append(CATEncoder().fit_transform(df).astype('uint32').rename(col_name))
    df_name_list.append(col_name)

    if len(df_target_list) == 0:
        return None, None
    return df_target_list, df_name_list

class CATEncoder:

    def __init__(self):
        pass

    def fit_transform(self, X):
        col = X.columns[0]
        count = X.groupby(col).size().reset_index().rename(columns={0: col + COUNT_FOR_MERGE})
        X = X.reset_index().merge(count, how='left', on=col).set_index('index')
        X_encode = X[col + COUNT_FOR_MERGE].astype('uint32')
        del X, count
        return X_encode