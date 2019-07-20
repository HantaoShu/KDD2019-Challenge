import pandas as pd
import CONSTANT
from utility import log, timeit, Timer, pre_cut_memory
import gc
from Lib_preprocess.preFeatureEngineer import before_merge_mv_cnt, before_cat_cnt
from Lib_preprocess.catCnt import cat_feature_cnt
from Lib_preprocess.mvCnt import multi_feature_cnt
from Lib_preprocess.mvProcess import MV_Combine_Divide, MV_Feature_Extract, multicat_sample
from Lib_preprocess.dtProcess import timeExtract, td_avg, t_delta_extract

NULL_COUNT = 'nc_null_count'


def NumberOfNull(df):
    df[NULL_COUNT] = df.isnull().sum(axis = 1)


@timeit
def pre_tables_memory_cut(tables):
    for name in tables:
        tables[name] = pre_cut_memory(tables[name])


@timeit
def pre_process(tables, config):
    for name in tables:
        NumberOfNull(tables[name])


@timeit
def pre_feature_extract(tables):
    for name in tables:
        if name != 'main':
            print(f'---------- {name} ------------')
            tables[name] = before_cat_mv_cnt(tables[name])

@timeit
def before_cat_mv_cnt(df):
    df = before_merge_mv_cnt(df)
    df = before_cat_cnt(df)
    return df


@timeit
def clean_tables(tables, fill_time = False):
    for name in tables:
        log(f"cleaning table {name}")
        clean_df(tables[name], fill_time)

@timeit
def clean_df(df, fill_time = False):
    fillna(df, fill_time)


@timeit
def fillna(df, fill_time=False):
    num_features = [c for c in df if c.startswith(CONSTANT.NUMERICAL_PREFIX)]
    cat_features = [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]
    multicat_features = [c for c in df if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]
    time_features = [c for c in df if c.startswith(CONSTANT.TIME_PREFIX)]

    drop_cols = []
    for c in num_features:
        if df[c].nunique() > 1:
            # 有>1个值
            mean = df[c].mean()
            df[c].fillna(mean, inplace=True)
        elif df[c].nunique(dropna=False) == 1:
            # 只有一个值
            drop_cols.append(c)
        else:
            # 只有一个值 + 一个空值
            mean = df[c].mean()
            df[c].fillna(mean-1, inplace=True)

    if len(drop_cols) >=1:
        df.drop(drop_cols, axis=1, inplace=True)

    for c in cat_features:
        df[c].fillna("0", inplace=True)

    for c in multicat_features:
        df[c].fillna("0", inplace=True)

    if fill_time:
        print('fill time!')
        for c in time_features:
            name = CONSTANT.CATEGORY_PREFIX + c + '_isnull'
            df[name] = df[c].isnull().apply(str)


@timeit
def num_cleaner(df, add_seri_list, train_data_length):
    df_columns = [c for c in list(df.columns) if (c.startswith(CONSTANT.NUMERICAL_PREFIX))]
    add_df_columns = [c.name for c in add_seri_list if (c.name.startswith(CONSTANT.NUMERICAL_PREFIX))]
    add_df_all_columns = [c.name for c in add_seri_list]

    n_features = df_columns + add_df_columns
    if len(n_features) > 1:
        print(n_features)

        for col in n_features[::-1]:
            if col in df_columns:
                if df[col].head(train_data_length).nunique() == 1:
                    df.drop(col, axis=1, inplace=True)
                    print('drop', col)
            else:
                col_index = add_df_all_columns.index(col)
                if add_seri_list[col_index].head(train_data_length).nunique() == 1:
                    del add_seri_list[col_index]
                    print(col)



@timeit
def feature_engineer(df, config):
    print('0', df.shape)
    #td_avg(df, config)
    t_delta_extract(df, config)
    timeExtract(df, config)
    feature_data_list = MV_Combine_Divide(df)
    print('1', len(feature_data_list) + df.shape[1])
    feature_data_list = MV_Feature_Extract(df, feature_data_list)
    print('2', len(feature_data_list) + df.shape[1])
    multicat_sample(df, feature_data_list)
    print('3', len(feature_data_list) + df.shape[1])

    feature_data_list = multi_feature_cnt(df, feature_data_list)

    print('4', len(feature_data_list) + df.shape[1])
    feature_data_list = cat_feature_cnt(df, feature_data_list)
    print('5', len(feature_data_list) + df.shape[1])

    num_cleaner(df, feature_data_list, config['train_data_length'])

    feature_data = pd.concat(feature_data_list, axis=1, copy=False)
    return df, feature_data


