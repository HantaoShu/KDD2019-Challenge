import CONSTANT
from CONSTANT import TIME_COL, TD_AVG_SUFFIX, TIME_DELTA, HOUR_SUFFIX, WKD_SUFFIX, MONTH_SUFFIX, MIN_SUFFIX, \
    TIME_PREFIX, TIME_SUFFIX, MIN_DAY_CONSIDER, MIN_MONTH_CONSIDER, MIN_HOUR_CONSIDER
from utility import log, timeit, Timer


@timeit
def td_avg(df, config):
    df.sort_values(config[TIME_COL], inplace=True)
    time_delta_avg = CONSTANT.NUMERICAL_PREFIX + TD_AVG_SUFFIX
    df[time_delta_avg] = df[config[TIME_COL]].diff().dt.total_seconds().rolling(5).mean().fillna(0)
    df.sort_index(inplace=True)


@timeit
def t_delta_extract(df, config):
    table_time_cols = [c for c in df if (c.startswith(TIME_PREFIX))]
    log(table_time_cols)
    for c in table_time_cols:
        # 不使用副表的t_01
        if df[c].nunique() == 0:
            df.drop(c, axis=1, inplace = True)
        else:
            if config[TIME_COL] == c: # ==
                continue
            tmp = (df[c] - df[config[TIME_COL]]).dt.total_seconds()/60
            mean = tmp.mean()
            tmp = tmp.fillna(mean)
            df[CONSTANT.NUMERICAL_PREFIX + c + TIME_DELTA] = tmp


@timeit
def timeExtract(df, config):
    print(df.keys())
    time_cols = [c for c in list(df.keys()) if c.startswith(TIME_PREFIX)]

    for c in config['test_time']:
        tmp = config['test_time'][c]
        if tmp['hour'] >= MIN_HOUR_CONSIDER:
            df[CONSTANT.CATEGORY_PREFIX + TIME_SUFFIX + c + HOUR_SUFFIX] = df[c].dt.hour.apply(str).astype(
                'category')
        if tmp['day'] >= MIN_DAY_CONSIDER:
            df[CONSTANT.CATEGORY_PREFIX + TIME_SUFFIX + c + WKD_SUFFIX] = df[c].dt.weekday.apply(str).astype(
                'category')
        if tmp['month'] >= MIN_MONTH_CONSIDER:
            df[CONSTANT.CATEGORY_PREFIX + TIME_SUFFIX + c + MONTH_SUFFIX] = df[c].dt.month.apply(str).astype(
                'category')
        df[CONSTANT.CATEGORY_PREFIX + TIME_SUFFIX + c + MIN_SUFFIX] = df[c].dt.minute.apply(str).astype('category')

    print(time_cols)
    for c in time_cols:
        if 'table' in c:
            tot_second = (df.iloc[config[CONSTANT.TRAIN_DATA_LENGTH]:][c].max() - df.iloc[config[CONSTANT.TRAIN_DATA_LENGTH]:][c].min()).total_seconds()
            log(('time last', c, tot_second / 3600, tot_second / 3600 / 24, tot_second / 3600 / 24 / 30))
            if tot_second / 3600 >= 12:
                df[CONSTANT.CATEGORY_PREFIX + TIME_SUFFIX + c + HOUR_SUFFIX] = df[c].dt.hour.apply(str).astype(
                    'category')
            if tot_second / 3600 / 24 >= MIN_DAY_CONSIDER:
                df[CONSTANT.CATEGORY_PREFIX + TIME_SUFFIX + c + WKD_SUFFIX] = df[c].dt.weekday.apply(str).astype(
                    'category')
            if tot_second / 3600 / 24 / 30 >= MIN_MONTH_CONSIDER:
                df[CONSTANT.CATEGORY_PREFIX + TIME_SUFFIX + c + MONTH_SUFFIX] = df[c].dt.month.apply(str).astype(
                    'category')
        df[CONSTANT.CATEGORY_PREFIX + TIME_SUFFIX + c + MIN_SUFFIX] = df[c].dt.minute.apply(str).astype('category')
    df.drop(time_cols, axis=1, inplace=True)