import time
from typing import Any
import CONSTANT
import statistics
import numpy as np

nesting_level = 0
is_start = None

class Timer:
    def __init__(self):
        self.start = time.time()
        self.history = [self.start]

    def check(self, info):
        current = time.time()
        log(f"[{info}] spend {current - self.history[-1]:0.2f} sec")
        self.history.append(current)

def timeit(method, start_log=None):
    def timed(*args, **kw):
        global is_start
        global nesting_level

        if not is_start:
            print()

        is_start = True
        log(f"Start [{method.__name__}]:" + (start_log if start_log else ""))
        nesting_level += 1

        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()

        nesting_level -= 1
        log(f"End   [{method.__name__}]. Time elapsed: {end_time - start_time:0.2f} sec.")
        is_start = False

        return result

    return timed


def log(entry: Any):
    global nesting_level
    space = "-" * (4 * nesting_level)
    print(f"{space}{entry}")

class Config:
    def __init__(self, info):
        self.data = {
            "start_time": time.time(),
            **info
        }
        self.data["tables"] = {}
        for tname, ttype in info['tables'].items():
            self.data['tables'][tname] = {}
            self.data['tables'][tname]['type'] = ttype


    def time_left(self):
        print(time.time(),self["start_time"],self['time_budget'])
        return self["time_budget"] - (time.time() - self["start_time"])

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def __contains__(self, key):
        return key in self.data

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return repr(self.data)

def feature_map(col_name, func_name):
    if col_name.startswith(CONSTANT.NULL_COUNT_PREFIX):
        return CONSTANT.NULL_COUNT_PREFIX
    if func_name in ['mean', 'std', 'sum', 'skew', 'count', 'nunique', 'median', 'duplicat']:
        return CONSTANT.NUMERICAL_PREFIX
    elif func_name in ['mode']:
        return CONSTANT.CATEGORY_PREFIX
    elif func_name in ['cat_2_multi_cat']:
        return CONSTANT.MULTI_CAT_PREFIX + '_c2m'
    elif func_name in ['multi_cat_2_multi_cat']:
        return CONSTANT.MULTI_CAT_PREFIX + '_m2m'
    elif func_name in ['max', 'min']:
        if col_name.startswith(CONSTANT.TIME_PREFIX):
            return CONSTANT.TIME_PREFIX
        elif col_name.startswith(CONSTANT.NUMERICAL_PREFIX):
            return CONSTANT.NUMERICAL_PREFIX
        else:
            assert False, f"{col_name}-{func_name} No match feature map"
    else:
        assert False, f"{col_name}-{func_name}No match feature map"

def aggregate_op(col):
    def cat_join_into_multi_cat(x):
        return ','.join(x)

    def multi_cat_join_into_multi_cat(x):
        return ';'.join(x)

    def my_nunique(x):
        return x.nunique()

    def my_duplicat(x):
        return x.nunique()

    my_nunique.__name__ = 'nunique'
    my_duplicat.__name__ = 'duplicat'
    cat_join_into_multi_cat.__name__ = 'cat_2_multi_cat'
    multi_cat_join_into_multi_cat.__name__ = 'multi_cat_2_multi_cat'
    ops = {
        CONSTANT.NUMERICAL_TYPE: ["mean", "sum", "std"],#]# "median"],
        CONSTANT.CATEGORY_TYPE: [cat_join_into_multi_cat],
        CONSTANT.TIME_TYPE: ["max", "min"],
        CONSTANT.MULTI_CAT_TYPE: [multi_cat_join_into_multi_cat],
        CONSTANT.NULL_COUNT_TYPE: ['mean']
    }
    # is not mode, online test will be 100s
    # add mode
    if col.startswith(CONSTANT.NUMERICAL_PREFIX):
        return ops[CONSTANT.NUMERICAL_TYPE]
    if col.startswith(CONSTANT.CATEGORY_PREFIX):
        return ops[CONSTANT.CATEGORY_TYPE]
    if col.startswith(CONSTANT.MULTI_CAT_PREFIX):
         return ops[CONSTANT.MULTI_CAT_TYPE]
    if col.startswith(CONSTANT.TIME_PREFIX):
        return ops[CONSTANT.TIME_TYPE]
    if col.startswith(CONSTANT.NULL_COUNT_PREFIX):
        return ops[CONSTANT.NULL_COUNT_TYPE]


def get_mode(x):
    try:
        return statistics.mode(x)
    except:
        try:
            return x[1]
        except:
            return '0'

@timeit
def table_memory_cut(props):
    timer = Timer()
    start_mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage of properties dataframe is :", start_mem_usg, " MB")
    NAlist = []  # Keeps track of columns that have missing values filled in.
    for col in props.columns:
        if col.startswith(CONSTANT.NUMERICAL_PREFIX):
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()

            # test if column can be converted to an integer
            asint = props[col].astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)

                    # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)
        if (col.startswith('c_') | col.startswith('m_')):
            props[col] = props[col].astype('category')

    timer.check('trans done')

    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage is: ", mem_usg, " MB")
    print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")
    return props


@timeit
def pre_cut_memory(props):
    start_mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage of properties dataframe is :", start_mem_usg, " MB")
    for col in props.columns:
        if col.startswith(CONSTANT.NUMERICAL_PREFIX):  # Exclude strings
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()

            # test if column can be converted to an integer
            asint = props[col].astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)

                        # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)

    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage is: ", mem_usg, " MB")
    print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")
    return props

# def rechange_data_type(props):
#     for col in props.columns:
#         if (col.startswith('c_') | col.startswith('m_')) :
#             props[col] = props[col].astype('category')
#     return props

def cut_by_ith_char(str, char, ith):
    if str.find(char) == -1:
        return str
    else:
        count = 0
        for i, c in enumerate(str):
            if c == char:
                count += 1
            if count == ith:
                break
        if count >= ith:
            return str[:i]
        else:
            return str


