from collections import defaultdict, deque
import numpy as np
import pandas as pd
import gc
import CONSTANT
from utility import Timer, log, timeit, feature_map, aggregate_op
from concurrent.futures import ProcessPoolExecutor


NUM_OP = [np.std, np.mean,np.median,]

def bfs(root_name, graph, tconfig):
    tconfig[CONSTANT.MAIN_TABLE_NAME]['depth'] = 0
    queue = deque([root_name])
    while queue:
        u_name = queue.popleft()
        for edge in graph[u_name]:
            v_name = edge['to']
            if 'depth' not in tconfig[v_name]:
                tconfig[v_name]['depth'] = tconfig[u_name]['depth'] + 1
                queue.append(v_name)

def hash_key_divide(df, key):
    # 按照key整切df 使得大小几乎均等
    df['sort_index'] = np.arange(len(df)) + 1
    idx_values= list(df.drop_duplicates(subset=[key], keep='last')['sort_index'].values)
    sp_25, sp_50, sp_75 = np.percentile(idx_values, [25, 50, 75])
    idx_25 = idx_values[np.argmin(abs(idx_values - sp_25))]
    idx_50 = idx_values[np.argmin(abs(idx_values - sp_50))]
    idx_75 = idx_values[np.argmin(abs(idx_values - sp_75))]
    df.drop('sort_index',axis = 1, inplace = True)
    df_25 = df.iloc[:idx_25, :]
    df_50 = df.iloc[idx_25 : idx_50, :]
    df_75 = df.iloc[idx_50 : idx_75, :]
    df_100 = df.iloc[idx_75: , :]
    return [df_25, df_50, df_75, df_100]


def join_agg_func(df, rehash_key):
    agg_funcs = {col: aggregate_op(col) for col in df if col != rehash_key}
    for col in df:
        if col != rehash_key:
            agg_funcs[col] = agg_funcs[col] + ['count']
            break
    return df.groupby(rehash_key).agg(agg_funcs)

@timeit
def join(u, v, v_name, key, type_):
    v = v.copy()
    if isinstance(key, list):
        assert len(key) == 1
        key = key[0]

    if type_.split("_")[2] == 'many':
        v.sort_values(key, inplace=True)
        split_tmp_u_list = hash_key_divide(v, key)

        ex = ProcessPoolExecutor(4)
        objs = []
        for d in split_tmp_u_list:
            objs.append(ex.submit(join_agg_func, d, key))
        ex.shutdown(wait=True)

        pool_result = []
        for obj in objs:
            pool_result.append(obj.result())

        v = pd.concat(pool_result, axis=0)
        print(f'after usage {len(v)}')

        del objs
        del pool_result

        gc.collect()

        v.columns = v.columns.map(lambda a:
                f"{feature_map(a[0], a[1])}{a[1].upper()}({a[0]})")
    else:
        v = v.set_index(key)
    v.columns = v.columns.map(lambda a: f"{a.split('_', 1)[0]}_{v_name}.{a}")

    return u.join(v, on=key)


def temporal_join_agg_f(df, rehash_key):
    agg_funcs = {col: aggregate_op(col) for col in df if col != rehash_key}
    for col in df:
        if col != rehash_key:
            agg_funcs[col] = agg_funcs[col] + ['count']
            break
    return df.groupby(rehash_key).agg(agg_funcs)

@timeit
def temporal_join(u, v, v_name, key, time_col, config, type_):
    v = v.copy()
    timer = Timer()

    if isinstance(key, list):
        assert len(key) == 1
        key = key[0]

    tmp_u = u[[time_col, key]]
    timer.check("select")
    # ---------------------------------- Drop 多余的行 ----------------------------
    # 注意到附表的key如果在主表没有出现，可以删除附表中多余的行
    # 这对结果不造成影响, 且降低时序join的复杂度
    log(('Duplicate Key', len(set(v[key]) - set(u[key]))))
    log('*' * 10)
    log(('before clean duplicate', len(v)))
    tmp_v_for_delete = v[[key]]
    tmp_v_for_delete['my_index'] = v.index
    keep_index = set(pd.merge(tmp_v_for_delete, tmp_u.drop_duplicates(subset=[key]), on=key, how='inner', left_index=True)['my_index'])
    delete_index = set(v.index) - keep_index
    v.drop(delete_index, inplace=True)
    log(('after clean duplicate', len(v)))
    log('*' * 10)
    del tmp_v_for_delete
    timer.check("clean duplicate")

    print(f'Final length of v is {len(v)}')

    if type_.split("_")[2] == 'many':
        print(f'before usage {len(v)}')
        v.sort_values([key, time_col], ascending=False, inplace=True)
        split_tmp_u_list = hash_key_divide(v, key)

        del v
        gc.collect()

        ex = ProcessPoolExecutor(4)
        objs = []

        for d in split_tmp_u_list:
            objs.append(ex.submit(temporal_join_agg_f, d, key))
        ex.shutdown(wait=True)

        pool_result = []
        for obj in objs:
            pool_result.append(obj.result())

        v = pd.concat(pool_result, axis=0)
        print(f'after usage {len(v)}')
        #
        del objs
        del pool_result
        gc.collect()
        v.columns = v.columns.map(lambda a:
                                  f"{feature_map(a[0], a[1])}_FT_{a[1].upper()}({a[0]})")
        timer.check("to many join")
    else:
        v = v.set_index(key)
    v.columns = v.columns.map(lambda a: f"{a.split('_', 1)[0]}_{v_name}.{a}")


    return u.join(v, on=key)




def dfs(u_name, config, tables, graph):
    u = tables[u_name]
    log(f"enter {u_name}")
    for edge in graph[u_name]:
        v_name = edge['to']
        if config['tables'][v_name]['depth'] <= config['tables'][u_name]['depth']:
            continue

        v = dfs(v_name, config, tables, graph)
        key = edge['key']
        type_ = edge['type']

        if config['time_col'] not in u and config['time_col'] in v:
            log(f"join {u_name} <--{type_}--nt {v_name}")
            u = join(u, v, v_name, key, type_)

        elif config['time_col'] in u and config['time_col'] in v:
            log(f"join {u_name} <--{type_}--t {v_name}")
            u = temporal_join(u, v, v_name, key, config['time_col'], config, type_)
        else:
            log(f"join {u_name} <--{type_}--nt {v_name}")
            u = join(u, v, v_name, key, type_)

        del v

    log(f"leave {u_name}")
    return u


@timeit
def merge_table(tables, config):
    graph = defaultdict(list)
    for rel in config['relations']:
        ta = rel['table_A']
        tb = rel['table_B']
        graph[ta].append({
            "to": tb,
            "key": rel['key'],
            "type": rel['type']
        })
        graph[tb].append({
            "to": ta,
            "key": rel['key'],
            "type": '_'.join(rel['type'].split('_')[::-1])
        })
    bfs(CONSTANT.MAIN_TABLE_NAME, graph, config['tables'])
    return dfs(CONSTANT.MAIN_TABLE_NAME, config, tables, graph)
