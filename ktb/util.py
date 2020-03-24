from typing import List, NoReturn, Union, Tuple, Optional, Text, Generic, Callable, Dict

import pandas as pd
import os
import pickle
import gzip
import numpy as np
import random as rn
from datetime import datetime

SEED = 777

output_dir = "../Results"

def write_submission(df, cols = None):
    if cols is None:
        cols = df.columns
    time_now = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S")
    df[cols].to_csv(os.path.join(output_dir,f'submission-{time_now}.csv'), index=False, float_format='%.4f')
    
def read_object(file):
    with gzip.open(file, "rb") as f:
        return pickle.load(f)

def write_object(file, obj):
    with gzip.open(file, "wb") as f:
        pickle.dump(obj, f)

def cache_func(func,key):
    if not os.path.exists(f"cache"):
        os.mkdir("cache")
    key = key+func.__name__
    def inner_func(*args, **kwargs):
        try: 
            if os.path.exists(f"cache/{key}"):
                return read_object(f"cache/{key}")
        except:
            pass
        obj = func(*args, **kwargs)
        write_object(f"cache/{key}", obj)
        return obj
    return inner_func

def seed_everything(seed):
    rn.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def reduce_mem_usage(df: pd.DataFrame,
                     verbose: bool = True) -> pd.DataFrame:
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if (c_min > np.iinfo(np.int8).min
                        and c_max < np.iinfo(np.int8).max):
                    df[col] = df[col].astype(np.int8)
                elif (c_min > np.iinfo(np.int16).min
                      and c_max < np.iinfo(np.int16).max):
                    df[col] = df[col].astype(np.int16)
                elif (c_min > np.iinfo(np.int32).min
                      and c_max < np.iinfo(np.int32).max):
                    df[col] = df[col].astype(np.int32)
                elif (c_min > np.iinfo(np.int64).min
                      and c_max < np.iinfo(np.int64).max):
                    df[col] = df[col].astype(np.int64)
            else:
                if (c_min > np.finfo(np.float16).min
                        and c_max < np.finfo(np.float16).max):
                    df[col] = df[col].astype(np.float16)
                elif (c_min > np.finfo(np.float32).min
                      and c_max < np.finfo(np.float32).max):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    reduction = (start_mem - end_mem) / start_mem
    msg = f'Mem. usage decreased to {end_mem:5.2f} MB ({reduction * 100:.1f} % reduction)'
    if verbose:
        print(msg)
    return df

def maddest(d : Union[np.array, pd.Series, List], axis : Optional[int]=None) -> np.array:  
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def batching(df : pd.DataFrame,
             batch_size : int,
             add_index : Optional[bool]=True) -> pd.DataFrame :

    df['batch_'+ str(batch_size)] = df.groupby(df.index//batch_size, sort=False)[df.columns[0]].agg(['ngroup']).values + 1
    df['batch_'+ str(batch_size)] = df['batch_'+ str(batch_size)].astype(np.uint16)
    if add_index:
        df['batch_' + str(batch_size) +'_idx'] = df.index  - (df['batch_'+ str(batch_size)] * batch_size)
        df['batch_' + str(batch_size) +'_idx'] = df['batch_' + str(batch_size) +'_idx'].astype(np.uint16)
    return df

def flattern_values(obj, func=None):
    res = []
    if isinstance(obj, dict):
        for v in obj.values:
            res.extend(flattern_values(v, func))
    elif isinstance(obj, list):
        for v in obj:
            res.extend(flattern_values(v, func))
    else:
        if func is not None:
            res.extend(flattern_values(func(obj), None))
        else:
            res.append(obj)

    return res


def apply2values(obj, func):
    res = None
    if isinstance(obj, dict):
        res = {k:apply2values(v, func) for k,v in obj.items}
    elif isinstance(obj, list):
        res = [apply2values(v, func) for v in obj]
    else:
        res = func(obj)
    return res

seed_everything(SEED)
