import streamlit as st
import pandas as pd
from functools import lru_cache

def required(*args):
    for a in args:
        if a is None or (isinstance(a, str) and not a.strip()):
            st.stop()

@lru_cache(maxsize=64)
def cache_df(key: str, loader):
    df = loader()
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    return df

def clear_cache():
    try:
        cache_df.cache_clear()
    except Exception:
        pass
