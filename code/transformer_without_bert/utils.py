
import os
import pandas as pd

import os
import pandas as pd
import numpy as np
import torch


# Original
# STOCK_DIR   = "data/enriched/sp500/csv"
# MKT_PATH    = "data/enriched/market_indexes_aggregated.csv"

# Normalized
STOCK_DIR   = "data/normalized/sp500/csv"
MKT_PATH    = "data/normalized/market_indexes_aggregated_normalized.csv"


# This loads all CSV files in the given folder into a tensor of shape (N, T, K),
# Without market information
def load_all_csv_data_without_index(data_folder=STOCK_DIR):
    """
    Loads all CSV files in the given folder into a tensor of shape (N, T, K),
    where N = number of stocks (files), T = timesteps, K = features.
    Assumes all CSVs have the same columns and timesteps.
    Returns: torch.Tensor of shape (N, T, K), list of stock names, list of feature names
    """
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    csv_files.sort()
    data_list = []
    stock_names = []
    feature_names = None

    for fname in csv_files:
        fpath = os.path.join(data_folder, fname)
        df = pd.read_csv(fpath)
        # If 'Date' is a column, drop it or set index
        if 'Date' in df.columns:
            df = df.drop(columns=['Date'])
        if 'Stock_symbol' in df.columns:
            df = df.drop(columns=['Stock_symbol'])
        if feature_names is None:
            feature_names = df.columns.tolist()
        data_list.append(df.values)
        stock_names.append(os.path.splitext(fname)[0])

    # Stack into (N, T, K)
    data_tensor = torch.tensor(np.stack(data_list), dtype=torch.float)
    return data_tensor, stock_names, feature_names




def load_all_csv_data_with_market_indexes(
    data_folder=STOCK_DIR,
    market_indexes_path=MKT_PATH
):
    """
    Loads all CSV files in the given folder into a tensor of shape (N, T, K),
    where N = number of stocks (files), T = timesteps, K = features.
    Concatenates market index features (same for all stocks) to each timestep.
    Assumes all CSVs and market_indexes_aggregated.csv are aligned and have the same dates in the same order.
    Returns: torch.Tensor of shape (N, T, K), list of stock names, list of feature names
    """
    # Load market indexes and drop Date column
    market_indexes = pd.read_csv(market_indexes_path)
    market_indexes_features = market_indexes.drop(columns=['Date']).values
    market_indexes_feature_names = [col for col in market_indexes.columns if col != 'Date']

    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    # csv_files.sort()
    # Shuffle the files
    np.random.shuffle(csv_files)
    data_list = []
    stock_names = []
    feature_names = None

    for fname in csv_files:
        fpath = os.path.join(data_folder, fname)
        df = pd.read_csv(fpath)
        # Drop Date and Stock_symbol columns if present
        if 'Date' in df.columns:
            df = df.drop(columns=['Date'])
        if 'Stock_symbol' in df.columns:
            df = df.drop(columns=['Stock_symbol'])
        # Concatenate stock features and market features (aligned by row order)
        combined = np.concatenate([df.values, market_indexes_features], axis=1)
        if feature_names is None:
            feature_names = df.columns.tolist() + market_indexes_feature_names
        data_list.append(combined)
        stock_names.append(os.path.splitext(fname)[0])

    # Stack into (N, T, K)
    data_tensor = torch.tensor(np.stack(data_list), dtype=torch.float)
    return data_tensor, stock_names, feature_names






# market_indexes = pd.read_csv('data/enriched/market_indexes_aggregated.csv')
# print("Market indexes shape:", market_indexes.shape)








# Qlib
########################################################################
import os, pandas as pd, numpy as np



def csvs_to_qlib_df(stock_dir=STOCK_DIR, mkt_file=MKT_PATH):
    # --- read market indices once -----------------------
    mkt = (pd.read_csv(mkt_file)
             .assign(datetime=lambda d: pd.to_datetime(d["Date"]))
             .drop(columns=["Date"])
             .set_index("datetime"))

    frames = []
    for fn in os.listdir(stock_dir):
        if not fn.endswith(".csv"): continue
        sym = os.path.splitext(fn)[0]          # “AAPL”, “MSFT”, …

        df  = (pd.read_csv(os.path.join(stock_dir, fn))
                 .drop(columns=["Stock_symbol"])
                 .assign(datetime=lambda d: pd.to_datetime(d["Date"]))
                 .drop(columns=["Date"])
                 .merge(mkt, on="datetime"))

        # simple next-day return label – replace by your own definition if needed
        df["LABEL"] = df["Close"].pct_change().shift(-1)

        df["instrument"] = sym
        frames.append(df.set_index(["datetime", "instrument"]))

    raw = (pd.concat(frames)
             .sort_index())          # MultiIndex  (datetime, instrument)

    return raw




class PandasDataLoader:
    """
    Minimal loader for DataHandlerLP that serves an in-memory
    (datetime, instrument) multi-indexed DataFrame.
    """
    def __init__(self, df):
        self._df = df

    # signature expected by DataHandler.load
    def load(self, instruments=None, fields=None,
             start_time=None, end_time=None, freq="day"):
        # Ignore freq because we already aligned to 'day'.
        df = self._df
        if instruments is not None:
            if isinstance(instruments, str):
                instruments = [instruments]
            df = df.loc[pd.IndexSlice[:, instruments], :]

        if start_time is not None:
            df = df.loc[pd.IndexSlice[start_time:], :]
        if end_time is not None:
            df = df.loc[pd.IndexSlice[:end_time], :]

        if fields is not None:
            df = df[fields]

        return df