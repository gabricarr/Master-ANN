
import os
import pandas as pd

import os
import pandas as pd
import numpy as np
import torch

# This loads all CSV files in the given folder into a tensor of shape (N, T, K),
# Without market information
def load_all_csv_data_without_index(data_folder='data/enriched/sp500/csv'):
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
    data_folder='data/enriched/sp500/csv',
    market_indexes_path='data/enriched/market_indexes_aggregated.csv'
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




