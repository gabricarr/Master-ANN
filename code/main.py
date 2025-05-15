from master_bert import MASTERModel
import pickle
import numpy as np
import time

from utils import load_all_csv_data_with_market_indexes, load_all_csv_data_without_index, csvs_to_qlib_df, PandasDataLoader
# Please install qlib first before load the data.

# Qlib
# import qlib
# from qlib.config import REG_US           # S&P 500 is a US market
# qlib.init(provider_uri=".", region=REG_US)   # provider_uri just needs to exist





# ------------------------------------------------------------
# 1.  Init Qlib and build *one* handler
import qlib, pandas as pd, numpy as np, torch
qlib.init()                               # client mode is fine

from qlib.data.dataset.loader import StaticDataLoader
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset import TSDatasetH          # <-- here
from qlib.data.dataset.processor import (
    DropnaProcessor, CSZScoreNorm, DropnaLabel,
)

# your tensor, names, dates exactly as before  ----------------
# stock_tensor, stock_names, feature_names = load_all_csv_data_without_index()
stock_tensor, stock_names, feature_names = load_all_csv_data_with_market_indexes()
N, T, K   = stock_tensor.shape
print("Shape: ", stock_tensor.shape)
# dates     = pd.read_csv("data/enriched/market_indexes_aggregated.csv")["Date"]
dates = pd.to_datetime(                     # <-- NEW
    pd.read_csv("data/enriched/market_indexes_aggregated.csv")["Date"]
)

# tensor ➜ tidy multi-index frame --------------------------------
def tensor_to_df(tensor, inst, feats, dt_index):
    flat = tensor.numpy().reshape(N * T, K)
    idx  = pd.MultiIndex.from_product([dt_index, inst],
                                      names=["datetime", "instrument"])
    cols = pd.MultiIndex.from_product([["feature"], feats])
    return pd.DataFrame(flat, index=idx, columns=cols)

df_raw = tensor_to_df(stock_tensor, stock_names, feature_names, dates)

# optional: build a forward-return label
df_raw[("label", "FWD_RET")] = (
    df_raw[("feature", "Adjusted Close")]
      .groupby("instrument").shift(-1) / df_raw[("feature", "Adjusted Close")] - 1
)

# handler with learn / infer processors ------------------------
# proc_feat = [
#     {"class": "DropnaProcessor", "kwargs": {"fields_group": "feature"}},
#     {"class": "CSZScoreNorm",   "kwargs": {"fields_group": "feature"}},
# ]

# proc_feat = [
#     {"class": "CSZScoreNorm",   "kwargs": {"fields_group": "feature"}},
# ]

proc_feat = [
    {"class": "Fillna",          # <— correct name
     "kwargs": {"fields_group": "feature", "fill_value": 0}},  # zero-fill; choose ffill/bfill/etc. if you like
    {"class": "CSZScoreNorm",
     "kwargs": {"fields_group": "feature"}},
]

proc_label = [{"class": "DropnaLabel"}]

handler = DataHandlerLP(
    data_loader      = StaticDataLoader(df_raw),
    infer_processors = proc_feat,          # what the model will see later
    learn_processors = proc_feat + proc_label,
)
handler.fit_process_data()                 # learn z-scores, etc.

# ------------------------------------------------------------
# 2.  Attach time splits in a TSDatasetH
split = {
    "train": (dates.iloc[0],              dates.iloc[int(T*0.8) - 1]),
    "valid": (dates.iloc[int(T*0.8)],     dates.iloc[int(T*0.9) - 1]),
    "test" : (dates.iloc[int(T*0.9)],     dates.iloc[-2]),
}

ts_ds = TSDatasetH(
    handler  = handler,
    segments = split,
    step_len = 8,          # same window the MASTER code expects
)

dl_train = ts_ds.prepare("train")   # ➜ TSDataSampler
dl_valid = ts_ds.prepare("valid")
dl_test  = ts_ds.prepare("test")

print(len(dl_train), len(dl_valid), len(dl_test))
#  → continue with your for-loop over seeds exactly as before
# ------------------------------------------------------------





#############################################################################
# # Pure tensor data
# # Load data without market indexes
# # stock_data, stock_names, features_names = load_all_csv_data_without_index()

# # Load data with market indexes
# stock_data, stock_names, features_names = load_all_csv_data_with_market_indexes()

# # Print the shape of the data
# if stock_data is not None:
#     print("Data shape:", stock_data.shape)
#     print("Data loaded successfully.")


# # Size without market indexes: 224
# # Size with market indexes: 276


# # Split into train, val, test (80%, 10%, 10%)
# N = stock_data.shape[0]
# train_end = int(N * 0.8)
# val_end = int(N * 0.9)

# dl_train = stock_data[:train_end]
# dl_valid = stock_data[train_end:val_end]
# dl_test = stock_data[val_end:]

# print("Train shape:", dl_train.shape)
# print("Val shape:", dl_valid.shape)
# print("Test shape:", dl_test.shape)



#######################################################################################
# Load data with market indexes
# data_tensor, stock_names, feature_names = load_all_csv_data_with_market_indexes()

# print("Data shape:", data_tensor.shape)

# import pandas as pd
# # Get dates from one of the CSVs (assuming all have the same dates)
# sample_csv = 'data/enriched/market_indexes_aggregated.csv'
# dates = pd.read_csv(sample_csv)['Date'].tolist()








# exit(1)



# universe = 'sp500' # ['csi300','csi800']
# prefix = 'opensource' # ['original','opensource'], which training data are you using
# train_data_dir = f'data'
# with open(f'{train_data_dir}\{prefix}\{universe}_dl_train.pkl', 'rb') as f:
#     dl_train = pickle.load(f)

# predict_data_dir = f'data\opensource'
# with open(f'{predict_data_dir}\{universe}_dl_valid.pkl', 'rb') as f:
#     dl_valid = pickle.load(f)
# with open(f'{predict_data_dir}\{universe}_dl_test.pkl', 'rb') as f:
#     dl_test = pickle.load(f)

# print("Data Loaded.")

universe = 'sp500'
d_feat = 224
d_model = 256
t_nhead = 4
s_nhead = 2
dropout = 0.5
gate_input_start_index = 224
gate_input_end_index = 276

if universe == 'sp500':
    beta = 5
else:
    raise ValueError("Invalid universe")

n_epoch = 1
lr = 1e-5
GPU = 0
train_stop_loss_thred = 0.95


ic = []
icir = []
ric = []
ricir = []

# Training
######################################################################################
for seed in [0, 1]: #[0, 1, 2, 3, 4]:
    model = MASTERModel(
        d_feat = d_feat, d_model = d_model, t_nhead = t_nhead, s_nhead = s_nhead, T_dropout_rate=dropout, S_dropout_rate=dropout,
        beta=beta, gate_input_end_index=gate_input_end_index, gate_input_start_index=gate_input_start_index,
        n_epochs=n_epoch, lr = lr, GPU = GPU, seed = seed, train_stop_loss_thred = train_stop_loss_thred,
        save_path='model', save_prefix=f'{universe}'
    )

    start = time.time()
    # Train
    model.fit(dl_train, dl_valid)

    print("Model Trained.")

    # Test
    predictions, metrics = model.predict(dl_test)
    
    running_time = time.time()-start
    
    print('Seed: {:d} time cost : {:.2f} sec'.format(seed, running_time))
    print(metrics)

    ic.append(metrics['IC'])
    icir.append(metrics['ICIR'])
    ric.append(metrics['RIC'])
    ricir.append(metrics['RICIR'])
######################################################################################


exit(1)



# Load and Test
######################################################################################
for seed in [0]:
    param_path = f'model\{universe}_{prefix}_{seed}.pkl'

    print(f'Model Loaded from {param_path}')
    model = MASTERModel(
            d_feat = d_feat, d_model = d_model, t_nhead = t_nhead, s_nhead = s_nhead, T_dropout_rate=dropout, S_dropout_rate=dropout,
            beta=beta, gate_input_end_index=gate_input_end_index, gate_input_start_index=gate_input_start_index,
            n_epochs=n_epoch, lr = lr, GPU = GPU, seed = seed, train_stop_loss_thred = train_stop_loss_thred,
            save_path='model/', save_prefix=universe
        )
    model.load_param(param_path)
    predictions, metrics = model.predict(dl_test)
    print(metrics)

    ic.append(metrics['IC'])
    icir.append(metrics['ICIR'])
    ric.append(metrics['RIC'])
    ricir.append(metrics['RICIR'])
    
######################################################################################

print("IC: {:.4f} pm {:.4f}".format(np.mean(ic), np.std(ic)))
print("ICIR: {:.4f} pm {:.4f}".format(np.mean(icir), np.std(icir)))
print("RIC: {:.4f} pm {:.4f}".format(np.mean(ric), np.std(ric)))
print("RICIR: {:.4f} pm {:.4f}".format(np.mean(ricir), np.std(ricir)))