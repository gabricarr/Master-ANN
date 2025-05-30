from master import MASTERModel
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
# dates = pd.to_datetime(                     # <-- NEW
#     pd.read_csv("data/enriched/market_indexes_aggregated.csv")["Date"]
# )

dates = pd.to_datetime(                     # <-- NEW
    pd.read_csv("data/normalized/market_indexes_aggregated_normalized.csv")["Date"]
)

# tensor ➜ tidy multi-index frame --------------------------------
def tensor_to_df(tensor, inst, feats, dt_index):
    flat = tensor.numpy().reshape(N * T, K)
    idx  = pd.MultiIndex.from_product([dt_index, inst],
                                      names=["datetime", "instrument"])
    cols = pd.MultiIndex.from_product([["feature"], feats])
    return pd.DataFrame(flat, index=idx, columns=cols)

df_raw = tensor_to_df(stock_tensor, stock_names, feature_names, dates)

# # OLD: build a forward-return label
# df_raw[("label", "FWD_RET")] = (
#     df_raw[("feature", "Adjusted Close")]
#       .groupby("instrument").shift(-1) / df_raw[("feature", "Adjusted Close")] - 1
# )

# last_date = dates.iloc[-1]
# df_raw = df_raw.drop(index=last_date, level="datetime")


# MASTER uses a d-day rank-normalized return, which reflects each stock's relative performance within the market at a specific date
# Steps: 
# Look Ahead:
# For each stock, MASTER looks a few days into the future (like 5 days) to see how much the price goes up or down.

# Calculate Return:
# It calculates the percentage change in price over those days — this is the raw return.

# Compare Stocks:
# On each day, it compares the returns of all stocks to see which ones performed better or worse.

# Z-score Normalization:
# It transforms those returns into standard scores (z-scores), so you know how each stock ranks relative to the others that day.

# Final Label:
# The model learns to predict this ranked performance score, not just the raw return.

# Oss: “The lookback window length T and prediction interval d are set as 8 and 5 respectively.” -- MaSTER paper

# Step 1: Compute d-day forward return
d = 5  # prediction interval
df_raw[("label", "FWD_RET")] = (
    df_raw[("feature", "Adjusted Close")]
      .groupby("instrument")
      .shift(-d) / df_raw[("feature", "Adjusted Close")] - 1
)

# Drop the last d rows since they can't have valid forward returns
for i in range(d):
    df_raw = df_raw.drop(index=dates.iloc[-(i+1)], level="datetime")

# Step 2: Z-score normalization across stocks (per date)
df_raw[("label", "Z_RET")] = (
    df_raw[("label", "FWD_RET")]
    .groupby("datetime")
    .transform(lambda x: (x - x.mean()) / x.std())
)

# handler with learn / infer processors ------------------------
proc_feat = [
    {"class": "DropnaProcessor", "kwargs": {"fields_group": "feature"}},
    # {"class": "CSZScoreNorm",   "kwargs": {"fields_group": "feature"}}, # slows down debugging
]

# proc_feat = [
#     {"class": "CSZScoreNorm",   "kwargs": {"fields_group": "feature"}},
# ]

# proc_feat = [
#     {"class": "Fillna",          # <— correct name
#      "kwargs": {"fields_group": "feature", "fill_value": 0}},  # zero-fill; choose ffill/bfill/etc. if you like
#     {"class": "CSZScoreNorm",
#      "kwargs": {"fields_group": "feature"}},
# ]

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
    "train": (dates.iloc[8],              dates.iloc[int(T*0.8) - 1]),
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
# ------------------------------------------------------------



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


# New metrics
ar = []
ir = []

# Training
######################################################################################
for seed in [i for i in range(100)]: 
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
    print("\n")

    ic.append(metrics['IC'])
    icir.append(metrics['ICIR'])
    ric.append(metrics['RIC'])
    ricir.append(metrics['RICIR'])
    ar.append(metrics['AR'])
    ir.append(metrics['IR'])
######################################################################################





# This reloads the first model and reprints the metrics. Not needed as we already do it above
# # Load and Test
# ######################################################################################
# for seed in [0]:
#     param_path = f'model/{universe}_{seed}.pkl'

#     print(f'Model Loaded from {param_path}')
#     model = MASTERModel(
#             d_feat = d_feat, d_model = d_model, t_nhead = t_nhead, s_nhead = s_nhead, T_dropout_rate=dropout, S_dropout_rate=dropout,
#             beta=beta, gate_input_end_index=gate_input_end_index, gate_input_start_index=gate_input_start_index,
#             n_epochs=n_epoch, lr = lr, GPU = GPU, seed = seed, train_stop_loss_thred = train_stop_loss_thred,
#             save_path='model/', save_prefix=universe
#         )
#     model.load_param(param_path)
#     predictions, metrics = model.predict(dl_test)
#     print(metrics)

#     ic.append(metrics['IC'])
#     icir.append(metrics['ICIR'])
#     ric.append(metrics['RIC'])
#     ricir.append(metrics['RICIR'])
    
# ######################################################################################
from scipy.stats import t

# Sample size
n = len(ic)

# Degrees of freedom
df = n - 1

# t critical value for 95% confidence
t_crit = t.ppf(0.975, df)  # two-tailed

# Function to compute mean and 95% CI error
def ci_95(arr):
    mean = np.mean(arr)
    se = np.std(arr, ddof=1) / np.sqrt(len(arr))
    margin = t_crit * se
    return mean, margin

# Print each metric with 95% CI
print("IC: {:.4f} pm {:.4f}".format(*ci_95(ic)))
print("ICIR: {:.4f} pm {:.4f}".format(*ci_95(icir)))
print("RIC: {:.4f} pm {:.4f}".format(*ci_95(ric)))
print("RICIR: {:.4f} pm {:.4f}".format(*ci_95(ricir)))
print("AR: {:.4f} pm {:.4f}".format(*ci_95(ar)))
print("IR: {:.4f} pm {:.4f}".format(*ci_95(ir)))





# print("IC: {:.4f} pm {:.4f}".format(np.mean(ic), np.std(ic)))
# print("ICIR: {:.4f} pm {:.4f}".format(np.mean(icir), np.std(icir)))
# print("RIC: {:.4f} pm {:.4f}".format(np.mean(ric), np.std(ric)))
# print("RICIR: {:.4f} pm {:.4f}".format(np.mean(ricir), np.std(ricir)))
# print("AR: {:.4f} pm {:.4f}".format(np.mean(ar), np.std(ar)))
# print("IR: {:.4f} pm {:.4f}".format(np.mean(ir), np.std(ir)))