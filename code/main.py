from master_bert import MASTERModel
import pickle
import numpy as np
import time

from utils import load_all_csv_data_with_market_indexes, load_all_csv_data_without_index, StockTensorDataset

# Please install qlib first before load the data.

# # Load data without market indexes
# stock_data, stock_names, features_names = load_all_csv_data_without_index()

# # Load data with market indexes
# # stock_data, stock_names, features_names = load_all_csv_data_with_market_indexes()

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



# New version
# Load data with market indexes
data_tensor, stock_names, feature_names = load_all_csv_data_with_market_indexes()

print("Data shape:", data_tensor.shape)

import pandas as pd
# Get dates from one of the CSVs (assuming all have the same dates)
sample_csv = 'data/enriched/market_indexes_aggregated.csv'
dates = pd.read_csv(sample_csv)['Date'].tolist()

N = data_tensor.shape[0]
train_end = int(N * 0.8)
val_end = int(N * 0.9)

# Split the underlying tensor and stock_names, but keep all dates
dl_train = StockTensorDataset(data_tensor[:train_end], stock_names[:train_end], dates)
dl_valid = StockTensorDataset(data_tensor[train_end:val_end], stock_names[train_end:val_end], dates)
dl_test = StockTensorDataset(data_tensor[val_end:], stock_names[val_end:], dates)

print("Train size:", len(dl_train))
print("Val size:", len(dl_valid))
print("Test size:", len(dl_test))




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
d_model = 276
t_nhead = 4
s_nhead = 2
dropout = 0.5
gate_input_start_index = 158
gate_input_end_index = 221

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
for seed in [0, 1, 2, 3, 4]:
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