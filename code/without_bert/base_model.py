import numpy as np
import pandas as pd
import copy

from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import torch
import torch.optim as optim

def calc_ic(pred, label):
    df = pd.DataFrame({'pred':pred, 'label':label})
    ic = df['pred'].corr(df['label'])
    ric = df['pred'].corr(df['label'], method='spearman')
    return ic, ric

def zscore(x):
    return (x - x.mean()).div(x.std() + 1e-8)  # Avoid division by zero

def drop_extreme(x):
    sorted_tensor, indices = x.sort()
    N = x.shape[0]
    percent_2_5 = int(0.025*N)  
    # Exclude top 2.5% and bottom 2.5% values
    filtered_indices = indices[percent_2_5:-percent_2_5]
    mask = torch.zeros_like(x, device=x.device, dtype=torch.bool)
    mask[filtered_indices] = True
    return mask, x[mask]

def drop_na(x):
    N = x.shape[0]
    mask = ~x.isnan()
    return mask, x[mask]

class DailyBatchSamplerRandom(Sampler):
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        # calculate number of samples in each batch
        self.daily_count = pd.Series(index=self.data_source.get_index()).groupby("datetime").size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)  # calculate begin index of each batch
        self.daily_index[0] = 0

    def __iter__(self):
        if self.shuffle:
            index = np.arange(len(self.daily_count))
            np.random.shuffle(index)
            for i in index:
                yield np.arange(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
        else:
            for idx, count in zip(self.daily_index, self.daily_count):
                yield np.arange(idx, idx + count)

    def __len__(self):
        return len(self.data_source)


class SequenceModel():
    def __init__(self, n_epochs, lr, GPU=None, seed=None, train_stop_loss_thred=None, save_path = 'model/', save_prefix= ''):
        self.n_epochs = n_epochs
        self.lr = lr
        self.device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.train_stop_loss_thred = train_stop_loss_thred

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
        self.fitted = -1

        self.model = None
        self.train_optimizer = None

        self.save_path = save_path
        self.save_prefix = save_prefix


    def init_model(self):
        if self.model is None:
            raise ValueError("model has not been initialized")

        self.train_optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.model.to(self.device)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        loss = (pred[mask]-label[mask])**2
        return torch.mean(loss)

    def train_epoch(self, data_loader):
        self.model.train()
        losses = []

        i=0
        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            '''
            data.shape: (N, T, F)
            N - number of stocks
            T - length of lookback_window, 8
            F - 158 factors + 63 market information + 1 label           
            '''
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            
            # Additional process on labels
            # If you use original data to train, you won't need the following lines because we already drop extreme when we dumped the data.
            # If you use the opensource data to train, use the following lines to drop extreme labels.
            #########################
            mask, label = drop_extreme(label)
            feature = feature[mask, :, :]
            label = zscore(label) # CSZscoreNorm
            #########################

            pred = self.model(feature.float())
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())

            self.train_optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.train_optimizer.step()

            # if i == 1:
                # Print feature, label, pred, loss then exit
                # print(f"Feature shape: {feature.shape}")
                # print(f"Label shape: {label.shape}")
                # print(f"Pred shape: {pred.shape}")
                # print(f"Loss shape: {loss.shape}")
                # print(f"Feature: {feature}")
                # print(f"Label: {label}")
                # print(f"Pred: {pred}")
                # print(f"Loss: {loss}")
                # exit(1)
            # print(f"Loss: {loss}")
            i += 1

        return float(np.mean(losses))

    def test_epoch(self, data_loader):
        self.model.eval()
        losses = []

        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            # You cannot drop extreme labels for test. 
            label = zscore(label)
                        
            pred = self.model(feature.float())
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())

        return float(np.mean(losses))

    def _init_data_loader(self, data, shuffle=True, drop_last=True):
        sampler = DailyBatchSamplerRandom(data, shuffle)
        data_loader = DataLoader(data, sampler=sampler, drop_last=drop_last)
        return data_loader

    def load_param(self, param_path):
        self.model.load_state_dict(torch.load(param_path, map_location=self.device))
        self.fitted = 1

    def fit(self, dl_train, dl_valid=None):
        train_loader = self._init_data_loader(dl_train, shuffle=True, drop_last=True)
        best_param = None
        saved = False
        for step in range(self.n_epochs):
            train_loss = self.train_epoch(train_loader)
            self.fitted = step
            if dl_valid:
                predictions, metrics = self.predict(dl_valid)
                print("Epoch %d, train_loss %.6f, valid ic %.4f, icir %.3f, rankic %.4f, rankicir %.3f." % (step, train_loss, metrics['IC'],  metrics['ICIR'],  metrics['RIC'],  metrics['RICIR']))
            else: print("Epoch %d, train_loss %.6f" % (step, train_loss))
        
            if train_loss <= self.train_stop_loss_thred:
                best_param = copy.deepcopy(self.model.state_dict())
                torch.save(best_param, f'{self.save_path}/{self.save_prefix}_{self.seed}.pkl')
                saved = True
                break
        if not saved:
            torch.save(self.model.state_dict(), f'{self.save_path}/{self.save_prefix}_{self.seed}.pkl')
        

    # Old, this does not have AR and IR
    # def predict(self, dl_test):
    #     if self.fitted<0:
    #         raise ValueError("model is not fitted yet!")
    #     else:
    #         print('Epoch:', self.fitted)

    #     test_loader = self._init_data_loader(dl_test, shuffle=False, drop_last=False)

    #     preds = []
    #     ic = []
    #     ric = []

    #     self.model.eval()
    #     for data in test_loader:
    #         data = torch.squeeze(data, dim=0)
    #         feature = data[:, :, 0:-1].to(self.device)
    #         label = data[:, -1, -1]
            
    #         # nan label will be automatically ignored when compute metrics.
    #         # zscorenorm will not affect the results of ranking-based metrics.

    #         with torch.no_grad():
    #             pred = self.model(feature.float()).detach().cpu().numpy()
    #         preds.append(pred.ravel())

    #         daily_ic, daily_ric = calc_ic(pred, label.detach().numpy())
    #         ic.append(daily_ic)
    #         ric.append(daily_ric)

    #     predictions = pd.Series(np.concatenate(preds), index=dl_test.get_index())





    #     metrics = {
    #         'IC': np.mean(ic),
    #         'ICIR': np.mean(ic)/np.std(ic),
    #         'RIC': np.mean(ric),
    #         'RICIR': np.mean(ric)/np.std(ric)
    #     }

    #     return predictions, metrics



    def predict(self, dl_test):
        if self.fitted < 0:
            raise ValueError("model is not fitted yet!")
        else:
            print("Epoch:", self.fitted)

        test_loader = self._init_data_loader(
            dl_test, shuffle=False, drop_last=False
        )

        preds, labels = [], []       # collect predictions AND raw labels
        ic, ric = [], []

        self.model.eval()
        for data in test_loader:
            data = torch.squeeze(data, dim=0)

            feature = data[:, :, 0:-1].to(self.device)
            label   = data[:, -1, -1]          # raw forward return

            with torch.no_grad():
                pred = self.model(feature.float()).cpu().numpy()

            preds.append(pred.ravel())
            labels.append(label.cpu().numpy().ravel())

            daily_ic, daily_ric = calc_ic(pred, label.numpy())
            ic.append(daily_ic)
            ric.append(daily_ric)

        # ------------------------------------------------------------------
        # Assuming predictions and labels have the same multi-index: (datetime, instrument)
        idx = dl_test.get_index()
        predictions = pd.Series(np.concatenate(preds), index=idx)
        label_series = pd.Series(np.concatenate(labels), index=idx)

        daily_port_ret = []
        daily_bench_ret = []

        # Loop over each datetime
        for dt, pred_slice in predictions.groupby(level="datetime"):
            label_slice = label_series.loc[dt]
            
            # Portfolio return: top-30 by prediction
            top30_idx = pred_slice.nlargest(30).index
            port_ret = label_series.loc[top30_idx].mean()
            daily_port_ret.append(port_ret)

            # Benchmark: equal-weight return across all instruments
            bench_ret = label_slice.mean()
            daily_bench_ret.append(bench_ret)

        # Convert to numpy arrays
        daily_port_ret = np.array(daily_port_ret)
        daily_bench_ret = np.array(daily_bench_ret)

        # AR and tracking error
        active_ret = daily_port_ret - daily_bench_ret

        # AR and IR
        AR = active_ret.mean()
        tracking_error = active_ret.std()
        IR = (active_ret.mean() / (tracking_error + 1e-12))
        # ------------------------------------------

        metrics = {
            "IC":     np.mean(ic),
            "ICIR":   np.mean(ic)  / np.std(ic),
            "RIC":    np.mean(ric),
            "RICIR":  np.mean(ric) / np.std(ric),
            "AR":     AR,
            "IR":     IR,
        }

        return predictions, metrics
