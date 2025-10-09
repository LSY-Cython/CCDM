import numpy as np
from einops import rearrange
import properscoring as ps
import torch
import torch.nn as nn

def cal_mse_loss(a, b):
    B = a.shape[0]
    loss_func = nn.MSELoss(reduction='none').to(a.device)
    inst_mse = loss_func(a, b)
    batch_mse = torch.mean(inst_mse.contiguous().view(B, -1), dim=1)  # (B, )
    return batch_mse

def cal_mse_norm(preds, truth):
    median = np.quantile(preds, q=0.5, axis=0)  # (pred_len, D)
    mse = np.mean(np.square(median-truth))
    return mse

def cal_mae_norm(preds, truth):
    median = np.quantile(preds, q=0.5, axis=0)  # (pred_len, D)
    mse = np.mean(np.abs(median-truth))
    return mse

def cal_crps_norm(preds, truth):
    pred_len = truth.shape[0]
    D = truth.shape[1]
    truth = rearrange(truth, "L D -> (L D)")
    preds = rearrange(preds, "N L D -> (L D) N")
    crps_per_point = ps.crps_ensemble(truth, preds)
    crps = np.sum(crps_per_point)/(pred_len*D)
    return crps

def cal_crps_sum(preds, truth):
    # sum over the feature dimension to measure the joint effect
    truth_sum = np.sum(truth, axis=1)  # (pred_len, ), guess not sum but mean over D
    preds_sum = np.sum(preds, axis=2).T  # (pred_len, n_samples)
    crps_per_step = ps.crps_ensemble(truth_sum, preds_sum)  # (pred_len, )
    crps_sum = np.mean(crps_per_step)
    return crps_sum