import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.distributions import Categorical
import numpy as np
import scipy.stats as stats
import pickle as pkl
import random
import time
import os
from diffusion import DDPM
from network import Denoiser
from utils.visual import *
from utils.metric import *

class DiffMTS:
    def __init__(self, configs, data_loader):
        super(DiffMTS, self).__init__()
        self.use_window_norm = configs.use_window_norm
        self.data_loader = data_loader
        self.n_steps = configs.n_steps
        self.device = configs.device
        self.n_epochs = configs.n_epochs
        self.data_name = configs.data_name
        self.num_feat = configs.num_feat
        self.cont_len = configs.cont_len
        self.pred_len = configs.pred_len
        self.parameterization = configs.parameterization
        self.step_dist = configs.step_dist
        self.use_contrast = configs.use_contrast
        self.n_negatives = configs.n_negatives
        self.contrast_weight = configs.contrast_weight
        self.temperature = configs.temperature
        self.init_lr = configs.init_lr
        self.non_attn = configs.non_attn
        self.task_name = f"{self.data_name}_I{self.cont_len}_O{self.pred_len}_Contrastive{self.contrast_weight}_Step{self.n_steps}_similarity"
        if self.non_attn:
            self.task_name += "_NonAttn"
        print(f"Running task: {self.task_name}")

        self.denoiser = Denoiser(configs).to(self.device)
        self.diffusion = DDPM(self.denoiser, configs).to(self.device)

        self.optimizer = optim.Adam(self.denoiser.parameters(), lr=self.init_lr, weight_decay=1e-6)
        # self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.n_epochs)
        p1 = int(0.75*self.n_epochs)
        p2 = int(0.9*self.n_epochs)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[p1, p2], gamma=0.1)

    def instance_normalization(self, x, y0):
        x_mean = x.mean(dim=1, keepdim=True)  # (B, 1, D)
        x_std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)  # (B, 1, D)
        x_norm = (x - x_mean) / x_std
        y0_norm = (y0 - x_mean) / x_std
        return x_norm, y0_norm, x_mean, x_std

    def instance_denormalization(self, y0, mean, std):
        B = mean.shape[0]
        n_samples = y0.shape[0]//B
        std = torch.repeat_interleave(std, n_samples, dim=0).repeat(1, self.pred_len, 1)  # (B*n_samples, pred_len, D)
        mean = torch.repeat_interleave(mean, n_samples, dim=0).repeat(1, self.pred_len, 1)  # (B*n_samples, pred_len, D)
        y0 = y0 * std + mean
        return y0

    def step_sampling(self, batch_size):
        if self.step_dist == "uniform":
            k_half0 = torch.randint(0, self.n_steps, (batch_size//2,))
            k_half1 = self.n_steps - 1 - k_half0
            k = torch.cat([k_half0, k_half1], dim=0)
        elif self.step_dist == "non-uniform":
            kn = torch.exp(1 - torch.linspace(0, 1, self.n_steps))
            probs = kn / torch.sum(kn)
            k = Categorical(probs).sample(torch.Size((batch_size,)))
        else:
            return NotImplementedError("Such step distribution is not valid!")
        return k.to(self.device)

    def negative_sampling(self, y0, mode):
        B = y0.shape[0]
        y0_len = y0.shape[1]

        # 1.variation-based intra-variate shuffle over patches
        if mode == "variation":
            patch_size = 8
            n_patches = y0_len // 8
            y0_patch = y0.view(B, n_patches, patch_size, self.num_feat)
            neg_samples = torch.zeros((B*self.n_negatives, y0_len, self.num_feat)).to(y0.device)
            for i in range(self.n_negatives):
                idx_patch = torch.randperm(y0_patch.shape[1])
                y0_shuffle = y0_patch[:, idx_patch, :, :].reshape(B, y0_len, self.num_feat)
                neg_indices = torch.arange(0, neg_samples.shape[0], self.n_negatives) + i
                neg_samples[neg_indices] = y0_shuffle

        # 2.PI-based scale change
        elif mode == "scaling":
            # different scaling factors for each variate
            scale_down = np.random.uniform(0.0, 0.5, (self.n_negatives//2, 1, self.num_feat))
            scale_up = np.random.uniform(1.5, 2.0, (self.n_negatives//2, 1, self.num_feat))
            scale = torch.from_numpy(np.concatenate([scale_up, scale_down], axis=0))  # (n_negatives, 1, 1 & num_feat)
            scale_rep = scale.repeat(B, 1, 1).to(y0.device)  # (B*n_negatives, 1, 1 & num_feat)
            y0_rep = torch.repeat_interleave(y0, self.n_negatives, dim=0)  # (B*n_negatives, pred_len, num_feat)
            neg_samples = y0_rep * scale_rep

        # 3.jitter by N(0, 0.3)
        elif mode == "jitter":
            neg_samples = torch.zeros((B*self.n_negatives, y0_len, self.num_feat)).to(y0.device)
            jit_noise = torch.randn_like(neg_samples)*(0.3**0.5)
            y0_rep = torch.repeat_interleave(y0, self.n_negatives, dim=0)  # (B*n_negatives, pred_len, num_feat)
            neg_samples = y0_rep + jit_noise

        # 4.cutout by 10% random zero masking
        elif mode == "cutout":
            mask_len = int(0.1*y0_len)
            end_point = y0_len - mask_len
            start_point = torch.randint(0, end_point, (self.n_negatives,))
            neg_samples = torch.zeros((B*self.n_negatives, y0_len, self.num_feat)).to(y0.device)
            for i in range(self.n_negatives):
                y0_mask = y0.clone()
                y0_mask[:, start_point[i]:start_point[i]+mask_len, :] = 0.0
                neg_indices = torch.arange(0, neg_samples.shape[0], self.n_negatives) + i
                neg_samples[neg_indices] = y0_mask

        else:
            raise NotImplementedError(f"No such negative sampling mode: {mode}!")

        return neg_samples.float()

    def cal_contrastive_loss(self, neg_samples, x, k, noise, loss_type="regression"):
        B = x.shape[0]
        n_negatives = neg_samples.shape[0] // B

        if noise is None:
            neg_noise = torch.randn_like(neg_samples)
        else:
            neg_noise = torch.repeat_interleave(noise, n_negatives, dim=0)  # (B*n_negatives, pred_len, D)
        neg_k = torch.repeat_interleave(k, n_negatives, dim=0)  # (B*n_negatives, )
        neg_x = torch.repeat_interleave(x, n_negatives, dim=0)  # (B*n_negatives, cont_len, D)
        neg_yk = self.diffusion.q_sample(neg_samples, neg_k, neg_noise)
        neg_pred = self.denoiser(neg_x, neg_yk, neg_k)  # (B*n_negatives, pred_len, D)

        if loss_type == "regression":
            neg_loss = cal_mse_loss(neg_pred, neg_noise).view(B, n_negatives)  # (B, n_negatives)
        else:
            neg_pred = neg_pred.reshape(B * n_negatives, -1)
            neg_noise = neg_noise.reshape(B * n_negatives, -1)
            neg_loss = F.cosine_similarity(neg_pred, neg_noise, dim=1).unsqueeze(1).view(B, n_negatives)  # (B, n_negatives)

        return neg_loss

    def cal_train_loss(self, x, y0, x_mark, y0_mark, loss_type="regression"):
        """
        x: (B, context_length, D)
        y0: (B, prediction_length, D)
        """
        B = x.shape[0]
        k = self.step_sampling(B)
        noise = torch.randn_like(y0)

        # positive original denoising loss
        yk = self.diffusion.q_sample(y0, k, noise)
        pred_k = self.denoiser(x, yk, k, x_mark, y0_mark)
        if self.parameterization == "noise":
            target = noise
        elif self.parameterization == "y0":
            target = y0
        else:
            raise NotImplementedError(f"No such parameterization: {self.parameterization}!")
        denoise_loss = cal_mse_loss(pred_k, target)  # (B, )
        if self.use_contrast == "non-contrast":
            infonce_loss = torch.zeros((1, ), device=self.device)
            return torch.mean(denoise_loss), infonce_loss

        # neg_variation_samples = self.negative_sampling(y0, mode="variation")  # (B*n_negatives, pred_len, D)
        # neg_scale_samples = self.negative_sampling(y0, mode="scaling")  # (B*n_negatives, pred_len, D)
        neg_variation_samples = self.negative_sampling(y0, mode="variation")  # (B*n_negatives, pred_len, D)
        neg_scale_samples = self.negative_sampling(y0, mode="scaling")  # (B*n_negatives, pred_len, D)
        neg_variation_loss = self.cal_contrastive_loss(neg_variation_samples, x, k, noise=None, loss_type=loss_type)
        neg_scale_loss = self.cal_contrastive_loss(neg_scale_samples, x, k, noise=None, loss_type=loss_type)

        if loss_type == "regression":
            pos_loss = denoise_loss.view(B, 1)  # (B, 1)
            contrast_loss = -torch.concatenate([pos_loss, neg_variation_loss, neg_scale_loss], dim=1)  # (B, 1+2*n_negatives)
            infonce_loss = -torch.log(torch.softmax(contrast_loss / self.temperature, dim=1)[:, 0])  # (B, )
        else:
            pred_k = pred_k.reshape(B, -1)
            target = target.reshape(B, -1)
            pos_loss = F.cosine_similarity(pred_k, target, dim=1).unsqueeze(1)  # (B, 1)
            contrast_loss = torch.concatenate([pos_loss, neg_variation_loss, neg_scale_loss], dim=1)  # (B, 1+2*n_negatives)
            infonce_loss = -torch.log(torch.softmax(contrast_loss / self.temperature, dim=1)[:, 0])  # (B, )

        return torch.mean(denoise_loss), torch.mean(infonce_loss)

    def train(self, is_refine=False, init_epoch=199, n_epochs=20):
        epoch_loss = {"Denoising": [], "Contrastive": [], "Total": []}
        print(f"Training stage is beginning!")
        if is_refine:
            self.load_pretrained_weights(init_epoch, n_epochs)
            print(f"Fine-tuning stage is beginning")
        for epoch_no in range(self.n_epochs):
            batch_loss = {"Denoising": [], "Contrastive": [], "Total": []}
            start_time = time.time()
            for batch_no, (batch_x, batch_y0, batch_x_mark, batch_y0_mark) in enumerate(self.data_loader):
                self.optimizer.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y0 = batch_y0.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y0_mark = batch_y0_mark.float().to(self.device)
                if self.use_window_norm:
                    batch_x, batch_y0, _, _ = self.instance_normalization(batch_x, batch_y0)
                denoise_loss, contrast_loss = self.cal_train_loss(batch_x, batch_y0, x_mark=None, y0_mark=None, loss_type="similarity")
                total_loss = denoise_loss + self.contrast_weight*contrast_loss
                total_loss.backward()
                batch_loss["Denoising"].append(denoise_loss.item())
                batch_loss["Contrastive"].append(contrast_loss.item())
                batch_loss["Total"].append(total_loss.item())
                self.optimizer.step()
            self.lr_scheduler.step()
            end_time = time.time()
            weight_folder = f"weights/{self.task_name}"
            if not os.path.exists(weight_folder):
                os.mkdir(weight_folder)
            # if (epoch_no >= 175 and (is_refine is False)) or is_refine is True:
            torch.save(self.denoiser.state_dict(), f"{weight_folder}/epoch{epoch_no}.pt")
            epoch_loss["Denoising"].append(np.mean(batch_loss["Denoising"]))
            epoch_loss["Contrastive"].append(np.mean(batch_loss["Contrastive"]))
            epoch_loss["Total"].append(np.mean(batch_loss["Total"]))
            print(f"[epoch {epoch_no}/{self.n_epochs}] Total loss: {epoch_loss['Total'][-1]}, "
                  f"Denoising loss: {epoch_loss['Denoising'][-1]}, Contrastive loss: {epoch_loss['Contrastive'][-1]}, "
                  f"time: {end_time-start_time}s")
        plot_train_loss(epoch_loss, self.task_name)

    def load_pretrained_weights(self, init_epoch, n_epochs):
        # task_name = self.task_name.replace(f"Contrastive{self.contrast_weight}", "Contrastive0.0")
        weight_path = f"weights/{self.task_name}/epoch{init_epoch}.pt"
        weight_dict = torch.load(weight_path, map_location=self.device)
        self.denoiser.load_state_dict(weight_dict)
        self.denoiser.train()
        self.n_epochs = n_epochs
        self.optimizer = optim.Adam(self.denoiser.parameters(), lr=2e-5)  # 1e-4, 2e-5
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=n_epochs)
        print(f"Load pretrained weight: {weight_path}!")

    def load_trained_weights(self, best_epoch):
        weight_path = f"weights/{self.task_name}/epoch{best_epoch}.pt"
        weight_dict = torch.load(weight_path, map_location=self.device)
        self.denoiser.load_state_dict(weight_dict)
        self.denoiser.eval()
        print(f"Load trained weight: {weight_path}!")

    def pred_sampling(self, x, n_samples, x_mark, y0_mark):
        pred_out = self.diffusion.sampling(n_samples, x, x_mark, y0_mark)
        return pred_out

    @torch.no_grad()
    def test(self, best_epoch, n_samples, is_out):
        self.metrics = {}
        plot_data = {}
        print(f"Testing stage on epoch {best_epoch} is beginning!")
        self.load_trained_weights(best_epoch)
        test_no = 0
        for batch_no, (batch_x, batch_y0, batch_x_mark, batch_y0_mark) in enumerate(self.data_loader):
            start_time = time.time()
            batch_x = batch_x.float().to(self.device)
            batch_y0 = batch_y0.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y0_mark = batch_y0_mark.float().to(self.device)
            if self.use_window_norm:
                batch_x, _, x_mean, x_std = self.instance_normalization(batch_x, batch_y0)
            pred_out = self.pred_sampling(batch_x, n_samples, batch_x_mark, batch_y0_mark)  # (B*n_samples, pred_len, D)
            if self.use_window_norm:
                pred_out = self.instance_denormalization(pred_out, x_mean, x_std)  # (B*n_samples, pred_len, D)
            B = batch_x.shape[0]
            pred_out = pred_out.view(-1, n_samples, self.pred_len, self.num_feat)
            for i in range(B):
                out_data = {"Predictions": pred_out[i].detach().cpu().numpy(),
                            "Ground truth": batch_y0[i].detach().cpu().numpy(),
                            "Conditions": batch_x[i].detach().cpu().numpy()}
                self.cal_metrics(out_data, test_no)
                if is_out:
                    plot_data[test_no] = out_data
                print(f"{test_no} prediction done!")
                test_no += 1
            end_time = time.time()
            print(f"Batch_{batch_no}: {end_time-start_time}s")
        mse_avr = np.mean([v["mse"] for v in self.metrics.values()])
        mae_avr = np.mean([v["mae"] for v in self.metrics.values()])
        crps_avr = np.mean([v["crps"] for v in self.metrics.values()])
        crps_sum_avr = np.mean([v["crps_sum"] for v in self.metrics.values()])
        print(f"[Final errors] mse: {mse_avr}, mae: {mae_avr}, crps: {crps_avr}, crps_sum: {crps_sum_avr}")
        with open(f"output/{self.task_name}_metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=4)
        if is_out:
            with open(f"output/{self.task_name}_plot.pkl", "wb") as f:
                pkl.dump(plot_data, f)

    def cal_metrics(self, test_data, test_no):
        pred = test_data["Predictions"]  # (n_samples, pred_len, num_feat)
        y0 = test_data["Ground truth"]  # (pred_len, num_feat)
        mse = cal_mse_norm(pred, y0)
        mae = cal_mae_norm(pred, y0)
        crps = cal_crps_norm(pred, y0)
        crps_sum = cal_crps_sum(pred, y0)
        self.metrics[f"{test_no}"] = {"mse": float(mse), "mae": float(mae), "crps": float(crps), "crps_sum": float(crps_sum)}
        print(f"[{test_no}]: ", self.metrics[f"{test_no}"])