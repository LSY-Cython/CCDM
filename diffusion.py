import torch
from torch import nn

def gather(a, k):
    B = k.shape[0]
    out = a.to(k.device).gather(-1, k)
    return out.reshape(B, 1, 1).to(k.device)

def cosine_beta_schedule(n_steps, s=0.008):
    x = torch.linspace(0, n_steps, n_steps+1)
    alphas_bar = torch.cos(((x / n_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return torch.clip(betas, 0, 0.999)

class DDPM(nn.Module):
    def __init__(self, denoiser, configs):
        super().__init__()
        if configs.beta_schedule == "linear":
            self.betas = torch.linspace(configs.beta_start, configs.beta_end, configs.n_steps)
        elif configs.beta_schedule == "quad":
            self.betas = torch.linspace(configs.beta_start**0.5, configs.beta_end**0.5, configs.n_steps)**2
        elif configs.beta_schedule == "cosine":
            self.betas = cosine_beta_schedule(configs.n_steps)
        else:
            raise NotImplementedError(configs.beta_schedule)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.alphas_bar_prev = torch.concatenate([torch.ones((1, )), self.alphas_bar[0:-1]], dim=0)

        self.denoiser = denoiser
        self.n_steps = configs.n_steps
        self.pred_len = configs.pred_len
        self.parameterization = configs.parameterization

    # q(y_k|y_0)
    def q_sample(self, y0, k, noise):
        alphas_bar_k = gather(self.alphas_bar, k)
        mean = (alphas_bar_k**0.5)*y0
        var = 1. - alphas_bar_k
        return mean + (var**0.5)*noise

    # q(y_{k-1}|y_k, y_0)
    def q_posterior(self, yk, pred_k, k):
        if self.parameterization == "noise":
            pred_k_coef = gather(self.betas, k) / (1.-gather(self.alphas_bar, k))**0.5
            mean = (yk - pred_k_coef*pred_k) / gather(self.alphas, k)**0.5
        elif self.parameterization == "y0":
            pred_k_coef = (gather(self.alphas_bar_prev, k)**0.5)*gather(self.betas, k)
            yk_coef = (gather(self.alphas, k)**0.5)*(1.-gather(self.alphas_bar_prev, k))
            mean = (pred_k_coef*pred_k + yk_coef*yk) / (1.-gather(self.alphas_bar, k))
        else:
            raise NotImplementedError(f"No such parameterization: {self.parameterization}")
        variance = (1.-gather(self.alphas_bar_prev, k))*gather(self.betas, k) / (1.-gather(self.alphas_bar, k))
        return mean, variance

    # p(y_{k-1}|y_k)
    @torch.no_grad()
    def p_sample(self, x, yk, k, x_mark, y0_mark):
        pred_k = self.denoiser(x, yk, k, x_mark, y0_mark)
        post_mean, post_var = self.q_posterior(yk, pred_k, k)
        z = torch.randn(yk.shape, device=yk.device)
        return post_mean + (post_var**0.5)*z

    @torch.no_grad()
    def sampling(self, n_samples, x, x_mark, y0_mark):
        B = x.shape[0]
        D = x.shape[2]
        sample_shape = (B*n_samples, self.pred_len, D)
        yk = torch.randn(sample_shape, device=x.device)
        x = torch.repeat_interleave(x, n_samples, dim=0)  # (B*n_samples, cont_len, D)
        x_mark = torch.repeat_interleave(x_mark, n_samples, dim=0)  # (B*n_samples, cont_len, 4)
        y0_mark = torch.repeat_interleave(y0_mark, n_samples, dim=0)  # (B*n_samples, pred_len, 4)
        # x_emb = torch.repeat_interleave(x_emb, n_samples, dim=0)  # (B*n_samples, D, d_model)
        for j in reversed(range(0, self.n_steps, 1)):
            k = torch.ones(B*n_samples, dtype=torch.long, device=x.device)*j
            yk = self.p_sample(x, yk, k, x_mark, y0_mark)
        return yk  # (B*n_samples, pred_len, D)
