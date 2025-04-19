import torch
import torch.nn as nn
import math

class MLPResidual(nn.Module):
    """
    Simple MLP residual network with one hidden state.
    """
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super(MLPResidual, self).__init__()
        self.lin_emb = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.Dropout(dropout)
        )
        self.lin_res = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x_emb = self.lin_emb(x)
        x_res = self.lin_res(x)
        x_out = self.norm(x_emb+x_res)
        return x_out

class DataEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, n_emb):
        super(DataEmbedding, self).__init__()
        self.feat_embedding = [MLPResidual(in_dim, out_dim)]
        if n_emb > 1:
            for i in range(n_emb-1):
                self.feat_embedding.append(MLPResidual(out_dim, out_dim))
        self.feat_embedding = nn.Sequential(*self.feat_embedding)

    def forward(self, x):
        return self.feat_embedding(x)

class StepEmbedding(nn.Module):
    def __init__(self, hidden_dim, freq_dim=256):
        super(StepEmbedding, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        self.freq_dim = freq_dim

    @staticmethod
    def sinusoidal_embedding(k, freq_dim, max_period=10000):
        half_dim = freq_dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim
        ).to(device=k.device)
        k_freqs = k[:, None].float() * freqs[None]
        k_emb = torch.cat([torch.cos(k_freqs), torch.sin(k_freqs)], dim=-1)
        return k_emb  # (B, freq_dim)

    def forward(self, k):  # (B, )
        k_emb = self.sinusoidal_embedding(k, self.freq_dim)
        k_emb = self.mlp(k_emb)
        return k_emb  # (B, hidden_dim)

class TimeEmbedding(nn.Module):
    def __init__(self, hidden_dim, cont_len, pred_len):
        super(TimeEmbedding, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear((cont_len+pred_len)*4, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )

    def forward(self, x_mark, y_mark):
        """
        x_mark: (B, cont_len, 4)
        y_mark: (B, pred_len, 4)
        """
        time_features = torch.cat([x_mark, y_mark], dim=1).permute(0, 2, 1)  # (B, 4, cont_len+pred_len)
        time_features = time_features.reshape(-1, time_features.shape[1]*time_features.shape[2])  # # (B, 4*(cont_len+pred_len))
        t_emb = self.mlp(time_features)
        return t_emb  # (B, hidden_dim)