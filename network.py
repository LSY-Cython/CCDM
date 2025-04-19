import torch
import torch.nn as nn
from embed import *
from attention import *

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_dim, d_model, n_heads, attn_dropout, mlp_ratio=1.0, non_attn=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        if non_attn:
            self.attn = nn.Linear(d_model, d_model, bias=True)
        else:
            self.attn = FullAttention(d_model=d_model, n_heads=n_heads, attn_dropout=attn_dropout)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(d_model*mlp_ratio)
        self.mlp = AttnMLP(in_dim=d_model, hidden_dim=mlp_hidden_dim, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6*d_model, bias=True)
        )
        self.non_attn = non_attn

    def forward(self, x, c):
        """
        x: (B, num_feat, d_model), d_model=hidden_dim*2
        c: (B, hidden_dim)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x_mod = modulate(self.norm1(x), shift_msa, scale_msa)
        # if self.non_attn:
        #     x = x + gate_msa.unsqueeze(1) * self.attn(x_mod)
        # else:
        x = x + gate_msa.unsqueeze(1) * self.attn(x_mod, x_mod, x_mod)
        x_mod = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_mod)
        return x  # (B, num_feat, d_model)

class Decoder(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_dim, d_model, pred_len, n_emb):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            DataEmbedding(d_model, d_model, n_emb-1),
            nn.Linear(d_model, pred_len)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2*d_model, bias=True)
        )

    def forward(self, x, k):
        """
        x: (B, num_feat, d_model)
        k: (B, hidden_dim)
        """
        shift, scale = self.adaLN_modulation(k).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        x = self.mlp(x)
        return x

class Denoiser(nn.Module):
    def __init__(self, configs):
        super(Denoiser, self).__init__()
        self.x_embedder = DataEmbedding(configs.cont_len, configs.cont_hidden_dim, configs.n_emb)
        self.y_embedder = DataEmbedding(configs.pred_len, configs.pred_hidden_dim, configs.n_emb)
        self.k_embedder = StepEmbedding(configs.step_hidden_dim, freq_dim=256)
        self.t_embedder = TimeEmbedding(configs.time_hidden_dim, configs.cont_len, configs.pred_len)
        
        d_model = configs.cont_hidden_dim + configs.pred_hidden_dim
        self.blocks = nn.ModuleList([
            DiTBlock(configs.step_hidden_dim, d_model, configs.n_heads, configs.attn_dropout, configs.mlp_ratio, configs.non_attn)
            for _ in range(configs.n_depth)])
        self.decoder = Decoder(configs.step_hidden_dim, d_model, configs.pred_len, configs.n_emb)
        # self.initialize_weights()

    def initialize_weights(self):
        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.decoder.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.decoder.adaLN_modulation[-1].bias, 0)

    def forward(self, x, y, k, x_mark=None, y_mark=None):
        """
        x: (B, context_length, num_feat)
        y: (B, prediction_length, num_feat)
        k: (B, )
        """
        x = self.x_embedder(x.permute(0, 2, 1))  # (B, num_feat, hidden_dim)
        y = self.y_embedder(y.permute(0, 2, 1))  # (B, num_feat, hidden_dim)
        k = self.k_embedder(k)  # (B, hidden_dim)

        # if x_mark is not None and y_mark is not None:
        #     t = self.t_embedder(x_mark, y_mark)  # (B, hidden_dim)
        #     c = k + t  # (B, hidden_dim)
        # else:
        #     c = k
        c = k

        h = torch.cat([x, y], dim=-1)  # (B, num_feat, d_model)
        for block in self.blocks:
            h = block(h, c)  # (B, num_feat, d_model)
        out = self.decoder(h, c).permute(0, 2, 1)  # (B, pred_len, num_feat)

        return out




