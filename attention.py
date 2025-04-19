import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from functools import partial
import numpy as np

class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        return context_in

    def forward(self, queries, keys, values):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context = self._update_context(context, values, scores_top, index, L_Q)

        return context.contiguous()

class iInformer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None):
        super(iInformer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = ProbAttention(False, factor=1)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(queries, keys, values)
        out = out.view(B, L, -1)

        return self.out_projection(out)

class FullAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout, d_key=None, d_value=None):
        super(FullAttention, self).__init__()
        d_key = d_key or (d_model // n_heads)
        d_value = d_value or (d_model // n_heads)
        self.n_heads = n_heads

        self.WQ = nn.Linear(d_model, d_key*n_heads, bias=False)
        self.WK = nn.Linear(d_model, d_key*n_heads, bias=False)
        self.WV = nn.Linear(d_model, d_value*n_heads, bias=False)
        self.WO = nn.Linear(d_value*n_heads, d_model)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, query, key, value):
        B, l_query, d_query = query.shape  # d_query=d_key
        _, l_key, _ = key.shape  # l_value=l_key

        # Matrices projection
        Q = self.WQ(query).view(B, l_query, self.n_heads, -1)
        K = self.WK(key).view(B, l_key, self.n_heads, -1)
        V = self.WV(value).view(B, l_key, self.n_heads, -1)

        # Scaled dot-product attention
        scale = 1./sqrt(Q.shape[-1])
        scores = torch.einsum("blhe,bshe->bhls", Q, K)
        A = self.dropout(torch.softmax(scale*scores, dim=-1))
        agg = torch.einsum("bhls,bshd->blhd", A, V).contiguous()
        O = self.WO(agg.view(B, l_query, -1))
        return O  # (B, L, d_model)

class AttnMLP(nn.Module):
    def __init__(
            self,
            in_dim,
            hidden_dim=None,
            out_dim=None,
            norm_layer=None,
            bias=True,
            drop=0.
    ):
        super(AttnMLP, self).__init__()
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim

        self.fc1 = nn.Linear(in_dim, hidden_dim, bias)
        self.act = nn.GELU(approximate="tanh")
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_dim) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x