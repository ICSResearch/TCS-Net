import math
import torch
from torch import nn
from einops.layers.torch import Rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class ScaledDotProduct(nn.Module):
    def __init__(self):
        super(ScaledDotProduct, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, scale=None):
        attention = torch.bmm(q, k.transpose(-2, -1))
        if scale:
            attention = attention * scale
        attention = self.softmax(attention)
        context = torch.bmm(attention, v)
        return context, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=16 ** 2, num_heads=8, dropout=0., out_dim=None):
        super(MultiHeadAttention, self).__init__()
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, model_dim)
        self.linear_v = nn.Linear(model_dim, model_dim)
        self.linear_q = nn.Linear(model_dim, model_dim)

        self.dot_product_attention = ScaledDotProduct()
        self.linear_out = nn.Linear(model_dim, out_dim if out_dim is not None else model_dim)
        self.dropout = nn.Dropout(dropout)

    def _reshape_to_heads(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.num_heads
        return x.reshape(batch_size, seq_len, self.num_heads, sub_dim) \
            .permute(0, 2, 1, 3) \
            .reshape(batch_size * self.num_heads, seq_len, sub_dim)

    def _reshape_from_heads(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.num_heads
        out_dim = in_feature * self.num_heads
        return x.reshape(batch_size, self.num_heads, seq_len, in_feature) \
            .permute(0, 2, 1, 3) \
            .reshape(batch_size, seq_len, out_dim)

    def forward(self, x):
        key = self._reshape_to_heads(self.linear_k(x))
        value = self._reshape_to_heads(self.linear_v(x))
        query = self._reshape_to_heads(self.linear_q(x))

        scale = key.size(-1) ** -0.5
        context, attention = self.dot_product_attention(query, key, value, scale)

        context = self._reshape_from_heads(context)

        output = self.linear_out(context)
        output = self.dropout(output)
        return output, attention


class OneUnit(nn.Module):
    def __init__(self, dim=16 ** 2, heads=8, dropout=0., out_dim=None):
        super().__init__()
        self.dim = dim
        self.flag = out_dim
        # self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout)
        self.attn = MultiHeadAttention(model_dim=dim, num_heads=heads, dropout=dropout, out_dim=out_dim)
        self.ffn = FeedForward(dim=dim if out_dim is None else out_dim, hidden_dim=int(dim * 4), dropout=dropout)
        self.norm1 = nn.LayerNorm(dim if out_dim is None else out_dim)
        self.norm2 = nn.LayerNorm(dim if out_dim is None else out_dim)

    def forward(self, x):
        context, attn = self.attn(x)
        x = self.norm1(context + x) if self.flag is None else self.norm1(context)
        ffn = self.ffn(x)
        x = self.norm2(ffn + x)
        return x, attn


class Units(nn.Module):
    def __init__(self, dim=16, depth=1, heads=8, dropout=0., out_dim=None):
        super().__init__()
        self.patch_size = dim
        self.dim = self.patch_size ** 2
        self.depth = depth
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(OneUnit(dim=self.dim, heads=heads, dropout=dropout, out_dim=out_dim))

    def forward(self, x):
        x = Rearrange('b l h w -> b l (h w)')(x)
        x = x * math.sqrt(self.dim)
        attn_maps = []
        for i in range(self.depth):
            x, attn = self.layers[i](x)
            attn_maps.append(attn)
        x = Rearrange('b l (h w)-> b l h w', h=self.patch_size, w=self.patch_size)(x)
        return x, attn_maps
