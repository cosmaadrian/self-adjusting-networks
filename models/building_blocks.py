import math
import torch
from torch import nn, einsum
import torch.nn.functional as F

from torch.nn.init import constant_
from einops import rearrange, repeat

from .utils import RMSNorm
from .rope import RotaryEmbedding
from .utils import init_method_normal

def default(val, default_val):
    return val if val is not None else default_val

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d j -> ... (d j)')

def apply_rotory_pos_emb(q, k, sinu_pos):
    sinu_pos = rearrange(sinu_pos, '() n (j d) -> n j d', j = 2)
    sin, cos = sinu_pos.unbind(dim = -2)
    sin, cos = map(lambda t: repeat(t, 'b n -> b (n j)', j = 2), (sin, cos))
    q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
    return q, k

# helper classes

class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(0, max_seq_len, dtype=torch.float)
        sinusoid_inp = torch.einsum("i,j->ij", position, inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.register_buffer('emb', emb)

    def forward(self, x):
        return self.emb[None, :x.shape[1], :].to(x)

class Attention(nn.Module):
    def __init__(self, args, dim, rotary_emb, nheads,
            qkv_bias = False,
            qk_norm = False,
            attn_drop = 0.,
            proj_drop = 0.,
        ):

        super().__init__()
        self.args = args

        self.nheads = nheads
        self.head_dim = dim // nheads
        self.qkv_bias = qkv_bias

        assert dim % nheads == 0, 'dim should be divisible by nheads'

        self.scale = 1 / self.head_dim

        self.to_q = nn.Linear(dim, dim, bias = qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias = qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias = qkv_bias)

        self.q_norm = RMSNorm(eps = 1e-5) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(eps = 1e-5) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim, bias = True)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rotary_emb = rotary_emb

        # muP fan-in style initialization
        self.init_method = init_method_normal((1 / dim) ** 0.5)
        self.reset_parameters()

    def reset_parameters(self):
        self.init_method(self.to_q.weight)
        self.init_method(self.to_k.weight)
        self.init_method(self.to_v.weight)

        self.init_method(self.proj.weight)

        if self.qkv_bias:
            constant_(self.to_q.bias, 0.)
            constant_(self.to_k.bias, 0.)
            constant_(self.to_v.bias, 0.)

    def forward(self, x, mask = None):
        h = self.nheads

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.rotary_emb is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if mask is not None:
            sim = sim.reshape((sim.shape[0] // h, h, sim.shape[1], sim.shape[2])) # (b h) n n -> b h n n
            # # mask is (b, n, n)
            mask = mask.unsqueeze(1) # (b, 1, n, n)
            mask = mask.expand(-1, h, -1, -1) # (b, h, n, n)

            sim = sim + mask # mask is float
            sim = sim.reshape((sim.shape[0] * h, sim.shape[2], sim.shape[3])) # b h n n -> (b h) n n

        attn = sim.softmax(dim = -1)
        attn = torch.nan_to_num(attn)

        attn = self.attn_drop(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)

        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class MLP(nn.Module):
    def __init__(self, args, in_features, hidden_features = None, out_features = None, bias = True, drop = 0.):
        super().__init__()
        self.args = args
        self.drop = drop
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
        self.dropout = nn.Dropout(drop)

        self.init_method = init_method_normal((1 / in_features) ** 0.5)
        self.init_method_w3 = init_method_normal((1 / hidden_features) ** 0.5)
        self.reset_parameters()

    def reset_parameters(self):
        self.init_method(self.w1.weight)
        self.init_method_w3(self.w3.weight)

        if self.w1.bias is not None:
            constant_(self.w1.bias, 0.)

        if self.w3.bias is not None:
            constant_(self.w3.bias, 0.)

    def forward(self, x, **kwargs):
        x =  F.gelu(self.w1(x))
        x = self.dropout(x)
        w3 = self.w3(x)
        return w3


class TransformerEncoder(nn.Module):
    def __init__(self,
                args,
                dmodel,
                depth,
                nheads,
                dropout = 0.,
            ):

        super().__init__()
        self.args = args

        self.norm = RMSNorm(eps = 1e-5)
        self.layers = nn.ModuleList([])

        dim_head = dmodel // nheads
        self.rotary_emb = RotaryEmbedding(dim_head // 2)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(
                    args = self.args,
                    dim = dmodel,
                    rotary_emb = self.rotary_emb,
                    nheads = nheads,
                    qkv_bias = False,
                    qk_norm = True,
                ),
                MLP(
                    args,
                    in_features = dmodel,
                    hidden_features = 4 * dmodel,
                    out_features = dmodel,
                    drop = dropout,
                    bias = True
                )
            ]))

    def forward(self, x, mask = None):
        for _, (attn, ff) in enumerate(self.layers):
            first_x = x
            x = self.norm(x)
            x = attn(x, mask = mask)
            x = first_x + x
            x = self.norm(x)
            x = ff(x)
            x = first_x + x
        return x

