import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, eps = 1e-5):
        super().__init__()
        self.eps = eps

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output

def init_method_normal(sigma):
    def init_(tensor):
        return nn.init.normal_(tensor, mean=0.0, std=sigma)
    return init_

def replace_layernorm_inplace(module, name):
     for attr_str in dir(module):
         target_attr = getattr(module, attr_str)
         if type(target_attr) == torch.nn.LayerNorm:
             new_bn = RMSNorm(eps = 1e-5)
             setattr(module, attr_str, new_bn)
     for name, immediate_child_module in module.named_children():
         replace_layernorm_inplace(immediate_child_module, name)