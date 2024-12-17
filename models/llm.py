import torch
from torch import nn
from .building_blocks import TransformerEncoder


class LLM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.transformer = TransformerEncoder(
            args = args,
            dmodel = args.dmodel,
            depth = args.depth,
            nheads = args.nheads,
            dropout = args.dropout,
        )

    def forward(self, x):
        input_ids = x["input_ids"]
        padding_mask = x['attention_mask']

        # TODO: add causal mask and stuff

        return self.transformer(x)