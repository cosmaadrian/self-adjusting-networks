import torch
from torch import nn
from .building_blocks import TransformerEncoder


# this is a char llm
class LLM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.transformer = TransformerEncoder(
            args = args,
            dmodel = args.model_args.dmodel,
            depth = args.model_args.num_layers,
            nheads = 8,
            dropout = args.model_args.dropout,
        )

        self.char_embedding = nn.Embedding(384, int(self.args.model_args.dmodel * self.args.model_width_multiplier))

    def _shift_right(self, input_ids):
        decoder_start_token_id = 0
        pad_token_id = decoder_start_token_id

        if decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. "
                "See T5 docs for more information."
            )

        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")

        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        return shifted_input_ids

    def forward(self, x):
        input_ids = x["input_ids"]
        padding_mask = x['attention_mask']

        decoder_input_ids = self._shift_right(input_ids)
        embedded_chars = self.char_embedding(decoder_input_ids)

        x = self.transformer(
            x = embedded_chars,
        )

        with torch.no_grad():
            entropies = {}
            # get all mlp layers of transformer
            for i, layer in enumerate(self.transformer.layers):
                mlp_weight = layer[1].fc1.fc.weight
                S = torch.linalg.svdvals(mlp_weight)

                # normalize S
                normalized_S = S / S.sum()
                spectral_entropy = -torch.sum(normalized_S * torch.log(normalized_S + 1e-6))
                entropies[f'entropy/mlp_layer_{i}'] = spectral_entropy.item()

        return {
            'output': x,
            'entropies': entropies,
        }