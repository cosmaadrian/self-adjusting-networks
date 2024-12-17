from lib.trainer_extra import AcumenTrainer
import torch

from torch.optim import AdamW


class LMTrainer(AcumenTrainer):
    def __init__(self, args, model):
        super().__init__(args, model)
        self.next_token_prediction_loss = torch.nn.CrossEntropyLoss()

        self.iter_idx = 0

    def configure_optimizers(self, lr = 0.1):
        if self._optimizer is not None:
            return self._optimizer

        impl = AdamW
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': self.args.optimizer_args.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        print("::::::::::: Using MuAdam :::::::::::")
        self._optimizer = AdamW(
            params = optim_groups,
            lr = lr,
            betas = [self.args.optimizer_args.beta1, self.args.optimizer_args.beta2],
            weight_decay = self.args.optimizer_args.weight_decay,
            eps = float(self.args.optimizer_args.eps),
            fused = True,
        )

        return self._optimizer

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]

        model_output = self.model(batch)
        output = model_output["output"]

        next_token_loss = self.next_token_prediction_loss(output.view(-1, output.size(-1)), input_ids.view(-1))
        final_loss = next_token_loss

        self.log('train/loss:next_byte', next_token_loss.item(), on_step = True)
        self.iter_idx += 1

        return final_loss