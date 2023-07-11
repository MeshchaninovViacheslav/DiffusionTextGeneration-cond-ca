import torch
import lightning as L
from transformers import AutoConfig, AutoModelForCausalLM
from torch.nn.functional import cross_entropy

from timm.scheduler.cosine_lr import CosineLRScheduler

from typing import Dict, Any
from torch import FloatTensor, Tensor

from lm_training.utils import calc_model_grads_norm


class GPTModel(L.LightningModule):
    def __init__(self, config):
        super(GPTModel, self).__init__()
        # Model Architecture
        self.config = config
        self.gpt_config = AutoConfig.from_pretrained("gpt2")
        self.model = AutoModelForCausalLM.from_config(self.gpt_config)

    def recon_loss(self, inputs, outputs, mask=None):
        if mask is None:
            mask = torch.ones(
                (inputs.shape[0], inputs.shape[1]),
                requires_grad=False,
                dtype=torch.int64,
            )

        losses = cross_entropy(
            input=inputs.reshape(-1, inputs.shape[-1]),
            target=outputs.reshape(-1),
            reduce=False,
        )
        losses = losses * mask.reshape(-1)
        loss = torch.sum(losses) / torch.sum(mask)
        return loss

    def get_loss(self, logits, targets, mask):
        loss = self.recon_loss(logits[:, :-1], targets[:, 1:], mask[:, 1:])
        return loss

    def forward(self, X):
        logits = self.model(**X).logits
        return logits

    def training_step(self, batch):
        target = batch["input_ids"]
        mask = batch["attention_mask"]

        logits = self.forward(batch)
        loss = self.get_loss(logits, target, mask)

        logs = {'loss': loss}
        self.log_dict(logs, is_train=True, sync_dist=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        target = batch["input_ids"]
        mask = batch["attention_mask"]

        logits = self.forward(batch)
        loss = self.get_loss(logits, target, mask)

        logs = {'loss': loss}
        self.log_dict(logs, is_train=False, sync_dist=True)
        return {'loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.optim.lr,
            weight_decay=self.config.optim.weight_decay,
            betas=(self.config.optim.beta_1, self.config.optim.beta_2),
            eps=self.config.optim.eps,
        )
        scheduler = CosineLRScheduler(
            optimizer=optimizer,
            t_initial=self.config.training.training_iters,
            lr_min=self.config.optim.min_lr,
            warmup_lr_init=self.config.optim.warmup_lr,
            warmup_t=self.config.optim.linear_warmup,
            cycle_limit=1,
            t_in_epochs=False,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step_update(num_updates=self.global_step)

    def log_dict(self, losses: Dict[str, Tensor], is_train: bool = True, *args, **kwargs):
        suffix = 'train' if is_train else 'valid'
        losses = {key + f'/{suffix}': value for key, value in losses.items()}
        return super().log_dict(losses, *args, **kwargs)

    def on_before_optimizer_step(self, *args, **kwargs) -> None:
        self.logger.log_metrics({'model/grads_norm': calc_model_grads_norm(self.model)})
        return super().on_before_optimizer_step(*args, **kwargs)

    def generate_text(self, inputs=None, max_new_tokens: int = 64, num_beams: int = 2):
        if inputs is None:
            return self.model.generate(max_new_tokens=max_new_tokens, num_beams=num_beams)
        else:
            return self.model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=num_beams)
