import torch
import lightning as L
from transformers import AutoConfig, BertForMaskedLM
from torch.nn.functional import cross_entropy

from timm.scheduler.cosine_lr import CosineLRScheduler

from typing import Dict, Any
from torch import FloatTensor, Tensor

from lm_training.util import calc_model_grads_norm

import torch.distributed as dist


def update_bert_config(bert_config, config):
    for key in config.bert_config:
        bert_config.__dict__[key] = config.bert_config[key]
    return bert_config


class BERTModel(L.LightningModule):
    def __init__(self, config, *args, **kwargs):
        super(BERTModel, self).__init__()
        # Model Architecture
        self.config = config
        self.bert_config = AutoConfig.from_pretrained("bert-base-uncased"),
        if config is not None:
            self.bert_config = update_bert_config(
                bert_config=AutoConfig.from_pretrained("bert-base-uncased"),
                config=config
            )
        self.model = BertForMaskedLM(self.bert_config)

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

    def get_loss(self, logits, targets):
        mask = (targets != -100).float()
        loss = self.recon_loss(logits, targets, mask)
        return loss

    def forward(self, X):
        logits = self.model(**X).logits
        return logits

    def training_step(self, batch):
        target = batch["labels"]

        logits = self.forward({
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        })
        loss = self.get_loss(logits, target)

        logs = {'loss': loss}
        self.log_dict(logs, is_train=True, sync_dist=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # if dist.get_rank() == 0:
        #     print("idx", dataloader_idx)
        #     print("batch", batch)
        if dataloader_idx == 0:
            return self.validation_step_mask(batch)
        elif dataloader_idx == 1:
            return self.validation_step_clean(batch)
        else:
            raise Exception()

    def validation_step_mask(self, batch):
        target = batch["labels"]

        logits = self.forward({
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"]
        })
        loss = self.get_loss(logits, target)

        logs = {'loss_mask': loss}
        self.log_dict(logs, is_train=False, sync_dist=True)
        return {"loss_mask": loss}

    def validation_step_clean(self, batch):
        target = batch["input_ids"]
        mask = batch["attention_mask"]

        logits = self.forward({
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"]
        })
        loss = self.recon_loss(logits, target, mask)

        logs = {'loss_clean': loss}
        self.log_dict(logs, is_train=False, sync_dist=True)
        return {"loss_clean": loss}

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
