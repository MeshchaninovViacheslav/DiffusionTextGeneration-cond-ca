import torch
import lightning as L
from transformers import AutoConfig, AutoModelForMaskedLM, BertPreTrainedModel
from torch.nn.functional import cross_entropy

from timm.scheduler.cosine_lr import CosineLRScheduler

from typing import Dict, Any, Optional
from torch import FloatTensor, Tensor

from lm_training.util import calc_model_grads_norm, MyAccuracy

import torch.distributed as dist


def update_bert_config(bert_config, config):
    for key in config.bert_config:
        bert_config.__dict__[key] = config.bert_config[key]
    return bert_config


class BertLMPredictionHead(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = torch.nn.Sequential(
            torch.nn.Linear(config.embedding_size, config.hidden_size),
            torch.nn.GELU(),
            torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        )

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = torch.nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BERT(BertPreTrainedModel):
    def __init__(self, config):
        super(BERT, self).__init__(config)

        from transformers import BertModel

        self.encoder = BertModel(config, add_pooling_layer=False)
        self.projector = torch.nn.Linear(config.hidden_size, config.embedding_size)
        self.cls = BertLMPredictionHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return_dict = {}

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]

        return_dict["last_hidden_state"] = outputs

        embeddings = self.projector(outputs)
        return_dict["embeddings"] = embeddings

        logits = self.cls(embeddings)
        return_dict["logits"] = logits
        return return_dict


class BERTModel(L.LightningModule):
    def __init__(self, config, *args, **kwargs):
        super(BERTModel, self).__init__()
        # Model Architecture
        self.config = config
        self.finetune = config.finetune
        self.bert_config = AutoConfig.from_pretrained(config.model_name),
        if config is not None:
            self.bert_config = update_bert_config(
                bert_config=AutoConfig.from_pretrained(config.model_name),
                config=config
            )
        self.model = BERT(self.bert_config)
        if self.bert_config.encoder_initialization is not None:
            self.model.encoder.from_pretrained(self.bert_config.encoder_initialization)

        self.loss_type = config.loss_type

        if config.hg_pretrain:
            self.model = AutoModelForMaskedLM.from_pretrained(config.model_name)
        # if self.finetune:
        #     self.model.cls = torch.nn.Linear(self.bert_config.hidden_size, 1)

        self.accuracy = MyAccuracy()

    def recon_loss(self, inputs, outputs, mask=None):
        if mask is None:
            mask = torch.ones(
                inputs.shape[:-1],
                requires_grad=False,
                dtype=torch.float,
                device=inputs.device,
            )

        # print("mask target", torch.sum(mask))
        losses = cross_entropy(
            input=inputs.reshape(-1, inputs.shape[-1]),
            target=outputs.reshape(-1),
            reduce=False,
        )
        losses = losses * mask.reshape(-1)
        loss = torch.sum(losses) / torch.sum(mask)
        return loss

    def get_loss(self, logits, batch):
        if self.loss_type == "mlm":
            targets = batch["labels"]
            mask = (targets != -100).float()
            # print("mask target", torch.mean(mask).item())
            loss = self.recon_loss(logits, targets, mask)
            # print("loss", loss)
            return loss
        elif self.loss_type == "denoising":
            mask = (batch["labels"] != -100).int()
            targets = batch["labels"] * mask + batch["input_ids"] * (1 - mask)
            mask = None
            loss = self.recon_loss(logits, targets, mask)
            return loss
        else:
            raise TypeError

    def forward(self, X):
        outputs = self.model(**X)
        logits = outputs["logits"]
        return logits

    def training_step(self, batch):
        if self.finetune:
            return self.finetune_step(batch)

        target = batch["labels"]

        logits = self.forward({
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            #"labels": target,
        })
        loss = self.get_loss(logits, batch)

        logs = {'loss': loss, "mask_percentage": torch.mean(batch["attention_mask"].float()).item()}
        self.log_dict(logs, is_train=True, sync_dist=True)
        return {'loss': loss}

    def finetune_step(self, batch):
        target = batch["input_ids"]
        target = target[:, 1]

        positive_ind = 2748
        negative_ind = 2053

        target = torch.where(
            target == positive_ind,
            1,
            0
        ).float()

        logits = self.forward({
            "input_ids": batch["cond_ids"],
            "attention_mask": batch["cond_mask"],
        })

        logits = logits[:, 0].reshape(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        logs = {'sst_loss': loss}
        self.log_dict(logs, is_train=True, sync_dist=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if self.finetune:
            return self.validation_fn_step(batch)
        if dataloader_idx == 0:
            return self.validation_step_mask(batch)
        elif dataloader_idx == 1:
            return self.validation_step_clean(batch)
        else:
            raise Exception()

    def validation_fn_step(self, batch):
        target = batch["input_ids"]
        target = target[:, 1]

        positive_ind = 2748
        negative_ind = 2053

        target = torch.where(
            target == positive_ind,
            1,
            0
        ).float()

        logits = self.forward({
            "input_ids": batch["cond_ids"],
            "attention_mask": batch["cond_mask"],
        })
        logits = logits[:, 0].reshape(-1)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)

        label = (torch.sigmoid(logits) > 0.5).float()

        accuracy = self.accuracy(label, target)

        logs = {'sst_accuracy': accuracy, 'sst_loss': loss}
        self.log_dict(logs, is_train=False, sync_dist=True, on_epoch=True, on_step=False)
        return logs

    def validation_step_mask(self, batch):
        target = batch["labels"]

        logits = self.forward({
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"]
        })
        loss = self.get_loss(logits, batch)

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
            list(self.model.projector.parameters()) + list(self.model.cls.parameters()),#self.model.parameters(),
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

    def on_validation_epoch_start(self) -> None:
        self.accuracy = MyAccuracy().to(device=self.device)

    def predict_step(self, batch, batch_idx):
        logits = self.forward({
            "input_ids": batch["cond_ids"],
            "attention_mask": batch["cond_mask"],
        })
        logits = logits[:, 0].reshape(-1)
        label = (torch.sigmoid(logits) > 0.5).float()
        return label.tolist()
