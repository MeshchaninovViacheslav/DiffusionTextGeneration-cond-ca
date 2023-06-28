import sys

sys.path.append("/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/")

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from torch.nn.functional import cross_entropy
from lightning import seed_everything, Trainer
import lightning as L

from data.dataset_clean_wiki import WikipediaCleanDatasetUnconditional


class GPTModel(L.LightningModule):
    def __init__(self, ):
        super(GPTModel, self).__init__()
        # Model Architecture
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
        # if self.config.wandb:
        #     wandb_log(loss=loss.item())
        return {'loss': loss, 'log': logs}

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0)
        return [opt], []


def main():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token_id = 50256

    dataset = next(WikipediaCleanDatasetUnconditional(
        split="test",
        tokenizer=tokenizer,
        max_sequence_len=128,
    ).get_data())
    loader = DataLoader(dataset, batch_size=2)

    trainer = Trainer()
    model = GPTModel()
    trainer.fit(model, train_dataloaders=loader)


main()