import torch
import wandb
from tqdm import tqdm
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertLMHeadModel, T5TokenizerFast

import os
import sys

sys.path.append("/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/")

from data.create_dataset import create_rocstory_dataset, create_wiki_dataset
from utils.util import dict_to_cuda
from model.t5_encoder import T5EncoderModel


def reconstruction_loss(target, prediction_scores, mask):
    if mask is None:
        return cross_entropy(
            input=prediction_scores.view(-1, prediction_scores.shape[-1]),
            target=target.view(-1),
        )

    ce_losses = cross_entropy(
        input=prediction_scores.view(-1, prediction_scores.shape[-1]),
        target=target.view(-1),
        reduce=False,
    )
    ce_losses = ce_losses * mask.reshape(-1)
    ce_loss = torch.sum(ce_losses) / torch.sum(mask)
    return ce_loss


def train(encoder, decoder, tokenizer, tokenizer_gen):
    max_sequence_len = 128
    batch_size = 512
    train_dataset = create_wiki_dataset()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=16,
        pin_memory=True,
    )

    optimizer = torch.optim.Adam(
        decoder.parameters(),
        lr=1e-4,
        #weight_decay=0.01,
        eps=1e-6,
        betas=(0.9, 0.98),
    )

    step = 0
    epochs = 1
    for epoch in range(epochs):
        decoder.train()
        T = tqdm(train_loader)
        for X in T:
            step += 1

            text = tokenizer.batch_decode(X["input_ids"], skip_special_tokens=True)
            X = tokenizer_gen(
                text=text,
                add_special_tokens=True,
                padding="max_length",
                truncation=True,
                max_length=max_sequence_len,
                return_tensors="pt",
            )

            X = dict_to_cuda(X)
            targets = X["input_ids"].type(torch.LongTensor).cuda()
            mask = X["attention_mask"]
            with torch.no_grad():
                emb = encoder(**X)

            sigma = 0.1
            eps = torch.randn_like(emb) * sigma
            logits = decoder(emb + eps)
            loss = reconstruction_loss(targets, logits, mask=None)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                decoder.parameters(),
                max_norm=1.0
            )
            optimizer.step()

            tokens = logits.argmax(dim=-1)
            acc = torch.mean((targets == tokens) * 1.)
            wandb.log({f'train loss': loss.item()}, step=step)
            wandb.log({f'train accuracy': acc.item()}, step=step)

            T.set_description(f"Loss: {loss.item():0.6f}")
            if step >= 10000:
                break

    checkpoints_folder = './checkpoints/'
    name = os.path.join(checkpoints_folder, "decoder-t5_base-wikipedia-128.pth")
    decoder.eval()
    torch.save(
        {
            "decoder": decoder.state_dict(),
        },
        name
    )
    print(f"Save model to: {name}")


class T5Decoder(torch.nn.Module):
    def __init__(self, hidden_size=768, vocab_size=32100, layer_norm_eps=1e-12):
        super().__init__()
        self.dense = torch.nn.Linear(hidden_size, hidden_size)
        self.act_fn = torch.nn.GELU()
        self.LayerNorm = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.decoder = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


def main():
    bert_cfg = "bert-base-uncased"
    tokenizer = BertTokenizerFast.from_pretrained(bert_cfg)

    t5_cfg = "t5-base"
    tokenizer_gen = T5TokenizerFast.from_pretrained(t5_cfg)
    encoder = T5EncoderModel.from_pretrained(
        t5_cfg, enc_normalizer=None
    ).eval().cuda()
    decoder = T5Decoder().train().cuda()

    wandb.init(project="decoders", name="decoder_training_t5_no_wd", mode="online")
    train(encoder, decoder, tokenizer, tokenizer_gen)


main()
