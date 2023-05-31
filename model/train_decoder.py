import torch
import wandb
from tqdm import tqdm
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertLMHeadModel

import os
import sys

sys.path.append("/home/vmeshchaninov/DiffusionTextGeneration/")

from data.create_dataset import create_rocstory_dataset, create_unsupervised_dataset
from utils.util import dict_to_cuda


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

def train(encoder, decoder, tokenizer):
    max_sequence_len = 64
    batch_size = 512
    train_dataset = create_unsupervised_dataset(
        split="train",
        tokenizer=tokenizer,
        max_sequence_len=max_sequence_len,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=16,
        pin_memory=True,
    )
    valid_dataset = create_rocstory_dataset(
        split="validation",
        tokenizer=tokenizer,
        max_sequence_len=max_sequence_len,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=16,
        pin_memory=True,
    )

    optimizer = torch.optim.Adam(
        decoder.parameters(),
        lr=1e-4,
    )

    step = 0
    epochs = 1
    for epoch in range(epochs):
        decoder.train()
        for X in tqdm(train_loader):
            step += 1
            X = dict_to_cuda(X)
            targets = X["input_ids"]
            mask = X["attention_mask"]
            with torch.no_grad():
                emb = encoder(**X).last_hidden_state

            sigma = 0.1
            eps = torch.randn_like(emb) * sigma
            logits = decoder(emb + eps)
            loss = reconstruction_loss(targets, logits, mask=None)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tokens = logits.argmax(dim=-1)
            acc = torch.mean((targets == tokens) * 1.)
            wandb.log({f'train loss': loss.item()}, step=step)
            wandb.log({f'train accuracy': acc.item()}, step=step)

        decoder.eval()
        with torch.no_grad():
            valid_loss = 0.
            valid_num = 0.
            for X in tqdm(valid_loader):
                X = dict_to_cuda(X)
                targets = X["input_ids"]
                mask = X["attention_mask"]
                emb = encoder(**X).last_hidden_state
                logits = decoder(emb)
                loss = reconstruction_loss(targets, logits, mask)
                valid_loss += loss.item() * emb.shape[0]
                valid_num += emb.shape[0]
            wandb.log({f'valid loss': valid_loss / valid_num}, step=step)

    checkpoints_folder = './checkpoints/'
    name = os.path.join(checkpoints_folder, "decoder-c4-pad-64.pth")
    decoder.eval()
    torch.save(
        {
            "decoder": decoder.state_dict(),
        },
        name
    )
    print(f"Save model to: {name}")


def main():
    pretrained_enc_type = "bert-base-uncased"
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_enc_type)
    bert = BertLMHeadModel.from_pretrained(pretrained_enc_type).cuda().eval()
    encoder = bert.bert.eval()
    decoder = bert.cls.train()

    wandb.init(project="bert_diffusion", name="decoder_training", mode="online")
    train(encoder, decoder, tokenizer)


main()
