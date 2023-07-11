import torch
import wandb
from tqdm import tqdm
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, RobertaTokenizerFast, T5TokenizerFast, ElectraTokenizerFast

import os
import sys

sys.path.append("/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/")

from data.create_dataset import create_rocstory_dataset, create_wiki_dataset
from utils.util import dict_to_cuda

from model.bert_encoder import BertEncoderModel
from model.t5_encoder import T5EncoderModel
from model.roberta_encoder import RobertaEncoderModel
from model.electra_encoder import ElectraEncoderModel
from model.emb_encoder import EmbEncoderModel

from model.decoder import Decoder


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
        num_workers=1,
        pin_memory=True,
    )

    optimizer = torch.optim.Adam(
        decoder.parameters(),
        lr=5e-3,
        # weight_decay=0.01,
        #eps=1e-6,
        betas=(0.9, 0.98),
    )

    eval_freq = 1000
    eval_mode = False
    step = 0
    epochs = 1
    for epoch in range(epochs):
        decoder.train()
        T = tqdm(train_loader)
        for X in T:
            if (step % eval_freq) == 0:
                eval_mode = True
            if eval_mode:
                decoder.eval()
            else:
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
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    emb = encoder(**X)

            if not eval_mode:
                sigma = 0.
                eps = torch.randn_like(emb) * sigma
                emb = emb + eps

            emb = emb[..., :384]
            logits = decoder(emb)

            loss = reconstruction_loss(targets, logits, mask=None)
            if not eval_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    decoder.parameters(),
                    max_norm=1.0
                )
                optimizer.step()

            tokens = logits.argmax(dim=-1)
            acc = torch.mean((targets == tokens) * 1.)
            if not eval_mode:
                wandb.log({f'train loss': loss.item()}, step=step)
                wandb.log({f'train accuracy': acc.item()}, step=step)
            else:
                wandb.log({f'valid loss': loss.item()}, step=step)
                wandb.log({f'valid accuracy': acc.item()}, step=step)

            T.set_description(f"Loss: {loss.item():0.6f}")
            if step > 2000:
                break

            if eval_mode:
                decoder.train()
                eval_mode = False
                step += 1

    checkpoints_folder = './checkpoints/'
    name = os.path.join(checkpoints_folder, "decoder-bert-encs-384.pth")
    decoder.eval()
    torch.save(
        {
            "decoder": decoder.state_dict(),
        },
        name
    )
    print(f"Save model to: {name}")


def main():
    bert_cfg = "bert-base-uncased"
    tokenizer = BertTokenizerFast.from_pretrained(bert_cfg)

    cfg = "bert-base-uncased"
    tokenizer_gen = BertTokenizerFast.from_pretrained(cfg)
    encoder = BertEncoderModel.from_pretrained(
        cfg, enc_normalizer=None
    ).eval().cuda()

    decoder = Decoder(hidden_size=encoder.config.hidden_size, vocab_size=encoder.config.vocab_size).train().cuda()

    wandb.init(project="decoders", name="bert-encs-384", mode="online")
    train(encoder, decoder, tokenizer, tokenizer_gen)


main()
