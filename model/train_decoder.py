import torch
import wandb
from tqdm import tqdm
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import os
import sys

sys.path.append("/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/")

from data.dataset import RocStoryDatasetDDP

from utils.util import dict_to_cuda

from model.encoder_t5 import T5EncoderModel
from model.encoder_roberta import RobertaEncoderModel
from model.electra_encoder import ElectraEncoderModel
from model.emb_encoder import EmbEncoderModel
from model.decoder import BertDecoder
from model.encoder_bert import BertEncoderModel
from create_config import create_config
from model.encoder_bart import BartEncoderModel


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


def get_loaders(tokenizer, max_sequence_len, batch_size):
    train_dataset = next(RocStoryDatasetDDP(
        split="train",
        tokenizer=tokenizer,
        max_sequence_len=max_sequence_len,
    ).get_data())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=20,
        pin_memory=True,
    )

    valid_dataset = next(RocStoryDatasetDDP(
        split="valid",
        tokenizer=tokenizer,
        max_sequence_len=max_sequence_len,
    ).get_data())

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=20,
        pin_memory=True,
    )

    return train_loader, valid_loader


def loss_step(X, tokenizer, encoder, decoder, eval=False):
    X = dict_to_cuda(X)
    targets = X["input_ids"].type(torch.LongTensor).cuda()
    
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        #with torch.autocast(device_type='cuda', dtype=torch.float32):
        latent = encoder(**{
            "input_ids": X["input_ids"],
            "attention_mask": X["input_mask"]
        }).cuda()

        embeddings = encoder.module.bert.embeddings.word_embeddings.weight.data.cpu()

        cls_emb = embeddings[tokenizer.cls_token_id].cuda()
        sep_emb = embeddings[tokenizer.sep_token_id].cuda()
        pad_emb = embeddings[tokenizer.pad_token_id].cuda()

        attention_mask = X["input_mask"]
        latent[:, 0] = cls_emb
        latent[torch.arange(len(latent)), attention_mask.sum(-1) - 1] = sep_emb
        latent[~attention_mask.bool()] = pad_emb

    if not eval:
        sigma = 0.2
        eps = torch.randn_like(latent) * sigma
        latent = latent + eps
    
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits = decoder(latent)
    loss = reconstruction_loss(targets, logits, mask=None)
    
    tokens = logits.argmax(dim=-1)
    acc = torch.mean((targets == tokens) * 1.)

    return loss, acc


def train(encoder, decoder, tokenizer, exp_name):
    total_number_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"Num params: {total_number_params}")

    max_sequence_len = 128
    batch_size = 512

    train_loader, valid_loader = get_loaders(
        tokenizer=tokenizer,
        max_sequence_len=max_sequence_len,
        batch_size=batch_size
    )

    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=5e-5,
        weight_decay=0.001,
        betas=(0.9, 0.98),
    )

    eval_freq = 100
    step = 0
    epochs = 10
    for _ in range(epochs):
        decoder.train()

        for X in tqdm(train_loader):
            loss, acc = loss_step(
                X=X,
                tokenizer=tokenizer,
                encoder=encoder,
                decoder=decoder
            )
       
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                decoder.parameters(),
                max_norm=1.0
            )
            optimizer.step()

            wandb.log({f'train loss': loss.item()}, step=step)
            wandb.log({f'train accuracy': acc.item()}, step=step)

            step += 1

            if step % eval_freq == 0:
                decoder.eval()
                for X in tqdm(valid_loader):
                    with torch.no_grad():
                        loss, acc = loss_step(
                            X=X,
                            tokenizer=tokenizer,
                            encoder=encoder,
                            decoder=decoder,
                            eval=True
                        )
                decoder.train()

                wandb.log({f'valid loss': loss.item()}, step=step)
                wandb.log({f'valid accuracy': acc.item()}, step=step)

    checkpoints_folder = '/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/checkpoints/'
    name = os.path.join(checkpoints_folder, f"decoder-{exp_name}.pth")
    decoder.eval()
    torch.save(
        {
            "decoder": decoder.state_dict(),
        },
        name
    )
    print(f"Save model to: {name}")


def main():
    config = create_config()
    cfg = config.model.encoder_name
    tokenizer = AutoTokenizer.from_pretrained(cfg)

    encoder = BertEncoderModel.from_pretrained(
        cfg,
        enc_normalizer=None
    ).eval()
    encoder = torch.nn.DataParallel(encoder).cuda()

    decoder = BertDecoder(model_name=cfg, mode="transformer").train().cuda()

    exp_name = f"{cfg}-transformer-spt"
    wandb.init(project="rocstory-decoders", name=exp_name, mode="online")
    train(encoder, decoder, tokenizer, exp_name=exp_name)


main()
