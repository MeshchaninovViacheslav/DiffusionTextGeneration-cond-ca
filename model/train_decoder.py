import torch
import wandb
from tqdm import tqdm
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import os
import sys

sys.path.append("/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/")

from data import create_dataset

from utils.util import dict_to_cuda

from model.decoder import BertDecoder
from model import Encoder
from model.enc_normalizer import EncNormalizer
from create_config import create_config


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


def get_loaders(config, tokenizer, max_sequence_len, batch_size):
    train_dataset = next(create_dataset(
        dataset_name=config.data.dataset_name,
    )(
        split="train",
        tokenizer_cond=tokenizer,
        tokenizer_gen=tokenizer,
        train_path=config.data.train_path,
        valid_path=config.data.valid_path,
        max_sequence_len=config.data.max_sequence_len + config.data.max_context_len,
        max_context_len=0,
    ).get_data())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=20,
        pin_memory=True,
    )

    train_dataset = next(create_dataset(
        dataset_name=config.data.dataset_name,
    )(
        split="train",
        tokenizer_cond=tokenizer,
        tokenizer_gen=tokenizer,
        train_path=config.data.train_path,
        valid_path=config.data.valid_path,
        max_sequence_len=config.data.max_sequence_len + config.data.max_context_len,
        max_context_len=0,
    ).get_data())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=20,
        pin_memory=True,
    )

    return train_loader, valid_loader


def loss_step(X, tokenizer, encoder, decoder, eval=False):
    X = dict_to_cuda(X)
    targets = X["input_ids"].type(torch.LongTensor).cuda()
    
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        latent = encoder(**{
            "input_ids": X["input_ids"],
            "attention_mask": X["input_mask"]
        }).cuda()

    if not eval:
        sigma = 0.2
        eps = torch.randn_like(latent) * sigma
        latent = latent + eps

    latent = encoder.module.enc_normalizer.denormalize(latent)
    
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits = decoder(latent)
    loss = reconstruction_loss(targets, logits, mask=None)
    
    tokens = logits.argmax(dim=-1)
    acc = torch.mean((targets == tokens) * 1.)

    return loss, acc


def train(config, encoder, decoder, tokenizer):
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

    os.makedirs(config.training.checkpoints_folder, exist_ok=True)
    decoder.eval()
    torch.save(
        {
            "decoder": decoder.state_dict(),
        },
        config.model.decoder_path
    )
    print(f"Save model to: {config.model.decoder_path}")


def main():
    config = create_config()
    tokenizer = AutoTokenizer.from_pretrained(config.model.encoder_name)
    enc_normalizer = EncNormalizer(
        enc_mean_path=config.data.enc_gen_mean,
        enc_std_path=config.data.enc_gen_std,
    )
    encoder = Encoder(
        config.model.encoder_name,
        enc_normalizer=enc_normalizer,
        is_change_sp_tokens=True,
    ).eval()
    encoder = torch.nn.DataParallel(encoder).cuda()


    decoder = BertDecoder(model_name=config.model.encoder_name, mode="transformer").train().cuda()

    exp_name = f"{config.model.encoder_name_hash}-transformer"
    wandb.init(project=config.project_name, name=exp_name, mode="online")
    train(config, encoder, decoder, tokenizer)


main()
