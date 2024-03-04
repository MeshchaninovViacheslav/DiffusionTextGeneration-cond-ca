import os
import torch
import wandb
from tqdm import tqdm
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data import create_dataset
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


def get_loaders(config, batch_size):
    train_dataset = next(create_dataset(
        config=config,
    )(
        split="train",
        config=config,
    ).get_data())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=20,
        pin_memory=True,
    )

    valid_dataset = next(create_dataset(
        config=config,
    )(
        split="test",
        config=config,
    ).get_data())

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=20,
        pin_memory=True,
    )

    return train_loader, valid_loader


def loss_step(batch, tokenizer, encoder, decoder, config, eval=False):
    trg = tokenizer(
            batch['text_trg'],
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=config.data.max_sequence_len,
            return_tensors="pt",
            return_special_tokens_mask=True,
            return_token_type_ids=False,
        )
    targets = trg["input_ids"].cuda()
    
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        latent = encoder(**{
            "input_ids": targets,
            "attention_mask": trg["attention_mask"].cuda()
        }).cuda()

    if not eval:
        sigma = config.decoder.std_aug
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

    batch_size = config.decoder.batch_size
    eval_freq = config.decoder.eval_freq
    step = 0

    train_loader, valid_loader = get_loaders(
        config=config,
        batch_size=batch_size
    )

    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=config.decoder.lr
    )
    
    for _ in range(config.decoder.num_epochs):
        decoder.train()

        for batch in tqdm(train_loader):
            loss, acc = loss_step(
                batch=batch,
                tokenizer=tokenizer,
                encoder=encoder,
                decoder=decoder,
                config=config,
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
                for batch in tqdm(valid_loader):
                    with torch.no_grad():
                        loss, acc = loss_step(
                            batch=batch,
                            tokenizer=tokenizer,
                            encoder=encoder,
                            decoder=decoder,
                            config=config,
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


    decoder = BertDecoder(model_name=config.model.encoder_name, bert_config=config.decoder.base_config).train().cuda()

    exp_name = config.model.decoder_path.split("/")[-1].replace(".pth", "")
    wandb.init(project=config.project_name, name=exp_name, mode="online")
    train(config, encoder, decoder, tokenizer)


main()
