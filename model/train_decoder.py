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
        dataset_name=config.data.dataset_name,
    )(
        split="train",
        base_path=config.data.dataset_path,
        max_sequence_len=config.data.max_sequence_len,
        max_context_len=config.data.max_context_len,
    ).get_data())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=20,
        pin_memory=True,
    )

    valid_dataset = next(create_dataset(
        dataset_name=config.data.dataset_name,
    )(
        split="test",
        base_path=config.data.dataset_path,
        max_sequence_len=config.data.max_sequence_len,
        max_context_len=config.data.max_context_len,
    ).get_data())

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=20,
        pin_memory=True,
    )

    return train_loader, valid_loader


def loss_step(batch, encoder_gen, tokenizer_gen, encoder_cond, tokenizer_cond, decoder, config, eval=False):
    cond = tokenizer_cond(
        batch["text_src"],
        add_special_tokens=True,
        padding=True,
        max_length=config.data.max_context_len,
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    trg = tokenizer_gen(
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
        cond_x = encoder_cond(**{
            "input_ids": cond["input_ids"].cuda(),
            "attention_mask": cond["attention_mask"].cuda()
        })
        latent = encoder_gen(**{
            "input_ids": targets,
            "attention_mask": trg["attention_mask"].cuda()
        }).cuda()

    if not eval:
        sigma = 0.2
        eps = torch.randn_like(latent) * sigma
        latent = latent + eps

    latent = encoder_gen.module.enc_normalizer.denormalize(latent)
    
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        mask = cond["attention_mask"].cuda()
        if not config.model.decoder_is_cond:
            cond_x = None
            mask = None
        logits = decoder(
            hidden_states=latent, 
            encoder_hidden_states=cond_x,
            encoder_attention_mask=mask,
            )
    loss = reconstruction_loss(targets, logits, mask=None)
    
    tokens = logits.argmax(dim=-1)
    acc = torch.mean((targets == tokens) * 1.)

    return loss, acc


def train(config, encoder_gen, tokenizer_gen, encoder_cond, tokenizer_cond, decoder):
    total_number_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"Num params: {total_number_params}")

    batch_size = 512

    train_loader, valid_loader = get_loaders(
        config=config,
        batch_size=batch_size
    )

    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=1e-4,
        weight_decay=0.001,
        betas=(0.9, 0.98),
    )

    eval_freq = 100
    step = 0
    epochs = 10
    for _ in range(epochs):
        decoder.train()

        for batch in tqdm(train_loader):
            loss, acc = loss_step(
                batch=batch,
                encoder_gen=encoder_gen, 
                tokenizer_gen=tokenizer_gen, 
                encoder_cond=encoder_cond, 
                tokenizer_cond=tokenizer_cond, 
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
                            encoder_gen=encoder_gen, 
                            tokenizer_gen=tokenizer_gen, 
                            encoder_cond=encoder_cond, 
                            tokenizer_cond=tokenizer_cond,
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
    
    tokenizer_cond = AutoTokenizer.from_pretrained(config.model.conditional_encoder_name)
    encoder_cond = Encoder(
        config.model.conditional_encoder_name,
        enc_normalizer=None,
        is_change_sp_tokens=False,
    ).eval()
    encoder_cond = torch.nn.DataParallel(encoder_cond).cuda()

    tokenizer_gen = AutoTokenizer.from_pretrained(config.model.encoder_name)
    enc_normalizer = EncNormalizer(
        enc_mean_path=config.data.enc_gen_mean,
        enc_std_path=config.data.enc_gen_std,
    )
    encoder_gen = Encoder(
        config.model.encoder_name,
        enc_normalizer=enc_normalizer,
        is_change_sp_tokens=True,
    ).eval()
    encoder_gen = torch.nn.DataParallel(encoder_gen).cuda()


    decoder = BertDecoder(model_name=config.model.encoder_name, mode="transformer", is_cond=config.model.decoder_is_cond).train().cuda()

    exp_name = config.model.decoder_path.replace(".pth", "").split("/")[-1]
    wandb.init(project=config.project_name, name=exp_name, mode="online")
    train(
        config, 
        encoder_gen=encoder_gen, 
        tokenizer_gen=tokenizer_gen, 
        encoder_cond=encoder_cond, 
        tokenizer_cond=tokenizer_cond, 
        decoder=decoder
    )


main()
