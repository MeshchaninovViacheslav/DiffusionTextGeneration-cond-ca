import os
import torch
import wandb
from tqdm import tqdm
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch.distributed as dist

from data import create_dataset
from autoencoder.config import create_config
from autoencoder.autoencoder import AutoEncoder
from utils import set_seed, setup_ddp, reduce_tensor


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


def loss_step(batch, tokenizer, autoencoder, config):
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
    mask = trg["attention_mask"].cuda()
    
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits = autoencoder(
            input_ids=targets,
            attention_mask=mask
        )

    loss = reconstruction_loss(targets, logits, mask=None)
    
    tokens = logits.argmax(dim=-1)
    acc = torch.mean((targets == tokens) * 1.)

    return loss, acc


def train(config, autoencoder, tokenizer):
    ddp_autoencoder = torch.nn.parallel.DistributedDataParallel(
        autoencoder,
        device_ids=[dist.get_rank()],
        broadcast_buffers=False,
        find_unused_parameters=True
    )

    total_number_params = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)
    print(f"Num params: {total_number_params}")

    batch_size = config.optim.batch_size
    eval_freq = config.optim.eval_freq
    step = 0

    train_loader, valid_loader = get_loaders(
        config=config,
        batch_size=batch_size
    )

    optimizer = torch.optim.AdamW(
        params=list(autoencoder.compressor.parameters()) + \
               list(autoencoder.decoder.parameters()) + \
               list(autoencoder.projector.parameters()),
        lr=config.optim.lr
    )
    
    for _ in range(config.optim.num_epochs):
        autoencoder.train()

        for batch in tqdm(train_loader):
            loss, acc = loss_step(
                batch=batch,
                tokenizer=tokenizer,
                autoencoder=ddp_autoencoder,
                config=config,
            )
       
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                autoencoder.parameters(),
                max_norm=config.optim.clip_norm
            )
            optimizer.step()

            if dist.get_rank() == 0:
                wandb.log({f'train loss': loss.item()}, step=step)
                wandb.log({f'train accuracy': acc.item()}, step=step)

            step += 1

            if step % eval_freq == 0:
                autoencoder.eval()
                total_loss = torch.Tensor([0.0])
                total_acc = torch.Tensor([0.0])

                for batch in tqdm(valid_loader):
                    with torch.no_grad():
                        loss, acc = loss_step(
                            batch=batch,
                            tokenizer=tokenizer,
                            autoencoder=autoencoder,
                            config=config
                        )

                        total_loss += loss.cpu()
                        total_acc += acc.cpu()
                
                total_loss = reduce_tensor(total_loss.cuda())
                total_acc = reduce_tensor(total_acc.cuda())

                autoencoder.train()

                if dist.get_rank() == 0:
                    wandb.log({f'valid loss': total_loss.item()}, step=step)
                    wandb.log({f'valid accuracy': total_acc.item()}, step=step)
    
    if dist.get_rank() == 0:
        os.makedirs(config.checkpoints_folder, exist_ok=True)
        autoencoder.eval()
        torch.save(
            {
                "autoencoder": autoencoder.state_dict(),
            },
            config.save_path
        )
        print(f"Save model to: {config.save_path}")


def main():
    config = create_config()
    config.optim.batch_size_per_gpu = config.optim.batch_size // dist.get_world_size()
    if dist.get_rank() == 0:
        print(config)

    tokenizer = AutoTokenizer.from_pretrained(config.encoder.name)
    autoencoder = AutoEncoder(config=config).cuda()

    exp_name = config.exp_name
    wandb.init(project=config.project_name, name=exp_name, mode="online")
    
    seed = config.seed + dist.get_rank()
    set_seed(seed)
    train(config, autoencoder, tokenizer)

if __name__ == '__main__':
    setup_ddp()
    main()
    
