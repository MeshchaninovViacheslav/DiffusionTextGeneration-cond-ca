import os
import torch
import wandb
from tqdm import tqdm, trange
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


def get_datasets(config):
    train_dataset = create_dataset(
        config=config,
    )(
        split="train",
        config=config,
    )

    test_dataset = create_dataset(
        config=config,
    )(
        split="test",
        config=config,
    )
    return train_dataset, test_dataset
    

def get_loaders(train_dataset, valid_dataset, batch_size):
    train_loader = DataLoader(
        next(train_dataset.get_data()),
        batch_size=batch_size,
        shuffle=True,
        num_workers=20,
        pin_memory=False,
    )

    valid_loader = DataLoader(
        next(valid_dataset.get_data()),
        batch_size=batch_size,
        num_workers=20,
        pin_memory=False,
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


def save_checkpoint(model, config, step):
    if dist.get_rank() == 0:
        os.makedirs(config.checkpoints_folder, exist_ok=True)
        model.eval()
        save_pth = f"{config.save_path}-{step:06d}.pth" 
        torch.save(
            {
                "autoencoder": model.state_dict(),
            },
            save_path 
        )
        print(f"Save model to: {save_path}")


def train(config, autoencoder, tokenizer):
    ddp_autoencoder = torch.nn.parallel.DistributedDataParallel(
        autoencoder,
        device_ids=[dist.get_rank()],
        broadcast_buffers=False,
        find_unused_parameters=True
    )

    total_number_params = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)
    print(f"Num params: {total_number_params}")

    batch_size = config.optim.batch_size_per_gpu
    eval_freq = config.optim.eval_freq
    total_step = 0

    train_dataset, valid_dataset = get_datasets(config=config)

    optimizer = torch.optim.AdamW(
        params=list(autoencoder.compressor.parameters()) + \
               list(autoencoder.decoder.parameters()) + \
               list(autoencoder.projector.parameters()),
        lr=config.optim.lr
    )
    train_range = iter(trange(1, config.optim.num_steps + 1))

    while True: 
        autoencoder.train()
        train_loader, valid_loader = get_loaders(
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            batch_size=batch_size
        )

        for batch in train_loader:
            next(train_range)
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
                wandb.log({f'train loss': loss.item()}, step=total_step)
                wandb.log({f'train accuracy': acc.item()}, step=total_step)

            total_step += 1

            if total_step % eval_freq == 0:
                autoencoder.eval()
                total_loss = torch.Tensor([0.0])
                total_acc = torch.Tensor([0.0])
                total_num = torch.Tensor([0.0])

                for batch in valid_loader:
                    with torch.no_grad():
                        loss, acc = loss_step(
                            batch=batch,
                            tokenizer=tokenizer,
                            autoencoder=autoencoder,
                            config=config
                        )

                        total_loss += loss.cpu()
                        total_acc += acc.cpu()
                        total_num += 1
                
                total_loss = reduce_tensor((total_loss / total_num).cuda())
                total_acc = reduce_tensor((total_acc / total_num).cuda())

                autoencoder.train()

                if dist.get_rank() == 0:
                    wandb.log({f'valid loss': total_loss.item()}, step=total_step)
                    wandb.log({f'valid accuracy': total_acc.item()}, step=total_step)

            if total_step % config.optim.checkpoint_freq == 0:
                save_checkpoint(autoencoder, config, total_step)

            if total_step >= config.optim.num_steps:
                break
    
    save_checkpoint(autoencoder, config, total_step)


def main():
    config = create_config()
    config.optim.batch_size_per_gpu = config.optim.batch_size // dist.get_world_size()
    if dist.get_rank() == 0:
        print(config)

    tokenizer = AutoTokenizer.from_pretrained(config.encoder.name)
    autoencoder = AutoEncoder(config=config).cuda()

    exp_name = config.exp_name
    if dist.get_rank() == 0:
        wandb.init(project=config.project_name, name=exp_name, mode="online")
    
    seed = config.seed + dist.get_rank()
    set_seed(seed)
    train(config, autoencoder, tokenizer)


if __name__ == '__main__':
    setup_ddp()
    main()
    
