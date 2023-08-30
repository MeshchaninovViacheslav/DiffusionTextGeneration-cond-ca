import torch
import wandb
import ml_collections
from tqdm import tqdm
from torch.nn.functional import cross_entropy
from transformers import BertConfig
import random
import os
import sys
import torch.distributed as dist

sys.path.append("/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/")

from utils.util import dict_to_cuda

import diffusion_utils.schedulers as schedulers
from diffusion_holder import DiffusionRunner
from utils.util import set_seed


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


def create_config():
    config = ml_collections.ConfigDict()

    training = config.training = ml_collections.ConfigDict()
    training.ode_sampling = False
    training.checkpoints_folder = '../checkpoints'
    training.batch_size = 512
    config.checkpoints_prefix = None

    validation = config.validation = ml_collections.ConfigDict()
    validation.batch_size = 512

    dynamic = config.dynamic = ml_collections.ConfigDict()
    dynamic.solver = 'euler'
    dynamic.scheduler = "sd"
    dynamic.N = 200
    dynamic.beta_min = 0.1
    dynamic.beta_max = 20
    dynamic.ode_sampling = False
    dynamic.coef_d = 10

    model = config.model = ml_collections.ConfigDict()
    model.ema_rate = 0.9999
    model.enc_type = "base"
    model.embeddings_type = "embeddings"
    model.dif_enc_type = "base"
    model.downstream_task = ""  # "qqp"
    model.dataset = "wikipedia"  # "glue"
    model.prediction = "x_0"
    model.loss = "L_x_0"
    model.decoder_path = "decoder-wikipedia-128.pth"

    data = config.data = ml_collections.ConfigDict()
    data.max_sequence_len = 64
    data.pos_begin = 0.0
    data.pos_end = 0.67
    data.enc_bert_mean = "/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/data/encodings-bert_base-wiki-mean.pt"
    data.enc_bert_std = "/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/data/encodings-bert_base-wiki-std.pt"

    data.enc_t5_mean = "/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/data/encodings-t5-wiki-mean.pth"
    data.enc_t5_std = "/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/data/encodings-t5-wiki-std.pth"

    config.finetuning = False
    config.seed = 0
    config.ddp = True
    config.bert_config = BertConfig.from_pretrained("bert-base-uncased")

    config.project_name = "dtg-exps-1.0"
    config.classifier_guidance_scale = 0.
    config.use_self_cond = True
    config.checkpoints_prefix = "wikipedia--t5-bert-self_cond_last_"

    return config


def train(config):
    seed = config.seed + dist.get_rank()
    set_seed(seed)

    exp_name = "noisy-diffusion"
    if dist.get_rank() == 0:
        wandb.init(project=config.project_name, name=exp_name, mode="online")

    diffusion = DiffusionRunner(config, latent_mode=config.model.embeddings_type, eval=True)

    decoder = diffusion.decoder.train()
    ddp_decoder = torch.nn.parallel.DistributedDataParallel(
        decoder,
        device_ids=[config.local_rank],
        broadcast_buffers=False,
    )

    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=5e-5,
        weight_decay=0.01,
        eps=1e-6,
        betas=(0.9, 0.98),
    )

    eval_freq = 100
    eval_mode = False
    step = 0
    epochs = 1
    for epoch in range(epochs):
        diffusion.set_train_data_generator()
        decoder.train()
        T = tqdm(diffusion.train_loader)
        for X in T:
            if (step % eval_freq) == 0:
                eval_mode = True
            if eval_mode:
                ddp_decoder.eval()
            else:
                step += 1

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    X = dict_to_cuda(X)
                    clean_X = diffusion.encoder_gen(**{"input_ids": X["input_ids"], "attention_mask": X["input_mask"]})
                    cond = diffusion.encoder_cond(**{"input_ids": X["cond_ids"], "attention_mask": X["cond_mask"]})

                    mask = None  # X["input_mask"]

                    # Noizing
                    batch_size = clean_X.size(0)

                    t = diffusion.sample_time(batch_size, eps=1e-5)
                    marg_forward = diffusion.dynamic.marginal(clean_X, t)
                    x_t, noise, score_clean = marg_forward['x_t'], marg_forward['noise'], marg_forward['score']

                    # self-cond estimate
                    x_0_self_cond = torch.zeros_like(x_t, dtype=x_t.dtype)
                    if diffusion.use_self_cond:
                        x_0_self_cond = diffusion.ddp_score_estimator(
                            x_t=x_t, time_t=t, cond=cond,
                            attention_mask=mask, cond_mask=X["cond_mask"],
                            x_0_self_cond=x_0_self_cond
                        ).detach()

                    # model prediction
                    x_0 = diffusion.calc_score(
                        diffusion.ddp_score_estimator,
                        x_t, t,
                        cond=cond,
                        cond_mask=X["cond_mask"],
                        attention_mask=mask,
                        x_0_self_cond=x_0_self_cond,
                    )["x_0"]
                    pred_embeddings = diffusion.gen_enc_normalizer.denormalize(x_0)

            logits = ddp_decoder(pred_embeddings)
            targets = X["input_ids"]

            loss = reconstruction_loss(targets, logits, mask=None)
            if not eval_mode:
                optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    decoder.parameters(),
                    max_norm=1.0
                )
                optimizer.step()

            tokens = logits.argmax(dim=-1)
            acc = torch.mean((targets == tokens) * 1.)

            if dist.get_rank() == 0:
                if not eval_mode:
                    wandb.log({f'train loss': loss.item()}, step=step)
                    wandb.log({f'train accuracy': acc.item()}, step=step)
                    wandb.log({f'grad_norm': grad_norm.item()}, step=step)
                else:
                    wandb.log({f'valid loss': loss.item()}, step=step)
                    wandb.log({f'valid accuracy': acc.item()}, step=step)

                T.set_description(f"Loss: {loss.item():0.6f}")

            if eval_mode:
                decoder.train()
                eval_mode = False
                step += 1

    if dist.get_rank() == 0:
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

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    config.local_rank = rank
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    config.training.batch_size_per_gpu = config.training.batch_size // dist.get_world_size()
    train(config)


main()
