import os
import sys

sys.path.append("/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/")

import torch
import ml_collections

from lm_training.bert_lightning import BERTModel


def create_config():
    config = ml_collections.ConfigDict()
    optim = config.optim = ml_collections.ConfigDict()
    optim.grad_clip_norm = 5.
    optim.linear_warmup = 5000
    optim.lr = 2e-4
    optim.min_lr = 1e-4
    optim.warmup_lr = 1e-8
    optim.weight_decay = 0.01
    optim.beta_1 = 0.9
    optim.beta_2 = 0.98
    optim.eps = 1e-6
    optim.precision = "16"

    training = config.training = ml_collections.ConfigDict()
    training.training_iters = 500_000
    training.training_iters = training.training_iters
    training.checkpoint_freq = 100_000
    training.eval_freq = 5_000
    training.batch_size = 512 // torch.cuda.device_count()

    data = config.data = ml_collections.ConfigDict()
    data.max_sequence_len = 128
    data.num_workers = 16
    data.bert_recon_dataset = True

    model = config.model = ml_collections.ConfigDict()
    model.mlm_probability = 0.15
    model.pad_to_multiple_of = None

    bert_config = config.bert_config = ml_collections.ConfigDict()
    bert_config.hidden_size = 768

    config.project_name = "lm-training"
    config.exp_name = f"bert-training-{bert_config.hidden_size}"
    config.seed = 0
    config.finetune = False
    config.hg_pretrain = False
    config.model_name = "bert-base-uncased"
    config.loss_type = "denoising"

    return config


config = create_config()

path_dir = "./checkpoints/bert-training-768-0.15-None-512-wiki_no_group+denoising/"

bert = BERTModel.load_from_checkpoint(
    config=config,
    checkpoint_path=f"{path_dir}/step_30000.ckpt"
)

# torch.save(bert.model.state_dict(), "../checkpoints/my_bert_pretrain.ckpt")


os.makedirs(f"{path_dir}/bert/", exist_ok=True)

bert.model.save_pretrained(f"{path_dir}/bert/")
