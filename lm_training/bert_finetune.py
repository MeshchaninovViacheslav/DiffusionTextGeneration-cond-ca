import sys

sys.path.append("/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/")

import torch
import ml_collections
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning import seed_everything, Trainer

from lm_training.dataset_lightning import WikiDataModule, SSTDataModule
from lm_training.bert_lightning import BERTModel

import os

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'


def create_config():
    config = ml_collections.ConfigDict()
    optim = config.optim = ml_collections.ConfigDict()
    optim.grad_clip_norm = 100.
    optim.linear_warmup = 5000
    optim.lr = 5e-5
    optim.min_lr = 5e-5
    optim.warmup_lr = 1e-8
    optim.weight_decay = 0.01
    optim.beta_1 = 0.9
    optim.beta_2 = 0.98
    optim.eps = 1e-6
    optim.precision = "16"

    training = config.training = ml_collections.ConfigDict()
    training.training_iters = 50_000
    training.training_iters = training.training_iters
    training.checkpoint_freq = 100_000
    training.eval_freq = 100
    training.batch_size = 128 // torch.cuda.device_count()

    data = config.data = ml_collections.ConfigDict()
    data.max_sequence_len = 128
    data.num_workers = 16
    data.bert_recon_dataset = True

    model = config.model = ml_collections.ConfigDict()
    model.mlm_probability = 0.15
    model.pad_to_multiple_of = 3

    bert_config = config.bert_config = ml_collections.ConfigDict()
    bert_config.hidden_size = 768

    config.project_name = "lm-training"
    config.exp_name = f"bert-finetune-{bert_config.hidden_size}-{model.mlm_probability}-{model.pad_to_multiple_of}"
    config.seed = 0
    config.hg_pretrain = False
    config.finetune = True

    return config


def main():
    config = create_config()
    seed_everything(config.seed, workers=True)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    datamodule = SSTDataModule(
        tokenizer=tokenizer,
        config=config,
        collate_fn=None,
    )

    if torch.cuda.device_count() > 1:
        strategy = DDPStrategy(
            find_unused_parameters=False,
        )
    else:
        strategy = 'auto'

    trainer = Trainer(
        max_steps=config.training.training_iters,
        log_every_n_steps=1,
        reload_dataloaders_every_n_epochs=1,
        gradient_clip_algorithm="norm",
        gradient_clip_val=config.optim.grad_clip_norm,
        precision=config.optim.precision,
        strategy=strategy,
        enable_checkpointing=True,
        callbacks=[
            ModelCheckpoint(
                dirpath=f"./checkpoints/{config.exp_name}/",
                filename='step_{step:d}',
                every_n_train_steps=config.training.checkpoint_freq,
                save_top_k=-1,
                auto_insert_metric_name=False,
                save_weights_only=False
            ),
            LearningRateMonitor(logging_interval='step'),
        ],
        logger=WandbLogger(
            project=config.project_name,
            name=config.exp_name,
            config=config,
        ),
        val_check_interval=config.training.eval_freq,
        check_val_every_n_epoch=None
    )
    model = BERTModel.load_from_checkpoint(
        checkpoint_path="./checkpoints/bert-training-768-0.15-3/step_400000.ckpt",
        config=config
    )

    trainer.fit(model, datamodule=datamodule)


main()
