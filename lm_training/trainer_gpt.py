import sys

sys.path.append("/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/")

import torch
import ml_collections
from transformers import AutoTokenizer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning import seed_everything, Trainer

from lm_training.dataset_lightning import WikiDataModule
from lm_training.gpt_lightning import GPTModel


def create_config():
    config = ml_collections.ConfigDict()
    optim = config.optim = ml_collections.ConfigDict()
    optim.grad_clip_norm = 1.
    optim.linear_warmup = 5000
    optim.lr = 2e-4
    optim.min_lr = 2e-4
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
    training.eval_freq = 50_000
    training.batch_size = 512 // torch.cuda.device_count()

    data = config.data = ml_collections.ConfigDict()
    data.max_sequence_len = 128

    config.project_name = "lm-training"
    config.exp_name = "gpt2-training"
    config.seed = 0

    return config


def main():
    config = create_config()
    seed_everything(config.seed, workers=True)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token_id = 50256

    datamodule = WikiDataModule(
        tokenizer=tokenizer,
        config=config,
    )

    if torch.cuda.device_count() > 1:
        strategy = DDPStrategy(
            find_unused_parameters=False,
        )
    else:
        strategy = 'auto'

    trainer = Trainer(
        max_steps=config.training.training_iters,
        log_every_n_steps=50,
        reload_dataloaders_every_n_epochs=1,
        gradient_clip_algorithm="norm",
        gradient_clip_val=config.optim.grad_clip_norm,
        precision=config.optim.precision,
        strategy=strategy,
        enable_checkpointing=True,
        callbacks=[
            ModelCheckpoint(
                dirpath="./checkpoints",
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
    model = GPTModel(config)
    trainer.fit(model, datamodule=datamodule)


main()
