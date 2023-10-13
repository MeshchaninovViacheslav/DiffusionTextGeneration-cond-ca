import sys

sys.path.append("/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/")

import torch
import ml_collections
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning import seed_everything, Trainer

from lm_training.dataset_lightning import WikiDataModule, WikiBooksDataModule
from lm_training.bert_lightning import BERTModel

import os

#os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'


def create_config():
    config = ml_collections.ConfigDict()
    optim = config.optim = ml_collections.ConfigDict()
    optim.grad_clip_norm = 5
    optim.linear_warmup = 5000
    optim.lr = 2e-4
    optim.min_lr = 2e-4
    optim.warmup_lr = 1e-8
    optim.weight_decay = 0.01
    optim.beta_1 = 0.9
    optim.beta_2 = 0.98
    optim.eps = 1e-6
    optim.precision = "bf16-mixed"

    training = config.training = ml_collections.ConfigDict()
    training.training_iters = 300_000
    training.checkpoint_freq = 20_000
    training.eval_freq = 20_000
    training.accumulate_grad_batches = 2
    training.device_count = torch.cuda.device_count()
    training.batch_size = 2048 // (training.device_count * training.accumulate_grad_batches)
    
    data = config.data = ml_collections.ConfigDict()
    data.max_sequence_len = 128
    data.num_workers = 15
    data.bert_recon_dataset = True

    model = config.model = ml_collections.ConfigDict()
    model.mlm_probability = 0.15
    model.pad_to_multiple_of = None

    bert_config = config.bert_config = ml_collections.ConfigDict()
    bert_config.hidden_size = 768
    bert_config.embedding_size = 768
    bert_config.encoder_initialization = None #"./checkpoints/bert-training-768-0.15-None-2048-wiki_no_group/bert/"
    bert_config.norm_output = True

    config.project_name = "lm-training"
    config.exp_name = f"bert-training-" \
                      f"{bert_config.hidden_size}-{bert_config.embedding_size}-{model.mlm_probability}-{model.pad_to_multiple_of}-" \
                      f"{training.batch_size * training.device_count * training.accumulate_grad_batches}-wiki_no_group-norm_output"
    config.seed = 0
    config.hg_pretrain = False
    config.finetune = False
    config.model_name = "bert-base-uncased"

    config.loss_type = "mlm"

    return config


def main():
    config = create_config()
    seed_everything(config.seed, workers=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=config.model.mlm_probability,
        pad_to_multiple_of=config.model.pad_to_multiple_of,
    )

    datamodule = WikiDataModule(
        tokenizer=tokenizer,
        config=config,
        collate_fn=data_collator,
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
        accumulate_grad_batches=config.training.accumulate_grad_batches,
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
        check_val_every_n_epoch=None,
    )
    # model = BERTModel.load_from_checkpoint(
    #     checkpoint_path="./checkpoints/bert-training-768-768-0.15-None-2048-wiki_no_group-norm_output/step_140000.ckpt",
    #     config=config
    # )
    model = BERTModel(config)
    trainer.fit(
        model,
        datamodule=datamodule,
        #ckpt_path="./checkpoints/bert-training-768-768-0.15-None-2048-wiki_no_group-norm_output/step_140000.ckpt",
    )


main()
