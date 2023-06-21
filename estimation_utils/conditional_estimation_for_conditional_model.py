import os
import json
import torch
import argparse
import torch.distributed as dist
import ml_collections
from datasets import disable_progress_bar
from transformers import BertConfig
import sys

sys.path.append("/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/")

from diffusion_holder import DiffusionRunner
from utils.util import set_seed, _BERT_SMALL
from estimation_utils.util import estimate_model, reduce_metrics, gather_texts
import diffusion_utils.schedulers as schedulers
from metrics import BloomMetricConditional, GPTMetric, BloomMetric, RobertaMetric

disable_progress_bar()


def create_config():
    config = ml_collections.ConfigDict()

    training = config.training = ml_collections.ConfigDict()
    training.ode_sampling = False
    training.checkpoints_folder = '../checkpoints'
    training.batch_size = 512
    config.checkpoints_prefix = None

    validation = config.validation = ml_collections.ConfigDict()
    validation.batch_size = 512

    sde = config.sde = ml_collections.ConfigDict()
    sde.typename = 'vp-sde'
    sde.solver = 'euler'
    sde.N = 200
    sde.beta_min = 0.1
    sde.beta_max = 20
    sde.ode_sampling = False
    sde.scheduler = schedulers.CosineSD(d=10)

    model = config.model = ml_collections.ConfigDict()
    model.ema_rate = 0.9999
    model.enc_type = "base"
    model.embeddings_type = "encodings"
    model.dif_enc_type = "base"
    model.downstream_task = "sst2"  # "qqp"
    model.dataset = "wikipedia"  # "glue"
    model.prediction = "x_0"
    model.loss = "L_x_0"
    model.decoder_path = "decoder-wikipedia-128.pth" #"decoder-roberta_base-wikipedia-128.pth" # "decoder-wikipedia-128.pth"  # "decoder-t5_base-wikipedia-128.pth"

    data = config.data = ml_collections.ConfigDict()
    data.max_sequence_len = 96
    data.enc_bert_mean = "/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/data/encodings-bert_base-wiki-mean.pt"
    data.enc_bert_std = "/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/data/encodings-bert_base-wiki-std.pt"
    data.enc_roberta_mean = "/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/data/encodings-roberta_base-wiki-mean.pt"
    data.enc_roberta_std = "/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/data/encodings-roberta_base-wiki-std.pt"
    data.enc_t5_mean = "/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/data/encodings-t5-wiki-mean.pth"
    data.enc_t5_std = "/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/data/encodings-t5-wiki-std.pth"

    config.finetuning = False
    config.seed = 0
    config.ddp = True
    config.bert_config = BertConfig.from_pretrained("bert-base-uncased")

    config.project_name = "bert-conditional-exps"

    return config


num_texts_ = 512
batch_size_ = 512

metrics_json = dict()
metrics_path = f"../metrics"
os.makedirs(metrics_path, exist_ok=True)
texts_path = "../generated_texts"
os.makedirs(texts_path, exist_ok=True)

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

metric_bloom_fn = BloomMetricConditional(device=f"cuda:{dist.get_rank()}")
metric_roberta_fn = RobertaMetric(device=f"cuda:{dist.get_rank()}")

model_names = [
    # "rocstory--encodings-prediction=x_0-loss=L_x_0-enc=base-bert=base-kl_cf=0.0-seq_len=32-clipgrad=1.0-lr=0.0002-min_lr=0.0002-lin_input=True-seed=0-wd=0.0-cond_launch-v1.0_100000_"
    #"wikipedia-sst2-prediction=x_0-loss=L_x_0-enc=base-bert=base-kl_cf=0.0-seq_len=96-clipgrad=1.0-lr=0.0002-min_lr=0.0002-lin_input=True-seed=0-wd=0.01-batch=512-ting-pretrain-t5-bert_encoder-wmask_200000_"
    # "wikipedia-sst2-encodings-prediction=x_0-loss=L_x_0-enc=base-bert=base-kl_cf=0.0-seq_len=96-clipgrad=1.0-lr=0.0002-min_lr=0.0002-lin_input=True-seed=0-wd=0.01-ting-pretrain_200000_"
    #"wikipedia-sst2-encodings-prediction=x_0-loss=L_x_0-enc=base-bert=base-kl_cf=0.0-seq_len=96-clipgrad=1.0-lr=0.0002-min_lr=0.0002-lin_input=True-seed=0-wd=0.01-ting-pretrain_200000_",
    #"wikipedia-sst2-encodings-prediction=x_0-loss=L_x_0-enc=base-bert=base-kl_cf=0.0-seq_len=96-clipgrad=1.0-lr=0.0002-min_lr=0.0002-lin_input=True-seed=0-wd=0.01-ting-pretrain_500000_",
    "wikipedia-sst2-prediction=x_0-loss=L_x_0-enc=base-bert=base-kl_cf=0.0-seq_len=96-clipgrad=1.0-lr=0.0002-min_lr=0.0002-lin_input=True-seed=0-wd=0.01-batch=512-t5-bert-womask_1000000_",
    #"wikipedia-sst2-prediction=x_0-loss=L_x_0-enc=base-bert=base-kl_cf=0.0-seq_len=96-clipgrad=1.0-lr=0.0002-min_lr=0.0002-lin_input=True-seed=0-wd=0.01-batch=512-t5-roberta-womask_500000_"
    ]

for model_name in model_names:
    config.checkpoints_prefix = model_name

    seed = config.seed + dist.get_rank()
    set_seed(seed)

    num_texts_ = int(num_texts_ / dist.get_world_size())
    diffusion_ = DiffusionRunner(config, latent_mode=config.model.embeddings_type, eval=True)
    seed = config.seed + dist.get_rank()
    set_seed(seed)
    metrics, joint_texts, cond_texts, gen_texts, gt_texts = estimate_model(
        diffusion_, num_texts_, batch_size_,
        metric_bloom_fn, metric_roberta_fn
    )
    metrics_json[model_name] = reduce_metrics(metrics)
    joint_texts = gather_texts(joint_texts)
    cond_texts = gather_texts(cond_texts)
    gen_texts = gather_texts(gen_texts)
    gt_texts = gather_texts(gt_texts)

    if dist.get_rank() == 0:
        print(model_name)
        print(f"Bloom metric: {metrics['Bloom metric']:0.5f}")
        print(f"Roberta metric: {metrics['Roberta metric']:0.5f}")
        print(len(joint_texts))
        prefix = f"num_texts={num_texts_}"
        metrics_file = os.path.join(metrics_path, f"{model_name}-{prefix}.json")
        with open(metrics_file, "w") as file:
            json.dump(metrics_json, file)

        text_list = []
        for i in range(len(cond_texts)):
            text_list.append(
                {
                    "CONDITION": cond_texts[i],
                    "GEN": gen_texts[i],
                    "GT": gt_texts[i]
                }
            )

        file_name = f"{texts_path}/{model_name}-{prefix}.json"
        json.dump(text_list, open(file_name, "w"), indent=4)

        print(file_name)

    del diffusion_
