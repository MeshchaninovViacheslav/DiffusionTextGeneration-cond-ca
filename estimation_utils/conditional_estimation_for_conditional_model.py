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
from estimation_utils.diversity_metrics import NGramStats

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

    config.project_name = "test"
    config.classifier_guidance_scale = 2.
    config.use_self_cond = True
    config.timesteps = "linear"
    config.model.delta = 0.05

    return config


num_texts_ = 8196
batch_size_ = 1024

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
    # "wikipedia-sst2-prediction=x_0-loss=L_x_0-enc=base-bert=base-kl_cf=0.0-seq_len=96-clipgrad=1.0-lr=0.0002-min_lr=0.0002-lin_input=True-seed=0-wd=0.01-batch=512-ting-pretrain-t5-bert_encoder-wmask_200000_"
    # "wikipedia-sst2-encodings-prediction=x_0-loss=L_x_0-enc=base-bert=base-kl_cf=0.0-seq_len=96-clipgrad=1.0-lr=0.0002-min_lr=0.0002-lin_input=True-seed=0-wd=0.01-ting-pretrain_200000_"
    # "wikipedia-sst2-encodings-prediction=x_0-loss=L_x_0-enc=base-bert=base-kl_cf=0.0-seq_len=96-clipgrad=1.0-lr=0.0002-min_lr=0.0002-lin_input=True-seed=0-wd=0.01-ting-pretrain_200000_",
    # "wikipedia-sst2-encodings-prediction=x_0-loss=L_x_0-enc=base-bert=base-kl_cf=0.0-seq_len=96-clipgrad=1.0-lr=0.0002-min_lr=0.0002-lin_input=True-seed=0-wd=0.01-ting-pretrain_500000_",
    #"wikipedia-sst2-prediction=x_0-loss=L_x_0-enc=base-bert=base-kl_cf=0.0-seq_len=96-clipgrad=1.0-lr=0.0002-min_lr=0.0002-lin_input=True-seed=0-wd=0.01-batch=512-t5-bert-womask_1000000_",
    # "wikipedia-sst2-prediction=x_0-loss=L_x_0-enc=base-bert=base-kl_cf=0.0-seq_len=96-clipgrad=1.0-lr=0.0002-min_lr=0.0002-lin_input=True-seed=0-wd=0.01-batch=512-t5-roberta-womask_500000_"
    #"wikipedia-sst2-prediction=x_0-loss=L_x_0-seq_len=96-clipgrad=1.0-lr=0.0002-min_lr=0.0002-lin_input=True-seed=0-wd=0.01-batch=512-SD=10-bert-bert-womask_1000000_",
    #"wikipedia-sst2-prediction=x_0-loss=L_x_0-seq_len=96-clipgrad=1.0-lr=0.0002-min_lr=0.0002-lin_input=True-seed=0-wd=0.01-batch=512-SD=10-bert-bert-womask_500000_",
    #"wikipedia-clean--prediction=x_0-loss=L_x_0-seq_len=96-cond_seg=[0.00, 0.67]-clipgrad=1.0-lr=0.0002-min_lr=0.0002-lin_input=True-seed=0-wd=0.01-batch=512-SD=10-bert-bert-womask_900000_",
    # "wikipedia-clean--prediction=x_0-loss=L_x_0-seq_len=96-cond_seg=[0.00, 0.67]-clipgrad=1.0-lr=0.0002-min_lr=0.0002-lin_input=True-seed=0-wd=0.01-batch=512-SD=10-t5-bert-womask_800000_",
    # "wikipedia-clean--prediction=x_0-loss=L_x_0-seq_len=96-cond_seg=[0.00, 0.67]-clipgrad=1.0-lr=0.0002-min_lr=0.0002-lin_input=True-seed=0-wd=0.01-batch=512-SD=10-t5-bert-womask_500000_",
    # "wikipedia-clean--prediction=x_0-loss=L_x_0-seq_len=96-cond_seg=[0.00, 0.67]-clipgrad=1.0-lr=0.0002-min_lr=0.0002-lin_input=True-seed=0-wd=0.01-batch=512-SD=10-t5-bert-womask_200000_"
    #"wikipedia-clean--prediction=x_0-loss=L_x_0-seq_len=96-cond_seg=[0.00, 0.67]-clipgrad=1.0-lr=0.0002-min_lr=0.0002-lin_input=True-seed=0-wd=0.01-batch=512-SD=10-t5-bert-womask_800000_"
    #"wikipedia-clean--prediction=x_0-loss=L_x_0-seq_len=96-cond_seg=[0.00, 0.67]-clipgrad=1.0-lr=0.0002-min_lr=0.0002-lin_input=True-seed=0-wd=0.01-batch=512-SD=10-t5-bert_10000_"
    # "wikipedia-sst2-prediction=x_0-loss=L_x_0-seq_len=96-clipgrad=1.0-lr=0.0002-min_lr=0.0002-lin_input=True-seed=0-wd=0.01-batch=512-SD=10-bert-bert-womask_900000_",
    # "wikipedia-clean--prediction=x_0-loss=L_x_0-seq_len=96-cond_seg=[0.00, 0.67]-clipgrad=1.0-lr=0.0002-min_lr=0.0002-lin_input=True-seed=0-wd=0.01-batch=512-SD=10-bert-bert-womask_900000_",
    #"wikipedia--prediction=x_0-loss=L_x_0-seq_len=96-cond_seg=[0.00, 0.67]-clipgrad=10.0-lr=0.0002-min_lr=0.0002-seed=0-wd=0.01-batch=512-SD=10-t5-mybert_1000000_"
    #"wikipedia--prediction=x_0-loss=L_x_0-seq_len=96-cond_seg=[0.00, 0.67]-clipgrad=10.0-lr=0.0002-min_lr=0.0002-seed=0-wd=0.01-batch=512-SD=10-t5-bert_800000_"
    "wikipedia--t5-bert-self_cond_last_"
    #"wikipedia--t5-bert-initial_last_"
]

for model_name in model_names:
    config.checkpoints_prefix = model_name

    seed = config.seed + dist.get_rank()
    set_seed(seed)

    num_texts = int(num_texts_ / dist.get_world_size())
    diffusion_ = DiffusionRunner(config, latent_mode=config.model.embeddings_type, eval=True)

    seed = config.seed + dist.get_rank()
    set_seed(seed)

    metrics, metrics_list, joint_texts, cond_texts, gen_texts, gt_texts = estimate_model(
        diffusion_, num_texts, batch_size_,
        metric_bloom_fn, metric_roberta_fn
    )
    metrics_json[model_name] = reduce_metrics(metrics)
    joint_texts = gather_texts(joint_texts)
    cond_texts = gather_texts(cond_texts)
    gen_texts = gather_texts(gen_texts)
    gt_texts = gather_texts(gt_texts)
    for key in metrics_list:
        metrics_list[key] = gather_texts(metrics_list[key])

    if dist.get_rank() == 0:
        print(model_name)
        print(f"Bloom metric: {metrics['Bloom metric']:0.5f}")
        print(f"Roberta metric: {metrics['Roberta metric']:0.5f}")
        print(len(joint_texts))
        prefix = f"num_texts={num_texts_}-scale={config.classifier_guidance_scale:0.1f}-delta=0.05"
        metrics_file = os.path.join(metrics_path, f"{model_name}-{prefix}.json")
        with open(metrics_file, "w") as file:
            json.dump(metrics_json, file)

        text_list = []
        for i in range(len(cond_texts)):
            text_list.append(
                {
                    "CONDITION": cond_texts[i],
                    "GEN": gen_texts[i],
                    "GT": gt_texts[i],
                    "Bloom metric": metrics_list["Bloom metric"][i],
                }
            )

        file_name = f"{texts_path}/{model_name}-{prefix}.json"
        json.dump(text_list, open(file_name, "w"), indent=4)

        print(file_name)

        metric_div = NGramStats()
        metric_div.compute(gen_texts)
        print(metric_div)

    del diffusion_
