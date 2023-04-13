import os
import json
import torch
import argparse
import torch.distributed as dist
import ml_collections
from datasets import disable_progress_bar
from transformers import BertConfig
import sys

sys.path.append("/home/vmeshchaninov/DiffusionTextGeneration/")

from diffusion_holder import DiffusionRunner
from utils.util import set_seed, _BERT_SMALL
from estimation_utils.util import estimate_model, reduce_metrics, gather_texts
import diffusion_utils.schedulers as schedulers

disable_progress_bar()

from metrics import BloomMetric, GPTMetric


def parse_option(config):
    parser = argparse.ArgumentParser("MMTD")
    if config.ddp:
        parser.add_argument('--local_rank', type=int, required=True)
    args, unparsed = parser.parse_known_args()
    return args


def create_config():
    config = ml_collections.ConfigDict()

    training = config.training = ml_collections.ConfigDict()
    training.ode_sampling = False
    training.checkpoints_folder = '../checkpoints'
    config.checkpoints_prefix = None

    sde = config.sde = ml_collections.ConfigDict()
    sde.typename = 'vp-sde'
    sde.solver = 'euler'
    sde.N = 1000
    sde.beta_min = 0.1
    sde.beta_max = 20
    sde.ode_sampling = False
    sde.scheduler = schedulers.Cosine(sde.beta_min, sde.beta_max)

    model = config.model = ml_collections.ConfigDict()
    model.ema_rate = 0.9999
    model.enc_type = "base"
    model.embeddings_type = "encodings"
    model.dif_enc_type = "base"
    model.prediction = "x_0"
    model.dataset = "rocstory"

    data = config.data = ml_collections.ConfigDict()
    data.config_path = "/home/vmeshchaninov/DiffusionTextGeneration/data/config.json"
    data.max_sequence_len = 32

    config.lin_input = False
    config.device = 'cuda:0'
    config.ddp = False
    config.seed = 0
    config.bert_config = BertConfig.from_pretrained("bert-base-uncased")
    return config


num_texts_ =  512 * 4
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
torch.cuda.set_device(rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
torch.distributed.barrier()

metric_bloom_fn = BloomMetric(device=f"cuda:{dist.get_rank()}")
metric_gpt_fn = GPTMetric(device=f"cuda:{dist.get_rank()}")

model_names = [
    "rocstory--encodings-prediction=x_0-loss=L_x_0-enc=base-bert=base-kl_cf=0.0-seq_len=32-clipgrad=1.0-lr=0.0002-min_lr=0.0002-lin_input=True-seed=0-wd=0.0-cond_launch-v1.0_100000_"
]

for model_name in model_names:
    config.checkpoints_prefix = model_name
    if "base" in model_name:
        config.bert_config = BertConfig.from_pretrained("bert-base-uncased")
    else:
        config.bert_config = BertConfig(**_BERT_SMALL)

    seed = config.seed + dist.get_rank()
    set_seed(seed)

    num_texts_ = int(num_texts_ / dist.get_world_size())
    diffusion_ = DiffusionRunner(config, latent_mode=config.model.embeddings_type, eval=True)
    seed = config.seed + dist.get_rank()
    set_seed(seed)
    metrics, texts = estimate_model(diffusion_, num_texts_, batch_size_, metric_bloom_fn, metric_gpt_fn, type_="uncond")
    metrics_json[model_name] = reduce_metrics(metrics)
    texts = gather_texts(texts)

    if dist.get_rank() == 0:
        print(model_name)
        print(f"Bloom metric: {metrics['Bloom metric']:0.5f}")
        print(f"GPT2 metric: {metrics['GPT2 metric']:0.5f}")
        print(len(texts))
        prefix = f"seq-len={config.data.max_sequence_len}-ode-{config.training.ode_sampling}-cosine"
        metrics_file = os.path.join(metrics_path, f"{model_name}-{prefix}.json")
        with open(metrics_file, "w") as file:
            json.dump(metrics_json, file)
        file_name = f"{texts_path}/{model_name}-{prefix}.txt"
        with open(file_name, "w") as file:
            for text in texts:
                print(text, file=file)
        print(file_name)

    del diffusion_
