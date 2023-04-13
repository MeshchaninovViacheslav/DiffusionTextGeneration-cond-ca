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
    model.dataset = "c4"

    data = config.data = ml_collections.ConfigDict()
    data.config_path = "/home/vmeshchaninov/DiffusionTextGeneration/data/config.json"
    data.max_sequence_len = 32

    config.lin_input = False
    config.device = 'cuda:0'
    config.ddp = False
    config.seed = 0
    config.bert_config = BertConfig.from_pretrained("bert-base-uncased")
    return config


num_texts_ = 512 * 4
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
    # "encodings-x0-bs=512-wopad-mask-small-roc-lr=2e-4_1000000",
    # "encodings-x0-bs=512-wopad-mask-base_500000",
    # "encodings-x0-bs=512-wopad-mask-base-roc_850000",
    # "encodings-x0-bs=512-wopad-mask-base-roc_500000_ema",
    # "enc_bert_x0_bs=512_lr=2e-4_350000",
    # "encodings-x0-bs=512-wopad-mask-base-roc-lr=2e-4-ce_200000",
    # "embeddings-x0-bs=512-wopad-mask-base-roc-lr=2e-4_200000",
    # "encodings-x0-bs=512-wopad-mask-base-roc-lr=2e-4-ce_1000000",
    # "encodings-x0-bs=512-wopad-mask-base-roc-lr=2e-4-ce_400000",
    # "encodings-x0-bs=512-wopad-mask-base-roc-lr=2e-4-ce_200000",
    # "encodings-x0-bs=512-wopad-mask-base-roc-lr=sch-ce_500000_",
    # "encodings-x0-bs=512-wopad-mask-base-roc-lr=sch-ce-seq=64_400000",
    # "encodings-enc=l-bert=base_500000_",
    # "encodings-enc=base-untrained-bert=base_500000_",
    # "encodings-enc=large-untrained-bert=base_500000_",
    # "mid_embeddings-enc=base-bert=base-kl_cf=0.0_100000_",
    # "encodings-enc=base-bert=base-kl_cf=0.001_500000_",
    # "encodings-enc=base-bert=base-kl_cf=0.0_500000_",
    # "mid_embeddings-enc=base-bert=base-kl_cf=0.0_500000_",
    # "wikipedia--encodings-enc=base-bert=base-kl_cf=0.1seq_len=64-v1_900000_"
    # "wikipedia--encodings-prediction=x_0_x_t-loss=L_x_0-enc=base-bert=base-kl_cf=0.0-seq_len=64-clipgrad=0.1-lr=5e-05-min_lr=1e-05-_1200000_"
    # "wikipedia--encodings-prediction=x_0_a_x_t-loss=L_x_0-enc=base-bert=base-kl_cf=0.0-seq_len=64-clipgrad=0.1-lr=5e-05-min_lr=1e-05-norm_repaired_1200000_"
    # "wikipedia--encodings-prediction=eps_a_x_t-loss=L_eps-enc=base-bert=base-kl_cf=0.0-seq_len=64-clipgrad=0.1-lr=5e-05-min_lr=1e-05-norm_repaired_1200000_"
    # "rocstory--encodings-prediction=x_0_a_x_t-loss=L_x_0-enc=base-bert=base-kl_cf=0.0-seq_len=64-clipgrad=0.1-lr=5e-05-min_lr=1e-05-norm_repaired_900000_"
    # "rocstory--encodings-prediction=eps_a_x_t-loss=L_eps-enc=base-bert=base-kl_cf=0.0-seq_len=64-clipgrad=0.1-lr=5e-05-min_lr=1e-05-norm_repaired_1800000_"
    # "rocstory--encodings-prediction=x_0_a_x_t-loss=L_x_0-enc=base-bert=base-kl_cf=0.0-seq_len=32-clipgrad=0.1-lr=5e-05-min_lr=1e-05-norm_repaired_1400000_"
    # "rocstory--encodings-prediction=x_0_a_x_t-loss=L_x_0-enc=base-bert=base-kl_cf=0.0-seq_len=32-clipgrad=0.1-lr=5e-05-min_lr=1e-05-norm_repaired_1000000_"
    # "rocstory--encodings-prediction=x_0-loss=L_x_0-enc=base-bert=base-kl_cf=0.0-seq_len=32-clipgrad=1-lr=0.0002-min_lr=0.0002-norm_repaired_800000_",
    # "rocstory--encodings-prediction=x_0-loss=L_x_0-enc=base-bert=base-kl_cf=0.0-seq_len=32-clipgrad=1-lr=0.0001-min_lr=1e-05-norm_repaired_800000_",
    # "rocstory--encodings-prediction=x_0-loss=L_x_0-enc=base-bert=base-kl_cf=0.0-seq_len=32-clipgrad=0.2-lr=0.0002-min_lr=0.0002-new_net_v1.3_time_norm_1000000_"
    # "rocstory--encodings-prediction=x_0-loss=L_x_0-enc=base-bert=base-kl_cf=0.0-seq_len=32-clipgrad=0.2-lr=0.0002-min_lr=0.0002-new_net_v1.5_time_norm_x_t_skip_overlin_1000000_"
    # "rocstory--encodings-prediction=x_0-loss=L_x_0-enc=base-bert=base-kl_cf=0.1-seq_len=32-clipgrad=0.2-lr=0.0002-min_lr=0.0002-new_net_v1.3_time-norm-finetuning_1050000_"
    # "rocstory--encodings-prediction=x_0-loss=L_x_0-enc=base-bert=base-kl_cf=0.0-seq_len=32-clipgrad=0.2-lr=0.0002-min_lr=0.0002-new_net_v1.3.4-woinputproj_1000000_"
    # "rocstory--encodings-prediction=x_0-loss=L_x_0-enc=base-bert=base-kl_cf=0.0-seq_len=32-clipgrad=0.2-lr=0.0002-min_lr=0.0002-new_net_v1.3.4-woinputproj-wooutputproj_1000000_"
    # "rocstory--embeddings-prediction=x_0-loss=L_x_0-enc=base-bert=base-kl_cf=0.0-seq_len=32-clipgrad=0.2-lr=0.0002-min_lr=0.0002-new_net_v1.3.4-woinputproj_1000000_"
    #"wikipedia--encodings-prediction=x_0-loss=L_x_0-enc=base-bert=base-kl_cf=0.0-seq_len=64-clipgrad=0.2-lr=0.0002-min_lr=0.0002-new_net_v1.3.4-woinputproj-wooutputproj_1000000_"
    # "rocstory--encodings-prediction=x_0-loss=L_x_0-enc=base-bert=base-kl_cf=0.0-seq_len=32-clipgrad=0.2-lr=0.0002-min_lr=0.0002-lin_input=True-seed=1-new_net_v1.3_time_norm_1000000_"
    # "wikipedia--encodings-prediction=eps-loss=L_eps-enc=base-bert=base-kl_cf=0.0-seq_len=32-clipgrad=1.0-lr=0.0002-min_lr=0.0002-lin_input=False-seed=1-wd=0.0-gradient_exploding-time_t_200000_"
    # "rocstory--encodings-prediction=x_0-loss=L_x_0-enc=base-bert=base-kl_cf=0.0-seq_len=32-clipgrad=0.2-lr=0.0002-min_lr=0.0002-new_net_v1.3_time_norm_1000000_"
    # "rocstory--encodings-prediction=x_0-loss=L_x_0-enc=base-bert=base-kl_cf=0.0-seq_len=32-clipgrad=1.0-lr=0.0002-min_lr=0.0002-lin_input=False-seed=0-wd=0.0-sch=cosine_1000000_",
    #"rocstory--encodings-prediction=x_0-loss=L_x_0-enc=base-bert=base-kl_cf=0.0-seq_len=32-clipgrad=1.0-lr=0.0002-min_lr=0.0002-lin_input=False-seed=0-wd=0.0-sch=quadratic_800000_",
    #"rocstory--encodings-prediction=x_0-loss=L_x_0-enc=base-bert=base-kl_cf=0.0-seq_len=32-clipgrad=1.0-lr=0.0002-min_lr=0.0002-lin_input=True-seed=0-wd=0.0-sch=Exp_a=10_200000_"
    #"c4--encodings-prediction=x_0-loss=L_x_0-enc=base-bert=base-kl_cf=0.0-seq_len=32-clipgrad=1.0-lr=0.0002-min_lr=0.0002-lin_input=True-seed=0-wd=0.0-_200000_"
    "c4--encodings-prediction=x_0-loss=L_x_0-enc=base-bert=base-kl_cf=0.0-seq_len=32-clipgrad=1.0-lr=0.0002-min_lr=0.0002-lin_input=True-seed=0-wd=0.0-_400000_"
    #"rocstory--encodings-prediction=x_0-loss=L_x_0-enc=base-bert=base-kl_cf=0.0-seq_len=64-clipgrad=1.0-lr=0.0002-min_lr=0.0002-lin_input=True-seed=0-wd=0.0-_400000_"
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
    metrics, texts = estimate_model(diffusion_, num_texts_, batch_size_, metric_bloom_fn, metric_gpt_fn)
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
