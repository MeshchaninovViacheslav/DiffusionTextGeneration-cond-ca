import os
import json
import torch
import argparse
import torch.distributed as dist
import ml_collections
from datasets import disable_progress_bar
from transformers import BertConfig
import sys
from tqdm import tqdm

sys.path.append("/home/vmeshchaninov/DiffusionTextGeneration/")

from diffusion_holder import DiffusionRunner
from utils.util import set_seed, _BERT_SMALL
from estimation_utils.util import reduce_metrics, gather_texts, compute_metric, clear_text
from diffusion_utils import schedulers

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
    config.seed = 1
    config.bert_config = BertConfig.from_pretrained("bert-base-uncased")
    return config


def loop_generation(diffusion, batch_size):
    shape = (
        batch_size,
        diffusion.config.data.max_sequence_len,
        diffusion.encoder.config.hidden_size
    )
    eps_t = 1 / diffusion.diff_eq_solver.dynamic.N
    timesteps = torch.linspace(diffusion.dynamic.T - 20 * eps_t, eps_t, diffusion.dynamic.N, device=diffusion.device)
    #timesteps = torch.linspace(diffusion.sde.T, eps_t, diffusion.sde.N, device=diffusion.device)

    t_map = []
    for t in timesteps:

        #alpha, _ = schedulers.cosine(t, diffusion.sde.beta_0, diffusion.sde.beta_1)
        alpha, _ = schedulers.linear(t)
        t_map.append(schedulers.cosine_rev(alpha.item(), diffusion.dynamic.beta_0, diffusion.dynamic.beta_1))

        #alpha, _ = schedulers.linear(t)
        #t_map.append(schedulers.linear_rev(alpha))


    with torch.no_grad():
        x = diffusion.dynamic.prior_sampling(shape).to(diffusion.device)
        for i in tqdm(range(diffusion.dynamic.N)):
            t_scale = t_map[i]# + 0.01
            t = timesteps[i]
            vec_t_scale = torch.ones(shape[0], device=diffusion.device) * t_scale
            vec_t = torch.ones(shape[0], device=diffusion.device) * t
            output = diffusion.diff_eq_solver.step(model=diffusion.score_estimator, x_t=x, t=vec_t, t_scale=vec_t_scale)
            x, x_mean = output["x"], output["x_mean"]
        pred_embeddings = x_mean

    output = diffusion.pred_logits(pred_embeddings)
    tokens = output.argmax(dim=-1)
    text = diffusion.tokenizer.batch_decode(tokens)
    return text


def generate_text(diffusion, num_texts, batch_size):
    generated_texts = []
    while len(generated_texts) < num_texts:
        n = int(min(batch_size, num_texts - len(generated_texts)))
        text = loop_generation(diffusion, batch_size=n)
        text = clear_text(text)
        generated_texts += text
    return generated_texts


def estimate_model(diffusion, num_texts, batch_size, metric_bloom_fn, metric_gpt_fn):
    texts = generate_text(diffusion, num_texts, batch_size)
    metric_bloom = compute_metric(metric_bloom_fn, texts)
    # print(f"Bloom metric: {metric_bloom:0.5f}")
    metric_gpt = compute_metric(metric_gpt_fn, texts)
    # print(f"GPT2 metric: {metric_gpt:0.5f}")
    return {"Bloom metric": metric_bloom, "GPT2 metric": metric_gpt}, texts


num_texts_ = 512 * 4 # 512 * 6# * 6
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

suffix = "cosine"
model_names = [
    f"rocstory--encodings-prediction=x_0-loss=L_x_0-enc=base-bert=base-kl_cf=0.0-seq_len=32-clipgrad=1.0-lr=0.0002-min_lr=0.0002-lin_input=False-seed=0-wd=0.0-sch={suffix}_800000_",
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
        prefix = f"seq-len={config.data.max_sequence_len}-ode-{config.training.ode_sampling}"
        metrics_file = os.path.join(metrics_path, f"{model_name}-{prefix}.json")
        with open(metrics_file, "w") as file:
            json.dump(metrics_json, file)
        with open(f"{texts_path}/{model_name}-{prefix}.txt", "w") as file:
            for text in texts:
                print(text, file=file)

    del diffusion_
