import os
import torch
import ml_collections
from tqdm import tqdm
from datasets import disable_progress_bar
from transformers import BertConfig

import sys
sys.path.insert(0, "/home/vmeshchaninov/DiffusionTextGeneration")

from diffusion_holder import DiffusionRunner
from utils.util import set_seed, dict_to_cuda, make_mask_wo_SEP_CLS
from diffusion_utils import schedulers

disable_progress_bar()


def create_config():
    config = ml_collections.ConfigDict()
    optim = config.optim = ml_collections.ConfigDict()
    optim.grad_clip_norm = 1.0
    optim.linear_warmup = 5000
    optim.lr = 2e-4
    optim.weight_decay = 0

    training = config.training = ml_collections.ConfigDict()
    training.training_iters = 350_000
    training.checkpoint_freq = 50_000
    training.eval_freq = 50_000
    training.batch_size = 2048
    training.ode_sampling = False
    training.checkpoints_folder = './checkpoints'
    config.checkpoints_prefix = ""

    validation = config.validation = ml_collections.ConfigDict()
    validation.batch_size = 1028
    validation.validation_iters = int(10_000 / validation.batch_size)

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
    model.downstream_task = None  # "qqp"
    model.dataset = "rocstory"  # "glue"
    model.prediction = "x_0"
    model.loss = "L_x_0"

    data = config.data = ml_collections.ConfigDict()
    data.max_sequence_len = 64

    config.seed = 0
    config.ddp = False
    config.bert_config = BertConfig.from_pretrained("bert-base-uncased")

    return config

if __name__ == "__main__":
    config = create_config()

    seed = config.seed
    set_seed(seed)

    diffusion = DiffusionRunner(config, latent_mode=config.model.embeddings_type, eval=False)
    #diffusion.encoder = torch.nn.DataParallel(diffusion.encoder)

    os.environ['CONFIG_PATH'] = "/home/vmeshchaninov/DiffusionTextGeneration/data/config.json"
    diffusion.set_train_data_generator()

    sum_ = None
    sqr_sum_ = None
    num = 0

    T = tqdm(diffusion.train_loader)

    for i, X in enumerate(T):
        X = dict_to_cuda(X)
        with torch.no_grad():
            output = diffusion.sampler_emb_impl(X)
            mask = X["attention_mask"]
            output = output * mask[:, :, None]
            cur_sum = torch.sum(output, dim=[0, 1])
            cur_sqr_sum = torch.sum(output ** 2, dim=[0, 1])

            sum_ = cur_sum if sum_ is None else cur_sum + sum_
            sqr_sum_ = cur_sqr_sum if sqr_sum_ is None else cur_sqr_sum + sqr_sum_
            num += torch.sum(mask).item()

            mean = sum_[:3] / num
            std2 = sqr_sum_[:3] / num - mean ** 2
            T.set_description(f"mean: {mean}, std2: {std2}")
        if i == 100:
            break

    mean = sum_ / num
    std = torch.sqrt(sqr_sum_ / num - mean ** 2)

    torch.save(mean, f'./data/{config.model.embeddings_type}-bert_base-{config.model.dataset}-wost-mean.pt')
    torch.save(std, f'./data/{config.model.embeddings_type}-bert_base-{config.model.dataset}-wost-std.pt')