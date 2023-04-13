import sys
import argparse
import ml_collections
from datasets import disable_progress_bar
from transformers import BertConfig

sys.path.append("/home/vmeshchaninov/DiffusionTextGeneration/")

from diffusion_holder import DiffusionRunner
from utils.util import set_seed, _BERT_SMALL, _BERT_BASE

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
    training.batch_size = 128
    training.ode_sampling = False
    training.checkpoints_folder = './checkpoints'
    config.checkpoints_prefix = ""

    validation = config.validation = ml_collections.ConfigDict()
    validation.batch_size = 1028
    validation.validation_iters = int(10_000 / validation.batch_size)

    sde = config.sde = ml_collections.ConfigDict()
    sde.typename = 'vp-sde'
    sde.solver = 'euler'
    sde.N = 2000
    sde.beta_min = 0.1
    sde.beta_max = 20
    sde.ode_sampling = False

    model = config.model = ml_collections.ConfigDict()
    model.ema_rate = 0.9999
    model.enc_type = "base"
    model.embeddings_type = "encodings"
    model.dif_enc_type = "base"

    data = config.data = ml_collections.ConfigDict()
    data.config_path = "/home/vmeshchaninov/DiffusionTextGeneration/data/config.json"
    data.max_sequence_len = 64

    config.device = 'cuda:0'
    config.ddp = False
    config.seed = 0
    config.bert_config = None
    return config

def clear_text(text):
    data = []
    for l in text:
        s = l.replace("..", "")
        s = s.replace("[SEP]", "").replace("[CLS]", "")
        data.append(s)
    return data


if __name__ == '__main__':
    model_name = "glue-encodings-enc=base-bert=base-kl_cf=0.1seq_len=64-6gpu_1000000_"
    print(model_name)
    config = create_config()
    config.checkpoints_prefix = model_name
    if "base" in model_name:
        config.bert_config = BertConfig(**_BERT_BASE) #.from_pretrained("bert-base-uncased")
    else:
        config.bert_config = BertConfig(**_BERT_SMALL)

    set_seed(config.seed)

    diffusion = DiffusionRunner(config, latent_mode="encodings", eval=True)
    text = diffusion.generate_text(batch_size=512)[0]

    for t in clear_text(text):
        print(t)
