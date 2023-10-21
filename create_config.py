import argparse
import ml_collections
from transformers import BertConfig

def parse_option(config):
    parser = argparse.ArgumentParser("MMTD")
    if config.ddp:
        parser.add_argument('--local_rank', type=int, required=True)
    args, unparsed = parser.parse_known_args()
    return args


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

    training = config.training = ml_collections.ConfigDict()
    training.training_iters = 500_000
    training.training_iters = training.training_iters
    training.checkpoint_freq = 100_000
    training.eval_freq = 50_000
    training.batch_size = 512  # * 8

    training.ode_sampling = False
    training.checkpoints_folder = './checkpoints/'
    config.checkpoints_prefix = ''

    loss = config.loss = ml_collections.ConfigDict()
    loss.ce_coef = 0.

    refresh = config.refresh = ml_collections.ConfigDict()
    refresh.true = False
    refresh.prefix = "./checkpoints/wikipedia--t5-bert-self_cond_500000_.pth"
    refresh.wand_id = "g5fb4af3"

    validation = config.validation = ml_collections.ConfigDict()
    validation.batch_size = 1024
    validation.validation_iters = int(10_000 / validation.batch_size)
    validation.num_gen_texts = 8192
    validation.p_uncond = 0.

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
    model.delta = 0.

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
    config.use_self_cond = True
    config.project_name = "dtg-exps-1.0"
    config.timesteps = "linear"

    return config