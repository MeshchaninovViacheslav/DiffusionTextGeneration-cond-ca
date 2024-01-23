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
    optim.linear_warmup = 1000
    optim.lr = 2e-4
    optim.min_lr = 2e-4
    optim.warmup_lr = 1e-8
    optim.weight_decay = 0.01
    optim.beta_1 = 0.9
    optim.beta_2 = 0.98
    optim.eps = 1e-6

    training = config.training = ml_collections.ConfigDict()
    training.training_iters = 100_000
    training.training_iters = training.training_iters
    training.checkpoint_freq = 50_000
    training.eval_freq = 10_000
    training.batch_size = 512  # * 8

    training.ode_sampling = False
    training.checkpoints_folder = './checkpoints/'
    config.checkpoints_prefix = ''

    loss = config.loss = ml_collections.ConfigDict()
    loss.ce_coef = 0.

    refresh = config.refresh = ml_collections.ConfigDict()
    refresh.true = False
    refresh.prefix = "./checkpoints/rocstory--t5-bert-uncond-64_100000_.pth"
    refresh.wand_id = "g5fb4af3"

    validation = config.validation = ml_collections.ConfigDict()
    validation.batch_size = 250
    validation.validation_iters = int(10_000 / validation.batch_size)
    validation.num_gen_texts = 1000
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
    model.dataset = "rocstory"
    model.prediction = "x_0"
    model.loss = "L_x_0"
    model.encoder_name = "bert-base-cased"
    encoder_name = "bert-base-cased"
    model.decoder_path = f"/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/checkpoints/decoder-{encoder_name}-transformer-spt.pth"
    model.delta = 0.
    
    data = config.data = ml_collections.ConfigDict()
    data.max_sequence_len = 80
    data.enc_bert_mean = f"/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/data/rocstory/encodings-{encoder_name}-mean.pt"
    data.enc_bert_std = f"/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/data/rocstory/encodings-{encoder_name}-std.pt"

    config.finetuning = False
    config.seed = 0
    config.ddp = True
    config.use_self_cond = True
    config.project_name = "article-rocstory_exps"
    config.timesteps = "linear"
    config.is_conditional = False
    config.bert_config = bert_config
    config.bert_config.is_decoder = config.is_conditional

    return config

bert_config = BertConfig(**{
    "hidden_size": 768,
    "hidden_act": "gelu",
    "initializer_range": 0.02,
    "vocab_size": 30522,
    "hidden_dropout_prob": 0.1,
    "num_attention_heads": 12,
    "type_vocab_size": 2,
    "max_position_embeddings": 512,
    "num_hidden_layers": 12,
    "intermediate_size": 3072,
    "attention_probs_dropout_prob": 0.1,
    "layer_norm_eps": 1e-12,
    "model_type": "bert",
    "pad_token_id": 0,
    "position_embedding_type": "absolute",
    "transformers_version": "4.6.0.dev0",
    "is_decoder": True,
})