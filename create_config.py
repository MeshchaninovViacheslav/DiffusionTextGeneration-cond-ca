import argparse
import ml_collections
from transformers import BertConfig

from model.config import create_decoder_config

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
    training.training_iters = 200_000
    training.checkpoint_freq = 50_000
    training.eval_freq = 10_000
    training.batch_size = 512
    training.ode_sampling = True
    training.checkpoints_folder = './checkpoints'
    training.teacher_folder = '/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/checkpoints/rocstory-bert-base-cased-sd-64-lr=0.0002-cls=27, sep=27, pad=0/last.pth'
    training.checkpoint_name = ""

    validation = config.validation = ml_collections.ConfigDict()
    validation.batch_size = 512

    dynamic = config.dynamic = ml_collections.ConfigDict()
    dynamic.solver = 'euler'
    dynamic.scheduler = "sd"
    dynamic.N = 50
    dynamic.beta_min = 0.1
    dynamic.beta_max = 20
    dynamic.ode_sampling = True
    dynamic.coef_d = 8

    generation = config.generation = ml_collections.ConfigDict()
    generation.batch_size = 512
    generation.t_min = 0.01
    generation.num_gen_texts = 1100
    generation.num_text_to_est = 1000
    generation.texts_path = "./generated_texts"
    generation.num_times_to_est = 5

    model = config.model = ml_collections.ConfigDict()
    model.ema_rate = 0.9999
    model.prediction = "x_0"
    model.loss = "L_x_0"
    model.encoder_name = "bert-base-cased"
    model.conditional_encoder_name = "t5-base"
    model.encoder_name_hash = model.encoder_name.replace("/", "-")
    model.conditional_encoder_name_hash = model.conditional_encoder_name.replace("/", "-")
    model.conditional_encoder_train = False
    
    data = config.data = ml_collections.ConfigDict()
    data.max_sequence_len = 64
    data.max_context_len = 0
    data.dataset_name = "rocstory"
    data.dataset_path = f"/home/vmeshchaninov/nlp_models/data/{data.dataset_name}"
    data.enc_gen_mean = f"{data.dataset_path}/statistics/encodings-{model.encoder_name_hash}-mean.pt"
    data.enc_gen_std = f"{data.dataset_path}/statistics/encodings-{model.encoder_name_hash}-std.pt"

    model.decoder_mode = "transformer"
    suffix = "cls=27, sep=27, pad=0"
    model.decoder_path = f"{data.dataset_path}/decoder-{model.encoder_name_hash}-{model.decoder_mode}-{data.max_sequence_len}-{suffix}.pth"

    config.seed = 0
    config.ddp = True
    config.use_self_cond = True

    config.project_name = "test"
    config.timesteps = "linear"
    config.is_conditional = False
    config.is_eval = False
    config.bert_config = create_se_config()
    config.bert_config.is_decoder = config.is_conditional
    config.decoder = create_decoder_config()
    training.checkpoints_prefix = f"{config.data.dataset_name}" + \
                                  f"-{config.model.encoder_name_hash}" + \
                                  f"-{config.dynamic.scheduler}" + \
                                  f"-{data.max_sequence_len}" + \
                                  f"-lr={optim.lr}-{suffix}"

    return config


def create_se_config():
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
    return bert_config