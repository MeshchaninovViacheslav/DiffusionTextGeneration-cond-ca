import ml_collections
from transformers import AutoConfig

def create_config():
    config = ml_collections.ConfigDict()

    encoder = config.encoder = ml_collections.ConfigDict()
    encoder.name = "bert-base-cased"
    
    compressor = config.compressor = ml_collections.ConfigDict()
    compressor.num_latents = 32
    compressor.embedding_dim = 768
    compressor.latent_dim = 768
    compressor.hidden_size = 768
    compressor.n_heads = 12
    compressor.n_layers = 3
    compressor.max_seq_len = 64
    compressor.dropout_p = 0.1
    compressor.ff_mult = 4
    compressor.qk_norm = True
    compressor.std_init = 0.02
    compressor.latent_normalize = True

    decoder = config.decoder = ml_collections.ConfigDict()
    decoder.num_latents = 64
    decoder.embedding_dim = 768
    decoder.latent_dim = 768
    decoder.hidden_size = 768
    decoder.n_heads = 12
    decoder.n_layers = 3
    decoder.max_seq_len = 32
    decoder.dropout_p = 0.1
    decoder.ff_mult = 4
    decoder.qk_norm = True
    decoder.std_init = 0.02
    decoder.latent_normalize = True

    projector = config.projector = ml_collections.ConfigDict()
    projector.hidden_size = 768
    projector.vocab_size = AutoConfig.from_pretrained(encoder.name).vocab_size

    optim = config.optim = ml_collections.ConfigDict()
    optim.lr = 1e-4
    optim.batch_size = 512
    optim.eval_freq = 5000
    optim.clip_norm = 1.
    optim.batch_size_per_gpu = 0
    optim.num_steps = 30_000
    optim.checkpoint_freq = 10_000

    data = config.data = ml_collections.ConfigDict()
    data.max_sequence_len = 64
    data.dataset_name = "wikipedia"
    data.dataset_path = f"./data/{data.dataset_name}"

    config.exp_name = f"recon-withpad-{compressor.num_latents}-{compressor.latent_dim}-{decoder.num_latents}-{decoder.latent_dim}"
    config.project_name = "compression_network-autoencoder"
    config.checkpoints_folder = "./autoencoder/checkpoints"
    config.save_path = f"{config.checkpoints_folder}/{config.exp_name}"
    config.is_conditional = False
    config.seed = 0

    return config



