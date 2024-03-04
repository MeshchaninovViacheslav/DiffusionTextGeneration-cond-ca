import ml_collections
from transformers import AutoConfig

def config():
    config = ml_collections.ConfigDict()
    encoder = config.encoder = ml_collections.ConfigDict()
    encoder.name = "bert-base-uncased"
    encoder.config = AutoConfig.from_pretrained(encoder.name)
    
    compressor = config.compressor = ml_collections.ConfigDict()
    compressor.num_latents = 32
    compressor.embedding_dim = 768
    compressor.hidden_size = 768
    compressor.n_heads = 12
    compressor.n_layers = 4
    compressor.dropout_p = 0.1

    decoder = config.decoder = ml_collections.ConfigDict()
    decoder.num_latents = 64
    decoder.embedding_dim = 768
    decoder.hidden_size = 768
    decoder.n_heads = 12
    decoder.n_layers = 4
    decoder.dropout_p = 0.1

    projector = config.projector = ml_collections.ConfigDict()
    projector.hidden_size = 768
    projector.vocab_size = encoder.config.vocab_size

