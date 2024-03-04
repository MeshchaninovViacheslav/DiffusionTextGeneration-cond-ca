import ml_collections
from transformers import AutoConfig

def create_decoder_config():
    decoder = ml_collections.ConfigDict()
    decoder.num_hidden_layers = 3
    decoder.base_config = AutoConfig.from_pretrained("bert-base-uncased")

    decoder.batch_size = 512
    decoder.lr = 1e-4
    decoder.eval_freq = 100
    decoder.num_epochs = 4
    decoder.std_aug = 0.2

    return decoder