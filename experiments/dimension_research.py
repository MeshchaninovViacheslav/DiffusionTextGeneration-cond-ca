import sys

sys.path.append("/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/")

from data.dataset_clean_wiki import WikipediaCleanDataset
from model.enc_normalizer import EncNormalizer
from model.bert_encoder import BertEncoderModel

import torch
import wandb
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertConfig, BertTokenizerFast

from utils.util import set_seed, dict_to_cuda

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

set_seed(0)

train_dataset = next(WikipediaCleanDataset(
    split="train",
    tokenizer_bert=tokenizer,
    tokenizer_cond=tokenizer,
    tokenizer_gen=tokenizer,
    max_sequence_len=128,
    pos_begin=0.,
    pos_end=0.,
).get_data())

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

bert_cfg = "bert-base-uncased"
gen_enc_normalizer = EncNormalizer(
    enc_mean_path="/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/data/encodings-bert_base-wiki-mean.pt",
    enc_std_path="/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/data/encodings-bert_base-wiki-std.pt",
)
encoder_gen = BertEncoderModel.from_pretrained(
    bert_cfg, enc_normalizer=gen_enc_normalizer
).eval().cuda()


class NN(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(NN, self).__init__()

        self.hidden_size = 768
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(input_size, self.hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_size, output_size),
        )

        # self.ffn = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.ffn(x)


input_size = 768
output_size = 768

model = NN(input_size, output_size).cuda()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=2e-3,
)

wandb.init(
    project="dimension_research",
    name="masked_ffn_correlation",
    mode="online"
)

loss_epoch = []
model.train()

for step, X in tqdm(enumerate(train_loader), total=len(train_loader)):
    with torch.no_grad():
        X = dict_to_cuda(X)
        clean_X = encoder_gen(**{"input_ids": X["input_ids"], "attention_mask": X["input_mask"]})

    p = 0.15
    mask = (torch.rand_like(clean_X) > p) * 1.
    x = (clean_X * mask).cuda()

    target = clean_X.cuda()

    pred = model(x)
    recon_loss = torch.sum(torch.square(target - pred) * (1 - mask)) / torch.sum(1 - mask)
    loss = recon_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_epoch.append(loss.item())

    wandb.log({f'loss/train': loss.item()}, step=step)
    wandb.log({f'recon_loss/train': recon_loss.item()}, step=step)
    if step % 1000 == 0:
        torch.save(
            model.state_dict(),
            "masked_ffn_correlation_model.pth"
        )

torch.save(
    model.state_dict(),
    "masked_ffn_correlation_model.pth"
)
