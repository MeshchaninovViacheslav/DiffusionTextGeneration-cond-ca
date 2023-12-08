import sys
import torch
from transformers import AutoTokenizer
from functools import partial

sys.path.insert(0, "/home/vmeshchaninov/DiffusionTextGeneration-cond-ca")

from data.dataset import RocStoryDatasetDDP
from model.bert_encoder import BertEncoderModel
from model.enc_normalizer import EncNormalizer

enc_bert_mean = "/home/vmeshchaninov/nlp_models/data/rocstories/grouped_data/encodings-grouped-rocstory-mean.pt"
enc_bert_std = "/home/vmeshchaninov/nlp_models/data/rocstories/grouped_data/encodings-grouped-rocstory-std.pt"


bert_cfg = "bert-base-uncased"
tokenizer_bert = AutoTokenizer.from_pretrained(bert_cfg)

cfg_cond = "bert-base-uncased"
tokenizer_cond = AutoTokenizer.from_pretrained(cfg_cond)
encoder_cond = torch.nn.DataParallel(
    BertEncoderModel.from_pretrained(
        cfg_cond, 
        enc_normalizer=EncNormalizer(
            enc_mean_path=enc_bert_mean,
            enc_std_path=enc_bert_std,
        )
    )
).eval().cuda()

cfg_gen = "bert-base-uncased"
tokenizer_gen = AutoTokenizer.from_pretrained(cfg_gen)
encoder_gen = torch.nn.DataParallel(
    BertEncoderModel.from_pretrained(
        cfg_gen, 
        enc_normalizer=EncNormalizer(
            enc_mean_path=enc_bert_mean,
            enc_std_path=enc_bert_std,
        )
    )
).eval().cuda()

max_sequence_len = 50
batch_size = 2048

train_dataset = next(RocStoryDatasetDDP(
    split="train",
    tokenizer_bert=tokenizer_bert,
    tokenizer_cond=tokenizer_cond,
    tokenizer_gen=tokenizer_gen,
    max_sequence_len=max_sequence_len,
    pos_begin=1.,
    pos_end=1.,
    is_conditional=True,
).get_data())

test_dataset = next(RocStoryDatasetDDP(
    split="valid",
    tokenizer_bert=tokenizer_bert,
    tokenizer_cond=tokenizer_cond,
    tokenizer_gen=tokenizer_gen,
    max_sequence_len=max_sequence_len,
    pos_begin=1.,
    pos_end=1.,
    is_conditional=True,
).get_data())


def get_encodings(X, encoder_gen, encoder_cond):
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        X["gen_x"] = encoder_gen(**{
            "input_ids": torch.tensor(X["input_ids"]).cuda(), 
            "attention_mask": torch.tensor(X["input_mask"]).cuda()
            }).detach().cpu()
        X["cond_x"] = encoder_cond(**{
            "input_ids": torch.tensor(X["cond_ids"]).cuda(), 
            "attention_mask": torch.tensor(X["cond_mask"]).cuda()
            }).detach().cpu()
    return X

train_dataset = train_dataset.map(
    partial(get_encodings, encoder_gen=encoder_gen, encoder_cond=encoder_cond),
    batched=True,
    desc="Encode running",
)
print(train_dataset.column_names)

test_dataset = test_dataset.map(
    partial(get_encodings, encoder_gen=encoder_gen, encoder_cond=encoder_cond),
    batched=True,
    desc="Encode running",
)

train_dataset.save_to_disk("/home/vmeshchaninov/nlp_models/data/rocstories/grouped_data/train_encodings/")
test_dataset.save_to_disk("/home/vmeshchaninov/nlp_models/data/rocstories/grouped_data/test_encodings/")
