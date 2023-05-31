import torch
from transformers import BloomTokenizerFast, BloomForCausalLM
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import GPTNeoForCausalLM
from torch.nn.functional import cross_entropy

from utils.util import dict_to_device


class BloomMetric:
    def __init__(self, device="cpu"):
        self.name = "bigscience/bloom-7b1"
        self.model = BloomForCausalLM.from_pretrained(self.name).eval().to(device)
        self.tokenizer = BloomTokenizerFast.from_pretrained(self.name)
        self.device = device

    @torch.no_grad()
    def __call__(self, text, reduce="mean"):
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = dict_to_device(inputs, self.device)

        loss = self.model(**inputs, labels=inputs["input_ids"]).loss.detach().cpu()
        num_tokens = torch.sum(inputs["attention_mask"]).item() - 1
        if reduce == "sum":
            return loss.item() * num_tokens, num_tokens
        return loss.item(), num_tokens


class BloomMetricConditional:
    def __init__(self, device="cpu"):
        self.name = "bigscience/bloom-7b1"
        self.model = BloomForCausalLM.from_pretrained(self.name).eval().to(device)
        self.tokenizer = BloomTokenizerFast.from_pretrained(self.name)
        self.device = device

    @torch.no_grad()
    def __call__(self, text_cond: str, text_gen: str, reduce="mean"):
        inputs_gen = self.tokenizer(text_gen, return_tensors="pt")
        inputs_cond = self.tokenizer(f" {text_cond}", return_tensors="pt")

        inputs = {
            "input_ids": torch.cat([inputs_cond["input_ids"], inputs_gen["input_ids"]], dim=-1).type(torch.long),
            "attention_mask": torch.cat([inputs_cond["attention_mask"], inputs_gen["attention_mask"]], dim=-1).type(
                torch.long)
        }
        inputs = dict_to_device(inputs, self.device)
        outputs = self.model(**inputs, labels=inputs["input_ids"])

        losses = cross_entropy(
            input=outputs.logits.reshape(-1, outputs.logits.shape[-1])[:-1],
            target=inputs["input_ids"].reshape(-1)[1:],
            reduce=False,
        )
        losses = losses[torch.sum(inputs_cond["attention_mask"]).item() - 1:]

        loss = torch.mean(losses)
        num_tokens = losses.shape[0]

        if reduce == "sum":
            return loss.item() * num_tokens, num_tokens
        return loss.item(), num_tokens


class GPTMetric:
    def __init__(self, device="cpu"):
        self.name = "gpt2-large"
        self.model = GPT2LMHeadModel.from_pretrained(self.name).eval().to(device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.name)
        self.device = device

    @torch.no_grad()
    def __call__(self, text, reduce="mean"):
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = dict_to_device(inputs, self.device)

        loss = self.model(**inputs, labels=inputs["input_ids"]).loss.detach().cpu()
        num_tokens = torch.sum(inputs["attention_mask"]).item() - 1
        if reduce == "sum":
            return loss.item() * num_tokens, num_tokens
        return loss.item(), num_tokens


class GPTNEOMetric:
    def __init__(self, device="cpu"):
        self.name = "EleutherAI/gpt-neo-2.7B"
        self.model = GPTNeoForCausalLM.from_pretrained(self.name).eval().to(device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.name)
        self.device = device

    @torch.no_grad()
    def __call__(self, text, reduce="mean"):
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = dict_to_device(inputs, self.device)

        loss = self.model(**inputs, labels=inputs["input_ids"]).loss.detach().cpu()
        num_tokens = torch.sum(inputs["attention_mask"]).item() - 1
        if reduce == "sum":
            return loss.item() * num_tokens, num_tokens
        return loss.item(), num_tokens


from transformers import T5TokenizerFast, T5ForConditionalGeneration


class T5Metric:
    def __init__(self, device):
        self.t5_name = "t5-3b"
        self.model = T5ForConditionalGeneration.from_pretrained(self.t5_name).eval().to(device)
        self.tokenizer = T5TokenizerFast.from_pretrained(self.t5_name)
        self.device = device

    @torch.no_grad()
    def __call__(self, batch_texts):
        condition = [""] * len(batch_texts)
        encoding = self.tokenizer(condition, return_tensors="pt")
        input_ids = encoding.input_ids.to(self.device)
        attention_mask = encoding.attention_mask.to(self.device)
        labels = self.tokenizer(
            batch_texts,
            padding="longest",
            max_length=64,
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(self.device)
        # labels = torch.tensor(labels)
        labels[labels == self.tokenizer.pad_token_id] = -100

        loss = self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          labels=labels).loss.detach().cpu()
        return loss.item()
