import torch
from transformers import BloomTokenizerFast, BloomForCausalLM
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import GPTNeoForCausalLM
from torch.nn.functional import cross_entropy
from typing import List

from utils.util import dict_to_device


class BloomMetric:
    def __init__(self, device="cpu"):
        self.name = "bigscience/bloom-7b1"
        self.model = BloomForCausalLM.from_pretrained(self.name).eval().to(device)
        self.tokenizer = BloomTokenizerFast.from_pretrained(self.name)
        self.device = device

    @torch.no_grad()
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def __call__(self, text, reduce="mean"):
        if not text:
            return 0, 0
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
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def __call__(self, cond_text: str, gen_text: str, reduce="mean"):
        # the first word is necessary for tokens to start with an unnecessary word, because metric doeesn't count it
        inputs = self.tokenizer(f" {cond_text} {gen_text}", return_tensors="pt")
        inputs_gen = self.tokenizer(f"{gen_text}", return_tensors="pt")

        inputs = dict_to_device(inputs, self.device)
        outputs = self.model(**inputs, labels=inputs["input_ids"])

        losses = cross_entropy(
            input=outputs.logits.reshape(-1, outputs.logits.shape[-1])[:-1],
            target=inputs["input_ids"].reshape(-1)[1:],
            reduce=False,
        )

        losses = losses[-torch.sum(inputs_gen["attention_mask"]).item():]
        num_tokens = losses.shape[0]
        loss = torch.mean(losses)

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


class RobertaMetric:
    def __init__(self, device: str = "cpu"):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.name = "textattack/roberta-base-CoLA"
        self.model = AutoModelForSequenceClassification.from_pretrained(self.name).eval().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        self.device = device

    @torch.no_grad()
    def __call__(self, texts: List[str], reduce="mean"):
        inputs = self.tokenizer(texts, padding=True, return_tensors="pt")
        inputs = dict_to_device(inputs, self.device)
        output = self.model(**inputs)
        probs = torch.softmax(output.logits, -1)[:, 1]
        naturalness = torch.mean(probs)
        return naturalness.item(), probs
