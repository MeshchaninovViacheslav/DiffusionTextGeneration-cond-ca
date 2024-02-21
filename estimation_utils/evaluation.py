import torch
from evaluate import load
from nltk.util import ngrams
from collections import defaultdict
import spacy
import numpy as np
from transformers import AutoTokenizer


def compute_perplexity(all_texts_list, model_id='gpt2-large'):
    torch.cuda.empty_cache() 
    perplexity = load("/home/vmeshchaninov/nlp_models/metrics/perplexity/", module_type="metric")
    results = perplexity.compute(predictions=all_texts_list, model_id=model_id, device='cuda')
    return results['mean_perplexity']


def compute_conditional_perplexity(all_joint_texts_list, all_prompts_list, model_id='gpt2-large'):
    torch.cuda.empty_cache() 
    perplexity = load("perplexity", module_type="metric", model_id=model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
    tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

    zero_prompt_texts = []
    no_zero_prompts = []
    no_zero_prompt_texts = []
    for i, p in enumerate(all_prompts_list):
        if not p:
            zero_prompt_texts.append(all_joint_texts_list[i])
        else:
            no_zero_prompts.append(p)
            no_zero_prompt_texts.append(all_joint_texts_list[i])

    zero_prompt_perplexity = perplexity.compute(predictions=zero_prompt_texts, model_id=model_id, device='cuda', add_start_token=True)

    prompt_tok = tokenizer(
        no_zero_prompts,
        add_special_tokens=False,
        padding=True,
        truncation=False,
        return_tensors="np",
        return_attention_mask=True,
    )
    joint_texts_tok = tokenizer(
        no_zero_prompt_texts,
        add_special_tokens=False,
        padding=True,
        truncation=False,
        return_tensors="np",
        return_attention_mask=True,
    )
    prompts_results = perplexity.compute(predictions=no_zero_prompts, model_id=model_id, device='cuda', add_start_token=True)
    joint_texts_results = perplexity.compute(predictions=no_zero_prompt_texts, model_id=model_id, device='cuda', add_start_token=True)
    no_zero_prompt_perplexity = np.exp(
        (
            np.log(joint_texts_results["perplexities"]) * np.sum(joint_texts_tok["attention_mask"], axis=1) - \
            np.log(prompts_results["perplexities"]) * np.sum(prompt_tok["attention_mask"], axis=1)
        ) / (np.sum(joint_texts_tok["attention_mask"], axis=1) - np.sum(prompt_tok["attention_mask"], axis=1))
    )
    result = (no_zero_prompt_perplexity.sum() + np.sum(zero_prompt_perplexity["perplexities"])) / (len(no_zero_prompts) + len(zero_prompt_texts))
    
    return result


def compute_wordcount(all_texts_list):
    wordcount = load("word_count")
    wordcount = wordcount.compute(data=all_texts_list)
    return wordcount['unique_words']

def compute_diversity(all_texts_list):
    ngram_range = [2, 3, 4]

    tokenizer = spacy.load("en_core_web_sm").tokenizer
    token_list = []
    for sentence in all_texts_list:
        token_list.append([str(token) for token in tokenizer(sentence)])
    ngram_sets = {}
    ngram_counts = defaultdict(int)

    metrics = {}
    for n in ngram_range:
        ngram_sets[n] = set()
        for tokens in token_list:
            ngram_sets[n].update(ngrams(tokens, n))
            ngram_counts[n] += len(list(ngrams(tokens, n)))
        metrics[f'{n}gram_repitition'] = (1-len(ngram_sets[n])/ngram_counts[n])
    diversity = 1
    for val in metrics.values():
        diversity *= (1-val)
    metrics['diversity'] = diversity
    return metrics


def compute_memorization(all_texts_list, human_references, n=4):
    tokenizer = spacy.load("en_core_web_sm").tokenizer
    unique_four_grams = set()
    for sentence in human_references:
        unique_four_grams.update(ngrams([str(token) for token in tokenizer(sentence)], n))

    total = 0
    duplicate = 0
    for sentence in all_texts_list:
        four_grams = list(ngrams([str(token) for token in tokenizer(sentence)], n))
        total += len(four_grams)
        for four_gram in four_grams:
            if four_gram in unique_four_grams:
                duplicate += 1

    return duplicate / total


def compute_mauve(all_texts_list, human_references, model_id='gpt2-large'):
    torch.cuda.empty_cache() 

    mauve = load("/home/vmeshchaninov/nlp_models/metrics/mauve/", module_type="metric")
    assert len(all_texts_list) == len(human_references)

    results = mauve.compute(
        predictions=all_texts_list, references=human_references,
        featurize_model_name=model_id, max_text_length=256, device_id=0, verbose=False
    )

    return results.mauve


def compute_rouge(all_texts_list, human_references, model_id='gpt2-large'):
    torch.cuda.empty_cache() 

    rouge = load('rouge')
    assert len(all_texts_list) == len(human_references)

    metrics = rouge.compute(predictions=all_texts_list, references=human_references)
    return metrics


def compute_bert_score(all_texts_list, human_references):
    torch.cuda.empty_cache()

    bertscore = load("bertscore")
    results = bertscore.compute(predictions=all_texts_list, references=human_references, model_type="microsoft/deberta-xlarge-mnli")
    return np.mean(results["f1"])
    