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
    perplexity = load("/home/vmeshchaninov/nlp_models/metrics/perplexity/", module_type="metric")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
    tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

    prompt_tok = tokenizer(
        all_prompts_list,
        add_special_tokens=False,
        padding=True,
        truncation=False,
        return_tensors="np",
        return_attention_mask=True,
    )
    joint_texts_tok = tokenizer(
        all_joint_texts_list,
        add_special_tokens=False,
        padding=True,
        truncation=False,
        return_tensors="np",
        return_attention_mask=True,
    )
    prompts_results = perplexity.compute(predictions=all_prompts_list, model_id=model_id, device='cuda')
    joint_texts_results = perplexity.compute(predictions=all_joint_texts_list, model_id=model_id, device='cuda')
    cond_result = np.exp(
        (
            np.log(joint_texts_results["perplexities"]) * np.sum(joint_texts_tok["attention_mask"], axis=1) - \
            np.log(prompts_results["perplexities"]) * np.sum(prompt_tok["attention_mask"], axis=1)
        ) / (np.sum(joint_texts_tok["attention_mask"], axis=1) - np.sum(prompt_tok["attention_mask"], axis=1))
    )
    return cond_result.mean()


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

def compute_rouge_hg(all_texts_list, human_references):
    torch.cuda.empty_cache() 

    rouge = load("/home/vmeshchaninov/nlp_models/metrics/rouge/", module_type="metric")
    assert len(all_texts_list) == len(human_references)

    metrics = rouge.compute(predictions=all_texts_list, references=human_references)
    return metrics

def compute_rouge(all_texts_list, human_references):
    torch.cuda.empty_cache() 

    #https://github.com/Yuanhy1997/SeqDiffuSeq/blob/0d428700076211081312c5f9a8c3cbdbd90ba4b8/rouge.py#L118
    from rouge_score import rouge_scorer

    rouge_types = ['rouge1', 'rouge2', 'rougeL']

    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=False)

    metrics_dict = {}
    for r in rouge_types:
        metrics_dict[r] = []
    
    for pred, ref in zip(all_texts_list, human_references):
        score = scorer.score(ref, pred)
        for r in rouge_types:
            metrics_dict[r].append(score[r].fmeasure)

    result = {}
    for r in rouge_types:
        result[r] = np.mean(metrics_dict[r])
    return result

def compute_bert_score(all_texts_list, human_references):
    torch.cuda.empty_cache()

    bertscore = load("/home/vmeshchaninov/nlp_models/metrics/bertscore/", module_type="metric")
    results = bertscore.compute(predictions=all_texts_list, references=human_references, model_type='microsoft/deberta-xlarge-mnli', lang='en', verbose=True)
    # https://github.com/Shark-NLP/DiffuSeq/blob/f78945d79de5783a4329695c0adb1e11adde31bf/scripts/eval_seq2seq.py#L128C48-L128C115
    return np.mean(results["f1"])
    
def compute_bleu_hg(predictions, references, max_order=4, smooth=False):
    torch.cuda.empty_cache()

    bleu = load("bleu")
    results = bleu.compute(predictions=predictions, references=references, max_order=max_order, smooth=smooth)
    return results["bleu"]
    
def compute_bleu(predictions, references, max_order=4, smooth=False):
    torch.cuda.empty_cache()
    
    from estimation_utils.nmt_bleu import compute_bleu as bleu
    tokenizer_mbert = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    references = [[tokenizer_mbert.tokenize(item)] for item in references]
    predictions = [tokenizer_mbert.tokenize(item) for item in predictions]

    results = bleu(reference_corpus=references, translation_corpus=predictions, max_order=max_order, smooth=smooth)
    return results[0]