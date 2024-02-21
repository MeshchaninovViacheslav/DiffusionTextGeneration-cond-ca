import os
import json
import torch.distributed as dist
from typing import Dict, List
from datasets import load_from_disk

from utils import set_seed
from .evaluation import *
from .util import gather_texts
from .diversity_metrics import NGramStats


def estimate(diffusion):
    seed = diffusion.config.seed + dist.get_rank()
    set_seed(seed)

    num_texts = int(diffusion.config.validation.num_gen_texts / dist.get_world_size())
    if (diffusion.config.validation.num_gen_texts % dist.get_world_size()) > dist.get_rank():
        num_texts += 1
    
    result_dict = diffusion.generate_text(num_texts)
    
    for key in result_dict:
        result_dict[key] = gather_texts(result_dict[key])

    if dist.get_rank() == 0:
        # Save texts
        text_list = []
        for i in range(len(result_dict["GEN"])):
            if result_dict["GEN"][i]:
                text_list.append({key: result_dict[key][i] for key in result_dict})

            if len(text_list) >= diffusion.config.validation.num_text_to_est:
                break
        
        os.makedirs(diffusion.config.validation.texts_path, exist_ok=True)
        prefix_folder = os.path.join(diffusion.config.validation.texts_path, diffusion.config.training.checkpoints_prefix)
        os.makedirs(prefix_folder, exist_ok=True)
        file_name = f"{diffusion.step}-N={diffusion.config.dynamic.N}-len={len(text_list)}.json"
        save_path = os.path.join(prefix_folder, file_name)
        json.dump(text_list, open(save_path, "w"), indent=4)

        # Metrics computation
        compute_metrics(diffusion, text_list)


def compute_metrics(diffusion, text_list: List[Dict[str, str]]):
    if not diffusion.config.is_conditional:
        compute_metrics_uncond(diffusion, text_list)
    else:
        if diffusion.data.dataset in ["rocstory"]:
            compute_metrics_cond(diffusion, text_list)


def compute_metrics_uncond(diffusion, text_list: List[Dict[str, str]]):
    dt = load_from_disk(f"{diffusion.data.dataset_path}/train/")
    train_references = dt["text"]

    references = [d["GT"] for d in text_list]
    predictions = [d["GEN"] for d in text_list]
    
    ppl = compute_perplexity(all_texts_list=predictions)
    div = compute_diversity(all_texts_list=predictions)['diversity']
    mem = compute_memorization(all_texts_list=predictions, human_references=train_references)
    try:
        mauve = compute_mauve(all_texts_list=predictions, human_references=references)
    except Exception:
        mauve = 0.
    metric_div = NGramStats()
    metric_div.compute(predictions)

    diffusion.log_metric(metric_name="GPT2-large ppl", loader_name="", value=ppl)
    diffusion.log_metric(metric_name="Diversity", loader_name="", value=div)
    diffusion.log_metric(metric_name="Memorization", loader_name="", value=mem)
    diffusion.log_metric(metric_name="Mauve", loader_name="", value=mauve)
    for key in metric_div.results:
        diffusion.log_metric(metric_name=f"diversity metrics", loader_name=key, value=metric_div.results[key])

    print(f"GPT2-large ppl: {ppl:0.5f}")
    print(f"Diversity: {div:0.5f}")
    print(f"Memorization: {mem:0.5f}")
    print(f"Mauve: {mauve:0.5f}")


def compute_metrics_cond(diffusion, text_list: List[Dict[str, str]]):
    dt = load_from_disk(f"{diffusion.data.dataset_path}/train/")
    train_references = dt["text"]

    references = [d["GT"] for d in text_list]
    predictions = [d["GEN"] for d in text_list]
    prompts = [d["COND"] for d in text_list]
    joint_texts = [f"{prompt} {pred}" for prompt, pred in  zip(prompts, predictions)]
    
    ppl = compute_conditional_perplexity(all_prompts_list=prompts, all_joint_texts_list=joint_texts)
    div = compute_diversity(all_texts_list=predictions)['diversity']
    mem = compute_memorization(all_texts_list=predictions, human_references=train_references)
    try:
        mauve = compute_mauve(all_texts_list=predictions, human_references=references)
    except Exception:
        mauve = 0.
    metric_div = NGramStats()
    metric_div.compute(predictions)

    diffusion.log_metric(metric_name="GPT2-large ppl", loader_name="", value=ppl)
    diffusion.log_metric(metric_name="Diversity", loader_name="", value=div)
    diffusion.log_metric(metric_name="Memorization", loader_name="", value=mem)
    diffusion.log_metric(metric_name="Mauve", loader_name="", value=mauve)
    for key in metric_div.results:
        diffusion.log_metric(metric_name=f"diversity metrics", loader_name=key, value=metric_div.results[key])

    print(f"GPT2-large ppl: {ppl:0.5f}")
    print(f"Diversity: {div:0.5f}")
    print(f"Memorization: {mem:0.5f}")
    print(f"Mauve: {mauve:0.5f}")