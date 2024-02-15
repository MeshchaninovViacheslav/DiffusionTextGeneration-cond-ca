import json

from estimation_utils.evaluation import *

file_name = "./generated_texts/common_gen-bert-base-cased-bert-base-cased-batch=512-lr=0.0004/45000-N=50-len=993-cfg=0.json"
text_list = json.load(open(file_name, "r"))

references = [d["GT"] for d in text_list]
predictions = [d["GEN"] for d in text_list]

metrics_rouge = compute_rouge(all_texts_list=predictions, human_references=references)
for rouge_type in ['2', 'L']:
    print(f"Rouge-{rouge_type}: {metrics_rouge[f'rouge{rouge_type}']:0.5f}")

meteor = compute_meteor(predictions=predictions, mult_references=references)
print(f"Meteor: {meteor:0.5f}")

bleu_res = compute_bleu(predictions=predictions, references=references)
for b in ["BLEU-3", "BLEU-4"]:
    print(f"{b}: {bleu_res[b]:0.5f}")

