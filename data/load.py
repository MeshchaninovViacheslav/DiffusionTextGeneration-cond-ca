from datasets import load_dataset
from collections import defaultdict
from create_config import create_config
from itertools import chain


def download_wikipedia(base_path):
    # for private dataset 
    # huggingface-cli login --token <token>

    dt = load_dataset("bigscience-data/roots_en_wikipedia")
    dt = dt["train"]
    dt = dt.remove_columns("meta")
    
    def split(batch, min_num_symbols=100):
        result = []
        for text in batch["text"]:
            texts = text.split("\n\n")
            texts = [t for t in texts if len(t) >= min_num_symbols]
            result.append(texts)
        result = list(chain(*result))
        return {"text": result}
    
    dt = dt.map(
        split,
        batched=True,
        num_proc=30,
        desc="Dataset split",
        batch_size=1000,
    )

    dt = dt.train_test_split(test_size=0.001, seed=0)
    dt.save_to_disk(
        base_path,
    )


def download_rocstory(base_path):
    def unite(batch):
        texts = []
        size = len(batch["storyid"])
        for i in range(size):
            text = " ".join([batch[f"sentence{k}"][i] for k in range(1, 6)])
            texts.append(text)
        return {"text": texts}

    dt = load_dataset("wza/roc_stories")
    dt = dt["train"]
    dt = dt.map(
        unite,
        batched=True,
        num_proc=30,
        desc="Loading...",
        remove_columns=dt.column_names,
    )
    dt = dt.train_test_split(test_size=1000, seed=0)
    dt.save_to_disk(base_path)


def download_glue(base_path):
    base_path = f"{base_path}/glue/"
    configs = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'qnli', 'rte']
    for config in configs:
        load_dataset("glue", config).save_to_disk(f"{base_path}/{config}")
    print(f"GLUE saved in {base_path}")


def download_super_glue(base_path):
    base_path = f"{base_path}/super_glue/"
    configs = ['copa', 'cb', 'boolq', 'wic', 'multirc', 'record']
    for config in configs:
        load_dataset("super_glue", config).save_to_disk(f"{base_path}/{config}")
    print(f"Super-GLUE saved in {base_path}")


def download_wiki_dpr(base_path):
    base_path = f"{base_path}/wiki_dpr/"
    load_dataset("wiki_dpr", "psgs_w100.multiset.no_index.no_embeddings").save_to_disk(base_path)


def с4(base_path):
    base_path = f"{base_path}/с4/"
    load_dataset("с4", "en").save_to_disk(base_path)


# def download_wikipedia(base_path):
#     def wiki_prep(batch):
#         minimal_len = 50
#         extended_batch = defaultdict(list)
#         keys = batch.keys()
#         for values in zip(*batch.values()):
#             x = {k: v for k, v in zip(keys, values)}
#             inputs = x['text'].split("\n")
#             for t in inputs:
#                 if len(t) > minimal_len:
#                     extended_batch["inputs"].extend([t])
#         return extended_batch

#     dt = load_dataset("wikipedia", "20220301.en")["train"]
#     dt = dt.map(wiki_prep, batched=True, remove_columns=dt.column_names, num_proc=32)
#     dt_dict = dt.train_test_split(test_size=0.01, shuffle=True)
#     dt_dict["validation"] = dt_dict.pop("test")
#     dt_dict.save_to_disk(f"{base_path}/wikipedia")


if __name__ == "__main__":
    config = create_config()

    if config.data.dataset_name == "rocstory":
        download_rocstory(config.data.dataset_path)
    if config.data.dataset_name == "wikipedia":
        download_wikipedia(config.data.dataset_path)