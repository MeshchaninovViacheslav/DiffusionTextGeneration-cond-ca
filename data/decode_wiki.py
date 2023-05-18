import os
from tqdm import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast


def path_format(split, n):
    if split == "train":
        path = f"/home/vmeshchaninov/nlp_models/data/wikipedia-bert-128/{split}/data-{n:05d}-of-00008.arrow"
    else:
        path = f"/home/vmeshchaninov/nlp_models/data/wikipedia-bert-128/{split}/data-{n:05d}-of-00001.arrow"
    return path


save_dir = f"/home/vmeshchaninov/nlp_models/data/wikipedia-bert-128-text/"
os.makedirs(save_dir, exist_ok=True)

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

splits = ["train", "test"]
for split in splits:
    os.makedirs(f"{save_dir}/{split}", exist_ok=True)

    N = 1 if split == "test" else 8

    for n in range(N):
        file_name = f"{save_dir}/{split}/data-{n:05d}-of-00008.txt"
        with open(file_name, "w") as file:
            path = path_format(split, n)
            dt = DataLoader(
                Dataset.from_file(path),
                batch_size=2048 * 4,
                num_workers=1,
            )
            for batch in tqdm(dt):
                text = tokenizer.batch_decode(
                    batch["input_ids"],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                for t in text:
                    if not t:
                        continue
                    print(t, file=file)
