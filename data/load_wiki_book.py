from datasets import load_dataset
from itertools import chain
from transformers import BertTokenizerFast

dt = load_dataset("gokuls/wiki_book_corpus_complete_processed_bert_dataset", num_proc=10)
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


def detokenizer(examples):
    block_size = 128
    key = "input_ids"
    concatenated_examples = list(chain(*examples[key]))
    total_length = (len(concatenated_examples) // block_size) * block_size
    # Split by chunks of max_len.
    result = {}
    result["sentences"] = tokenizer.batch_decode(
        [concatenated_examples[i: i + block_size] for i in range(0, total_length, block_size)]
    )
    return result

print("Dataset is loaded")

dt = dt["train"].remove_columns(["token_type_ids", "attention_mask", "special_tokens_mask"])

print(f"Grouping texts in chunks")

dt_chunked = dt.map(
    detokenizer,
    batched=True,
    num_proc=60,
    desc=f"Grouping texts in chunks",
    remove_columns=["input_ids"]
)

dt_dict = dt_chunked.train_test_split(test_size=0.005, seed=0)

path_dir = "/home/vmeshchaninov/nlp_models/data/wikipedia-books-128-text"
dt_dict["train"].save_to_disk(f"{path_dir}/train",
                         max_shard_size="10GB",
                         num_proc=8)

dt_dict["test"].save_to_disk(f"{path_dir}/test")