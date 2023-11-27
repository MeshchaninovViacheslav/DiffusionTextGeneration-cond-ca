from datasets import load_dataset
from itertools import chain
from transformers import BertTokenizerFast

def load_wiki():
    print("Dataset loading")
    dt = load_dataset("Graphcore/wikipedia-bert-512")
    dt = dt.remove_columns(["token_type_ids", "labels", "next_sentence_label", "attention_mask"])
    dt = dt["train"].train_test_split(test_size=0.001, seed=0)


    def find_indx(sample):
        ind1_102, ind2_102 = None, None
        ind = 0

        while ind1_102 is None:
            if sample[ind] == 102:
                if ind2_102 is not None:
                    ind1_102 = ind2_102
                ind2_102 = ind
            ind += 1
        return ind1_102, ind2_102

    def parse_sample(sample):
        sample = sample["input_ids"]
        ind1_102, ind2_102 = find_indx(sample)
        sample1 = sample[1: ind1_102]
        sample2 = sample[ind1_102 + 1: ind2_102]
        return {"input_ids": [sample1, sample2]}

    def flatten(batch):
        batch = list(chain(*batch["input_ids"]))
        return {"input_ids": batch}

    # def detokenize(sample, tokenizer):
    #     sample = chain(*sample["text"])
    #     texts = tokenizer.batch_decode(sample)
    #     return {"text": texts}


    print("Separation")
    dt_train = dt["train"].map(lambda sample: parse_sample(sample), num_proc=100)
    dt_test = dt["test"].map(lambda sample: parse_sample(sample), num_proc=100)


    print("Flatten")
    dt_train = dt_train.map(lambda batch: flatten(batch), batched=True, num_proc=100)
    dt_test = dt_test.map(lambda batch: flatten(batch), batched=True, num_proc=100)


    min_sequence_length = 128
    print("Before filter", len(dt_train))
    dt_train = dt_train.filter(lambda sample: len(sample["input_ids"]) >= min_sequence_length, num_proc=100)
    print("After filter", len(dt_train))

    print("Before filter", len(dt_test))
    dt_test = dt_test.filter(lambda sample: len(sample["input_ids"]) >= min_sequence_length, num_proc=100)
    print("After filter", len(dt_test))


    dt_train.save_to_disk(
        "/home/vmeshchaninov/nlp_models/data/wikipedia/filtered_input_ids/train",
        max_shard_size="10GB",
        num_proc=8
    )
    dt_test.save_to_disk("/home/vmeshchaninov/nlp_models/data/wikipedia/filtered_input_ids/test")


def filter_wiki():
    from datasets import Dataset

    min_sequence_length = 128
    number_of_datasets = 8
    dir_path = "/home/vmeshchaninov/nlp_models/data/wikipedia/"
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    def detokenize(batch, tokenizer):
        batch = batch["text"]
        texts = tokenizer.batch_decode(batch)
        return {"text": texts}
    
    # train
    list_of_datasets = ["/test/data-00000-of-00001.arrow"] + [
        f"/train/data-{i:05d}-of-{number_of_datasets:05d}.arrow"
        for i in range(number_of_datasets)
    ]

    for name in list_of_datasets:
        path = f"{dir_path}/input_ids/{name}"
        dt = Dataset.from_file(path)
        print("Before filter", len(dt))
        dt = dt.filter(lambda sample: len(sample["text"]) >= min_sequence_length, num_proc=100)
        print("After filter", len(dt))
        # dt = dt.map(
        #     lambda batch: detokenize(batch, tokenizer),
        #     batched=True,
        #     num_proc=30,
        #     desc="Dataset tokenization",
        #     batch_size=1000,
        # )
        dt.save_to_disk(f"{dir_path}/filtered_input_ids/{name}")
        

load_wiki()