from datasets import load_dataset

dt = load_dataset("Graphcore/wikipedia-bert-128")
dt = dt.remove_columns(["token_type_ids", "labels", "next_sentence_label"])
dt = dt["train"].train_test_split(test_size=0.001, seed=0)

dt["train"].save_to_disk("/home/vmeshchaninov/nlp_models/data/wikipedia-bert-128/train",
                         max_shard_size="10GB",
                         num_proc=8)

dt["test"].save_to_disk("/home/vmeshchaninov/nlp_models/data/wikipedia-bert-128/test")