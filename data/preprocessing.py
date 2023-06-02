import functools
from collections import defaultdict
import numpy as np


def text_preprocessor(dt, benchmark_name, config):
    if benchmark_name == "c4":
        return preprocessing_c4(dt)
    elif benchmark_name == "wiki_dpr":
        return preprocessing_wiki_dpr(dt)
    return glue_text_preprocessor(dt, benchmark_name, config)


def preprocessing_c4(dt):
    return dt.rename_column(original_column_name="text", new_column_name="inputs"). \
        remove_columns(column_names=["url", "timestamp"])


def preprocessing_wiki_dpr(dt):
    return dt.rename_column(original_column_name="text", new_column_name="inputs"). \
        rename_column(original_column_name="id", new_column_name="idx"). \
        remove_columns(column_names=["title"])


def glue_text_preprocessor(dt, benchmark_name, config):
    if benchmark_name == "stsb":
        return dt.map(functools.partial(stsb_prep,
                                        benchmark_name=benchmark_name,
                                        feature_keys=config["data"][benchmark_name]["feature_keys"]),
                      remove_columns=dt.column_names,
                      num_proc=32)
    elif benchmark_name == "record":
        dt_new = dt.map(record_prep, batched=True, remove_columns=dt.column_names, num_proc=32)
        return dt_new  # .add_column(name="idx", column=list(range(dt_new.num_rows)))
    else:
        return dt.map(functools.partial(
            glue_prep,
            benchmark_name=benchmark_name,
            feature_keys=config["data"][benchmark_name]["feature_keys"],
            label_classes=config["data"][benchmark_name]["label_classes"]
        ),
            remove_columns=dt.column_names,
            num_proc=4
        )


def make_glue_input(x, benchmark_name, feature_keys):
    strs_to_join = [benchmark_name]
    for key in feature_keys:
        strs_to_join.append("{}:".format(key))
        strs_to_join.append(x[key])
    inputs = " ".join(strs_to_join)
    return inputs


def glue_prep(x, benchmark_name, feature_keys, label_classes):
    """Convert a dataset from glue to text2text examples.

        This function uses the feature names from the dataset to unpack examples into
        a format amenable for a text2text problem. For example, consider the Quora
        Question Pairs (QQP) benchmark, which would suggest
        benchmark_name="qqp"
        label_names=['not_duplicate', 'duplicate']
        For QQP, a typical example might look like
        {
            "question1": "Why do I easily get bored of my friends?",
            "question2": "Why do I get bored of friends so quickly?",
            "label": 1,
            "idx": 10,
        }

        This example would be transformed to
        {
             "inputs": (
                 "qqp question1: Why do I easily get bored of my friends? question2: "
                 "Why do I get bored of my friends so quickly?"
             ),
             "targets": "duplicate",
            "idx": 10,
        }

        Args:
          x: an example to process.
          benchmark_name: the name of the GLUE benchmark for this dataset.
          label_names: a list of label names corresponding to class index.
          feature_names: an optional ordered list of feature names. If provided,
            features will be ordered in this way in the output. If not provided, all
            features (except 'idx' and 'label') will be used, sorted by name.
          id_key: str, key for id in the dataset. If not provided, 'idx' will be used.
            if None, no id will be added to the dataset.

        Returns:
          A preprocessed example.

        Tasks:
            cola, sst2, mrpc, qqp, mnli, 'qnli', 'rte'
        """

    inputs = make_glue_input(x, benchmark_name, feature_keys)
    targets = "<unk>" if x["label"] == -1 else label_classes[x["label"]]
    return {'inputs': inputs, 'targets': targets, 'idx': x['idx']}


def stsb_prep(x, benchmark_name, feature_keys):
    """Convert STSB examples to text2text format.

    STSB maps two sentences to a floating point number between 1 and 5
    representing their semantic similarity. Since we are treating all tasks as
    text-to-text tasks we need to convert this floating point number to a string.
    The vast majority of the similarity score labels in STSB are in the set
    [0, 0.2, 0.4, ..., 4.8, 5.0]. So, we first round the number to the closest
    entry in this set, and then we convert the result to a string (literally e.g.
    "3.4"). This converts STSB roughly into a 26-class classification dataset.
    This function uses the feature names from the dataset to unpack examples into
    a format amenable for a text2text problem.

    For example, a typical example from STSB might look like
    {
        "sentence1": "Three more US soldiers killed in Afghanistan",
        "sentence2": "NATO Soldier Killed in Afghanistan",
        "label": 1.8,
    }

    This example would be transformed to
    {
         "inputs": (
             "stsb sentence1: Three more US soldiers killed in Afghanistan "
             "sentence2: NATO Soldier Killed in Afghanistan"
         ),
         "targets": "1.8",
    }

    Args:
      x: an example to process.
    Returns:
      A preprocessed example.
    """

    inputs = make_glue_input(x, benchmark_name, feature_keys)
    targets = f"{round(x['label'] * 5) / 5:0.1f}"
    return {'inputs': inputs, 'targets': targets, 'idx': x['idx']}


def record_prep(batch):
    """Convert ReCoRD examples to text2text examples.

    ReCoRD contains a passage, query containing a '@placeholder' string, and a set
    of entities that are the possible values of the placeholder. Each train and
    validation example will have a list of answers, any of which would be
    considered correct.

    For example, a typical example from ReCoRD might look like
    {
        'passsage': 'This is the passage.',
        'query': 'A @placeholder is a bird.',
        'entities': ['penguin', 'potato', 'pigeon'],
        'answers': ['penguin', 'pigeon'],
    }
    which this preprocessor would turn into the following two examples:
    {
        'inputs': 'record query: A @placeholder is a bird. entities: penguin, '
                  'potato, pigeon passage: This is the passage.',
        'targets': 'penguin',
    }
    and
    {
        'inputs': 'record query: A @placeholder is a bird. entities: penguin, '
                  'potato, pigeon passage: This is the passage.',
        'targets': 'pigeon',
    }

    Args:
      dataset: Dataset to process.

    Returns:
      a Dataset
    """

    extended_batch = defaultdict(list)
    keys = batch.keys()
    for values in zip(*batch.values()):
        x = {k: v for k, v in zip(keys, values)}
        inputs = f"record query: {x['query']} entities: {', '.join(x['entities'])} passage: {x['passage']}"
        extended_batch["inputs"].extend([inputs] * len(x["answers"]))
        extended_batch["targets"].extend(x["answers"])
    return extended_batch


def unsupervised_preprocessor(dt, benchmark_name):
    def prep_fn(x):
        inputs = f"{x['inputs']} answer: {x['targets']}"
        return {"inputs": inputs}

    if "targets" in dt.column_names:
        return dt.map(prep_fn, num_proc=32, remove_columns=["idx", "targets"])
    return dt


def supervised_preprocessor(dt, benchmark_name):
    def prep_fn(x):
        inputs = f"{x['inputs']} answer: "
        return {"inputs": inputs, "targets": x["targets"]}

    if "idx" in dt.column_names:
        return dt.map(prep_fn, num_proc=32, remove_columns=["idx"])
    return dt


def conditional_preprocessing_wiki(
        element,
        tokenizer_bert, tokenizer_cond, tokenizer_gen,
        max_sequence_len: int = 96,
        pos_begin: float = 0.33,
        pos_end: float = 0.67,
):
    elem_count = sum(element["attention_mask"])
    delimeter_pos = int(
        (
                np.random.rand() * (pos_end - pos_begin) + pos_begin
        ) * elem_count
    )

    cond_ids = element["input_ids"][:delimeter_pos]
    input_ids = element["input_ids"][delimeter_pos:]

    cond_ = tokenizer_cond.encode_plus(
        text=tokenizer_bert.decode(cond_ids, skip_special_tokens=True),
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=max_sequence_len,
    )

    input_ = tokenizer_gen.encode_plus(
        text=tokenizer_bert.decode(input_ids, skip_special_tokens=True),
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=max_sequence_len,
    )

    output = {
        "input_ids": input_["input_ids"],
        "cond_ids": cond_["input_ids"],
        "input_mask": input_["attention_mask"],
        "cond_mask": cond_["attention_mask"],
    }
    return output


def glue_tokenize(element, tokenizer_cond, tokenizer_gen, max_sequence_len):
    cond_ = tokenizer_cond.encode_plus(
        text=element["inputs"],
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=max_sequence_len,
    )

    input_ = tokenizer_gen.encode_plus(
        text=element["targets"],
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=max_sequence_len,
    )
    output = {
        "input_ids": input_["input_ids"],
        "cond_ids": cond_["input_ids"],
        "input_mask": input_["attention_mask"],
        "cond_mask": cond_["attention_mask"],
    }
    return output
