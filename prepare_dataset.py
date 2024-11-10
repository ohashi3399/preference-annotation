import json
from typing import List, Dict
from datasets import load_dataset, Dataset, DatasetDict


def convert_jsonl_to_dataset(jsonl_list: List[Dict]) -> Dataset:
    """
    Convert a list of JSON objects to a Hugging Face Dataset.

    Args:
        jsonl_list (List[Dict]): List of dictionaries containing the data

    Returns:
        Dataset: A Hugging Face Dataset object
    """
    # すべてのレコードのキーが同じであることを確認
    first_keys = set(jsonl_list[0].keys())
    for item in jsonl_list[1:]:
        if set(item.keys()) != first_keys:
            raise ValueError("All records must have the same keys")

    # データを列形式に変換
    columns = {key: [] for key in first_keys}
    for item in jsonl_list:
        for key in first_keys:
            columns[key].append(item[key])

    # Datasetオブジェクトを作成
    dataset = Dataset.from_dict(columns)

    return dataset


ds1_name = "Aratako/WIP-Dataset-For-Self-Taught-Evaluators"
ds1 = load_dataset(ds1_name)["train"]

ds2_name = "Aratako/WIP-Dataset-For-Self-Taught-Evaluators-v2"
ds2 = load_dataset(ds2_name)["train"]

corpus_list = list()
for record1, record2 in zip(ds1, ds2):

    data_point = dict()

    data_point["instruction"] = record1["instruction"]
    data_point["response1"] = record1["output"]
    data_point["response2"] = record2["output"]
    data_point["annotation_done"] = 0
    corpus_list.append(data_point)

corpus = convert_jsonl_to_dataset(corpus_list)
corpus = DatasetDict({"train": corpus})
corpus.push_to_hub("preference_test")
