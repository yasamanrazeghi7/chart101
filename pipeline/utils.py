from typing import Literal, Union
from enum import Enum
from pathlib import Path

import pandas as pd

from evaluation_script.pipeline.constants import (
    CONSODLIATED_DATASET_PATH,
    OPENAI_API_KEY_PATH,
    SEEDS,
    DatasetName,
    DatasetSplit,
    ModelName,
)
from openai import OpenAI


def get_dataset_name(dataset_split: DatasetSplit) -> DatasetName:
    if dataset_split in [DatasetSplit.BAR, DatasetSplit.SCATTER, DatasetSplit.PIE]:
        return DatasetName.SYNTHETIC
    elif dataset_split in [DatasetSplit.ADDITIONAL, DatasetSplit.ORIGINAL]:
        return DatasetName.CHART_QA
    raise ValueError(f"Invalid dataset split: {dataset_split}")


def get_question_source_path(dataset_split: DatasetSplit) -> Path:
    dataset_name = get_dataset_name(dataset_split)
    question_source_path = (
        CONSODLIATED_DATASET_PATH
        / "SourceQuestion"
        / dataset_name.value
        / f"{dataset_split.value}.jsonl"
    )
    return question_source_path


def get_model_raw_path(
    model_name: ModelName, dataset_split: DatasetSplit, seed: int
) -> Path:
    dataset_name = get_dataset_name(dataset_split)
    model_raw_path = (
        CONSODLIATED_DATASET_PATH
        / "ModelRawOutput"
        / model_name.value
        / dataset_name.value
        / dataset_split.value
        / f"{seed}.jsonl"
    )
    return model_raw_path


def get_processed_model_path(
    model_name: ModelName, dataset_split: DatasetSplit, seed: int
) -> Path:
    dataset_name = get_dataset_name(dataset_split)
    model_processed_path = (
        CONSODLIATED_DATASET_PATH
        / "ModelProcessedOutput"
        / model_name.value
        / dataset_name.value
        / dataset_split.value
        / f"{seed}.jsonl"
    )
    return model_processed_path


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def aggregate_results(dataset_name: DatasetName) -> pd.DataFrame:
    all_dfs = []
    for model_name in ModelName:
        for split in DatasetSplit:
            if dataset_name != get_dataset_name(split):
                continue
            for seed in SEEDS:
                model_processed_path = get_processed_model_path(model_name, split, seed)
                df = pd.read_json(model_processed_path, lines=True)
                all_dfs += [df]
    aggregated_df = pd.concat(all_dfs, ignore_index=True)
    return aggregated_df


#### -------- Open AI -------- ####
client = None


def get_openai_key(api_key_file: Union[str, Path] = OPENAI_API_KEY_PATH):
    # Read the API key from the file
    with open(api_key_file, "r") as file:
        api_key = file.read().strip()
    return api_key


def ask_gpt4(prompt):
    global client
    if client is None:
        client = OpenAI(api_key=get_openai_key())
    response = client.completions.create(model="gpt-3.5-turbo-instruct", prompt=prompt)
    return response.choices[0].text
