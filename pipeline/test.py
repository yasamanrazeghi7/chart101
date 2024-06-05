import pandas as pd
import pytest

from evaluation_script.pipeline.constants import (
    SEEDS,
    DatasetName,
    DatasetSplit,
    ModelName,
    ProcessedModelResultDataTypes,
    SyntheticProcessedModelResult,
)
from evaluation_script.pipeline.utils import (
    get_dataset_name,
    get_model_raw_path,
    get_processed_model_path,
    get_question_source_path,
)


@pytest.mark.parametrize("split", DatasetSplit)
def test_question_source_path(split):
    path = get_question_source_path(split)
    assert path.exists(), f"Path {path} for {split} does not exist"
    assert path.is_file()
    assert path.suffix == ".jsonl"


@pytest.mark.parametrize("model_name", ModelName)
@pytest.mark.parametrize("split", DatasetSplit)
@pytest.mark.parametrize("seed", SEEDS)
def test_model_raw_path(model_name, split, seed):
    path = get_model_raw_path(model_name, split, seed)
    assert path.exists(), f"Path {path} does not exist"
    assert path.is_file()
    assert path.suffix == ".jsonl"


@pytest.mark.parametrize("split", DatasetSplit)
def test_vadliate_question_source(split):
    path = get_question_source_path(split)
    dataset_name = get_dataset_name(split)
    df = pd.read_json(path, lines=True)
    assert len(df) > 0, f"Empty dataframe for {path}"
    assert not df["q_id"].duplicated().any(), f"Duplicate q_id in {path}!"
    required_columns = dataset_name.get_required_source_columns()
    missing_columns = set(required_columns) - set(df.columns)
    assert not missing_columns, f"Columns {missing_columns} are missing!"
    if dataset_name == DatasetName.SYNTHETIC:
        assert (
            df["x_range"].apply(lambda x: x > 0).all()
        ), "x_range is not a list of two floats!"
        assert (
            df["y_range"].apply(lambda y: y > 0).all()
        ), "y_range is not a list of two floats!"


@pytest.mark.parametrize("model_name", ModelName)
@pytest.mark.parametrize("split", DatasetSplit)
@pytest.mark.parametrize("seed", SEEDS)
def test_validate_model_raw_results(model_name, split, seed):
    if model_name == ModelName.PALI and split == DatasetSplit.SCATTER:
        return
    model_raw_path = get_model_raw_path(model_name, split, seed)

    for model_name_2 in ModelName:
        for split_2 in DatasetSplit:
            for seed_2 in SEEDS:
                if model_name == model_name_2 and split == split_2 and seed == seed_2:
                    continue
                if (
                    model_name == model_name_2 == ModelName.CHART_LLAMA
                    and split == DatasetSplit.BAR
                    and seed in [0, 3]
                    and seed_2 in [0, 3]
                ):
                    continue
                if (
                    model_name == model_name_2 == ModelName.PALI
                    and split == DatasetSplit.SCATTER
                ):
                    continue
                model_raw_path_2 = get_model_raw_path(model_name_2, split_2, seed_2)
                if not model_raw_path_2.exists():
                    continue
                content_1 = model_raw_path.read_text()
                content_2 = model_raw_path_2.read_text()
                assert (
                    content_1 != content_2
                ), f"Content of {model_raw_path} and {model_raw_path_2} are identical!"
    question_source_path = get_question_source_path(split)
    model_raw_df = pd.read_json(model_raw_path, lines=True)
    model_output_column_name = model_name.get_model_output_column_name(
        get_dataset_name(split)
    )
    if model_name == ModelName.PALI:
        assert (
            model_raw_df[model_output_column_name].str.strip().astype(bool).any()
        ), f"Empty values in {model_raw_path}!"
        assert (
            model_raw_df[model_output_column_name].notna().any()
        ), f"Missing values in {model_raw_path}!"
    else:
        assert (
            model_raw_df[model_output_column_name].str.strip().astype(bool).all()
        ), f"Empty values in {model_raw_path}!"
        assert (
            model_raw_df[model_output_column_name].notna().all()
        ), f"Missing values in {model_raw_path}!"
    dataset_name = get_dataset_name(split)
    required_columns = model_name.get_required_columns(dataset_name)
    missing_columns = set(required_columns) - set(model_raw_df.columns)
    assert not missing_columns, f"Columns {missing_columns} are missing!"
    q_id_column_name = model_name.get_question_id_column_name()
    assert (
        not model_raw_df[q_id_column_name].duplicated().any()
    ), f"Duplicate q_id in {model_raw_path}!"
    questions_df = pd.read_json(question_source_path, lines=True)
    assert len(model_raw_df) == len(
        questions_df
    ), f"Number of rows in {model_raw_path} does not match {question_source_path}"
    merged_df = pd.merge(
        model_raw_df,
        questions_df,
        left_on=q_id_column_name,
        right_on="q_id",
        how="outer",
    )
    assert len(merged_df) == len(
        questions_df
    ), f"Number of rows in merged_df does not match {question_source_path}"
    model_raw_question_column_name = model_name.get_question_column_name(dataset_name)
    question_source_question_column_name = dataset_name.get_question_column_name()
    if model_raw_question_column_name == question_source_question_column_name:
        model_raw_question_column_name = f"{model_raw_question_column_name}_x"
        question_source_question_column_name = (
            f"{question_source_question_column_name}_y"
        )
    assert not (
        merged_df[model_raw_question_column_name]
        != merged_df[question_source_question_column_name]
    ).any(), "question_x and question_y do not match!"
    # missing_after_merge = merged_df[['gold_answer', 'correct_answer']].isna().any().any()
    # assert not missing_after_merge, f"Missing values in golden_answer or correct_answer after merge!"
    # assert not (merged_df["gold_answer"] != merged_df["correct_answer"]).any(), "gold_answer and correct_answer do not match!"


@pytest.mark.parametrize("model_name", ModelName)
@pytest.mark.parametrize("split", DatasetSplit)
@pytest.mark.parametrize("seed", SEEDS)
def test_validate_model_processed_results(model_name, split, seed):
    if model_name == ModelName.PALI and split == DatasetSplit.SCATTER:
        return
    dataset_name = get_dataset_name(split)
    question_source_path = get_question_source_path(split)
    model_raw_path = get_model_raw_path(model_name, split, seed)
    model_proccessed_path = get_processed_model_path(model_name, split, seed)
    questions_df = pd.read_json(question_source_path, lines=True)
    model_raw_df = pd.read_json(model_raw_path, lines=True)
    model_processed_df = pd.read_json(model_proccessed_path, lines=True)
    dtypes = (
        ProcessedModelResultDataTypes
        if dataset_name == DatasetName.CHART_QA
        else SyntheticProcessedModelResult
    )
    missing_columns = set(dtypes.keys()) - set(model_processed_df.columns)
    assert not missing_columns, f"Columns {missing_columns} are missing!"

    
    merged_processed_question_df = pd.merge(
        model_processed_df,
        questions_df,
        left_on="question_id",
        right_on="q_id",
        how="outer",
    )
    assert (
        len(merged_processed_question_df) == len(questions_df),
        f"Number of rows in merged_processed_question_df does not match {question_source_path}",
    )
    model_processed_question_column_name = "question"
    question_source_question_column_name = dataset_name.get_question_column_name()
    if model_processed_question_column_name == question_source_question_column_name:
        model_processed_question_column_name = f"{model_processed_question_column_name}_x"
        question_source_question_column_name = (
            f"{question_source_question_column_name}_y"
        )
    assert not (
        merged_processed_question_df[model_processed_question_column_name]
        != merged_processed_question_df[question_source_question_column_name]
    ).any(), "question_x and question_y do not match!"

    q_id_column_name = model_name.get_question_id_column_name()
    merged_processed_raw_df = pd.merge(
        model_processed_df,
        model_raw_df,
        left_on="question_id",
        right_on=q_id_column_name,
        how="outer",
    )
    assert (
        len(merged_processed_raw_df) == len(model_raw_df),
        f"Number of rows in merged_processed_question_df does not match {model_raw_path}",
    )
    model_processed_question_column_name = "question"
    model_raw_question_column_name = model_name.get_question_column_name(dataset_name)
    if model_processed_question_column_name == model_raw_question_column_name:
        model_processed_question_column_name = f"{model_processed_question_column_name}_x"
        model_raw_question_column_name = f"{model_raw_question_column_name}_y"
    assert not (
        merged_processed_raw_df[model_processed_question_column_name]
        != merged_processed_raw_df[model_raw_question_column_name]
    ).any(), "question_x and question_y do not match!"
