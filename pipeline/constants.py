from enum import Enum
from pathlib import Path
from typing import List

BASE_DATASET_PATH = Path("/Users/yasaman/Documents/PhD/figure_understanding/dataset/")
CONSODLIATED_DATASET_PATH = BASE_DATASET_PATH / "Consolidated"
OPENAI_API_KEY_PATH = (
    "/Users/yasaman/Documents/PhD/figure_understanding/evaluation_script/models/key.txt"
)
SEEDS = tuple(range(5))


class DatasetName(Enum):
    SYNTHETIC = "Synthetic"
    CHART_QA = "ChartQA"

    def get_required_source_columns(self) -> List[str]:
        required_columns = [
            "q_id",
            self.get_question_column_name(),
            self.get_correct_answer_column(),
            self.get_figure_id_column(),
        ]
        if self == DatasetName.SYNTHETIC:
            return required_columns + ["question_type", "x_range", "y_range"]
        elif self == DatasetName.CHART_QA:
            return [self.get_question_column_name()]
        raise ValueError(f"Invalid dataset name: {self}")

    def get_question_column_name(self):
        if self == DatasetName.SYNTHETIC:
            return "question"
        elif self == DatasetName.CHART_QA:
            return "query"
        return "UNKNOWN"

    def get_correct_answer_column(self) -> str:
        if self == DatasetName.SYNTHETIC:
            return "gold_answer"
        elif self == DatasetName.CHART_QA:
            return "label"

    def get_figure_id_column(self) -> str:
        if self == DatasetName.SYNTHETIC:
            return "figure_id"
        elif self == DatasetName.CHART_QA:
            return "imgname"


class DatasetSplit(Enum):
    BAR = "bar"
    SCATTER = "scatter"
    PIE = "pie"
    ADDITIONAL = "additional"
    ORIGINAL = "original"


class ModelName(Enum):
    GPT4 = "GPT4"
    GEMINI = "GeminiPro"
    CHART_LLAMA = "ChartLlama"
    COGVLM = "CogVLM"
    PALI = "Pali"

    def get_required_columns(self, dataset_name: DatasetName) -> List[str]:
        required_columns = [
            self.get_question_id_column_name(),
            self.get_question_column_name(dataset_name),
            self.get_model_output_column_name(dataset_name),
        ]
        return required_columns

    def get_question_id_column_name(self) -> str:
        if self in [ModelName.GPT4, ModelName.GEMINI, ModelName.COGVLM, ModelName.PALI]:
            return "q_id"
        elif self == ModelName.CHART_LLAMA:
            return "question_id"
        return "UNKNOWN"

    def get_question_column_name(self, dataset_name: DatasetName) -> str:
        if self in [ModelName.GPT4, ModelName.GEMINI]:
            return "question"
        elif self == ModelName.CHART_LLAMA:
            return "prompt"
        elif self in [ModelName.COGVLM, ModelName.PALI]:
            if dataset_name == DatasetName.SYNTHETIC:
                return "question"
            elif dataset_name == DatasetName.CHART_QA:
                return "query"
        raise ValueError(f"Invalid model name: {self}")

    def get_model_output_column_name(self, dataset_name: DatasetName) -> str:
        if self in [ModelName.GPT4, ModelName.GEMINI, ModelName.COGVLM]:
            return "model_output"
        elif self == ModelName.CHART_LLAMA:
            return "text"
        elif self == ModelName.PALI:
            if dataset_name == DatasetName.SYNTHETIC:
                return "model_answer"
            else:
                return "model_output"
        raise ValueError(f"Invalid model name: {self}")


ProcessedModelResultDataTypes = {
    "model_name": "string",
    "split": "string",
    "seed": "int32",
    "question_id": "int32",
    "question": "string",
    "correct_answer": "string",
    "model_raw_output": "string",
    "model_formatted_output": "string",
    "figure_id": "string",
}
SyntheticProcessedModelResult = dict(ProcessedModelResultDataTypes)
SyntheticProcessedModelResult.update(
    {
        "question_type": "string",
        "x_range": "float64",
        "y_range": "float64",
    }
)
