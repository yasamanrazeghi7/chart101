from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum
from typing import List, Union

import pandas as pd

from evaluation_script.pipeline.constants import (
    SEEDS,
    DatasetName,
    DatasetSplit,
    ModelName,
    ProcessedModelResultDataTypes,
    SyntheticProcessedModelResult,
)
from evaluation_script.pipeline.utils import (
    ask_gpt4,
    get_dataset_name,
    get_model_raw_path,
    get_processed_model_path,
    get_question_source_path,
    is_iterable,
)


class ModelProcessor(ABC):
    def __init__(self, model_name: ModelName):
        self.model_name = model_name

    @abstractmethod
    def _format_model_output(self, question: str, model_raw_output: str) -> str:
        pass

    def _get_raw_model_output(self, original_model_raw_output) -> str:
        return original_model_raw_output

    def process_single_run(self, split: DatasetSplit, seed: int):
        dataset_name = get_dataset_name(split)
        model_name = self.model_name
        dtypes = ProcessedModelResultDataTypes
        if dataset_name == DatasetName.SYNTHETIC:
            dtypes = SyntheticProcessedModelResult
        data = {column: pd.Series(dtype=typ) for column, typ in dtypes.items()}
        df = pd.DataFrame(data)
        model_raw_path = get_model_raw_path(model_name, split, seed)
        model_raw_df = pd.read_json(model_raw_path, lines=True)
        question_source_path = get_question_source_path(split)
        questions_df = pd.read_json(question_source_path, lines=True)
        df["model_name"] = [model_name.value] * len(model_raw_df)
        df["split"] = [split.value] * len(model_raw_df)
        df["seed"] = [seed] * len(model_raw_df)
        df["question_id"] = model_raw_df[model_name.get_question_id_column_name()]
        question_column_name = model_name.get_question_column_name(dataset_name)
        model_column_name = model_name.get_model_output_column_name(dataset_name)
        df["question"] = model_raw_df[question_column_name]
        df["model_raw_output"] = self._get_raw_model_output(model_raw_df[model_column_name])
        df["correct_answer"] = questions_df[dataset_name.get_correct_answer_column()]
        df["figure_id"] = questions_df[dataset_name.get_figure_id_column()]
        if dataset_name == DatasetName.SYNTHETIC:
            df["question_type"] = questions_df["question_type"]
            df["x_range"] = questions_df["x_range"]
            df["y_range"] = questions_df["y_range"]
        df["model_formatted_output"] = model_raw_df.apply(
            lambda row: self._format_model_output(
                row[question_column_name],
                self._get_raw_model_output(row[model_column_name]),
            ).strip(),
            axis=1,
        )

        model_processed_path = get_processed_model_path(self.model_name, split, seed)
        df.to_json(model_processed_path, orient="records", lines=True)


class GPT4Processor(ModelProcessor):
    def __init__(self):
        super().__init__(model_name=ModelName.GPT4)

    def _format_model_output(self, question: str, model_raw_output: str) -> str:
        return model_raw_output

class GeminiProcessor(ModelProcessor):
    def __init__(self):
        super().__init__(model_name=ModelName.GEMINI)

    def _format_model_output(self, question: str, model_raw_output: str) -> str:
        return model_raw_output

class PaliProcessor(ModelProcessor):
    def __init__(self):
        super().__init__(model_name=ModelName.PALI)

    def _get_raw_model_output(self, original_model_raw_output: Union[str, List[str]]) -> str:
        # assert is_iterable(original_model_raw_output), f"Pali expects to get a list of outputs, got: {type(original_model_raw_output)=} {original_model_raw_output=}"
        def _get_res(x):
            if len(x) == 0:
                return ""
            return x[0]
        if isinstance(original_model_raw_output, list):
            return _get_res(original_model_raw_output)
        elif isinstance(original_model_raw_output, str):
            return original_model_raw_output
        return original_model_raw_output.apply(lambda x: _get_res(x))

    def _format_model_output(self, question: str, model_raw_output: str) -> str:
        # Few-shot prompt with examples and the task for GPT-4
        prompt = f"""
        Extract the concise answer from the model's response as shown in the examples below, make sure the answer is in this format:
        "The answer is ANS. I hope the answer is correct."
        
        Example 1:
        Question: "How many food items are shown in the bar graph?"
        Model Answer: "<extra_id_0> 0"
        Extracted Answer: The answer is 0. I hope the answer is correct.

        Example 2:
        Question: "How many bars are in the figure?"
        Model Answer: "<extra_id_0> There are three bars in the figure."
        Extracted Answer: The answer is three. I hope the answer is correct.

        Example 3:
        Question: "Find missing data of the sequence 24, _ ,32, 33, 42?"
        Model Answer: "<extra_id_0> 33"
        Extracted Answer: The answer is 33. I hope the answer is correct.

        Example 4:
        Question: "Which country has highest secondary graduation rate in 2018?"
        Model Answer: "<extra_id_0> Italy"
        Extracted Answer: The answer is Italy. I hope the answer is correct.

        Your Task:
        Given the question and model answer below, extract the concise answer.

        Question: "{question}"
        Model Answer: "{model_raw_output}"
        Extracted Answer:
        """
        
        # Send the prompt to GPT-4 and get the extracted answer
        formatted_answer = ask_gpt4(prompt)
        return formatted_answer


class CogVLMProcessor(ModelProcessor):
    def __init__(self):
        super().__init__(model_name=ModelName.COGVLM)

    def _format_model_output(self, question: str, model_raw_output: str) -> str:
                # Few-shot prompt with examples and the task for GPT-4
        prompt = f"""
        Extract the concise answer from the model's response as shown in the examples below, make sure the answer is in this format:
        "The answer is ANS. I hope the answer is correct."
        
        Example 1:
        Question: "How many food items are shown in the bar graph?"
        Model Answer: "There are 11 food items shown in the bar graph.</s>"
        Extracted Answer: The answer is 11. I hope the answer is correct.

        Example 2:
        Question: "Is the percentage value of "STEM" segment 52?"
        Model Answer: "Yes, the percentage value of the "STEM" segment is 52.</s>"
        Extracted Answer: The answer is Yes. I hope the answer is correct.

        Example 3:
        Question: "What is the colour of India in the graph?"
        Model Answer: "India is represented by the color orange in the graph.</s>"
        Extracted Answer: The answer is orange. I hope the answer is correct.

        Your Task:
        Given the question and model answer below, extract the concise answer.

        Question: "{question}"
        Model Answer: "{model_raw_output}"
        Extracted Answer:
        """
        
        # Send the prompt to GPT-4 and get the extracted answer
        formatted_answer = ask_gpt4(prompt)
        return formatted_answer

class ChartLlamaProcessor(ModelProcessor):
    def __init__(self):
        super().__init__(model_name=ModelName.CHART_LLAMA)

    def _format_model_output(self, question: str, model_raw_output: str) -> str:
        # Few-shot prompt with examples and the task for GPT-4
        prompt = f"""
        Extract the concise answer from the model's response as shown in the examples below, make sure the answer is in this format:
        "The answer is ANS. I hope the answer is correct."
        
        Example 1:
        Question: "How many food items are shown in the bar graph?"
        Model Answer: "There are 11 food items shown in the bar graph.</s>"
        Extracted Answer: The answer is 11. I hope the answer is correct.

        Example 2:
        Question: "Is the percentage value of "STEM" segment 52?"
        Model Answer: "Yes, the percentage value of the "STEM" segment is 52.</s>"
        Extracted Answer: The answer is Yes. I hope the answer is correct.

        Example 3:
        Question: "What is the colour of India in the graph?"
        Model Answer: "India is represented by the color orange in the graph.</s>"
        Extracted Answer: The answer is orange. I hope the answer is correct.

        Your Task:
        Given the question and model answer below, extract the concise answer.

        Question: "{question}"
        Model Answer: "{model_raw_output}"
        Extracted Answer:
        """
        
        # Send the prompt to GPT-4 and get the extracted answer
        formatted_answer = ask_gpt4(prompt)
        return formatted_answer


def get_processor(model_name: ModelName) -> ModelProcessor:
    if model_name == ModelName.GPT4:
        return GPT4Processor()
    elif model_name == ModelName.GEMINI:
        return GeminiProcessor()
    elif model_name == ModelName.PALI:
        return PaliProcessor()
    elif model_name == ModelName.COGVLM:
        return CogVLMProcessor()
    elif model_name == ModelName.CHART_LLAMA:
        return ChartLlamaProcessor()
    raise ValueError(f"Invalid model name: {model_name}")

