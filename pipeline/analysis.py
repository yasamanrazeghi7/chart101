from word2number import w2n

from evaluation_script.pipeline.constants import DatasetSplit


def convert_text_to_number(text):
    try:
        return w2n.word_to_num(text)
    except ValueError:
        # Handle the case where the text is not a number
        return None


def extract_answer(model_formatted_output):
    # Split the output to isolate the answer
    start_phrase = "The answer is "
    end_phrase = ". I hope the answer is correct"

    # Find the starting and ending indices of the answer
    start_idx = model_formatted_output.find(start_phrase) + len(start_phrase)
    end_idx = model_formatted_output.find(end_phrase)
    # print("start index is: ", start_idx)
    # print("end index is: ", end_idx)

    # Extract the answer using the indices
    if start_idx > len(start_phrase) - 1 and end_idx != -1:
        answer = model_formatted_output[start_idx:end_idx]
        # print("answer is: ", answer)
        # Remove trailing % sign if present
        answer = answer.replace("%", "")
        answer = answer.replace("approximately", "")
        answer = answer.replace("$", "")
        answer = answer.strip()
        # print("answer after stripping is: ", answer)
    # elif model_formatted_output.strip() == "I'm sorry, but I can't assist with that request.":
    else:
        return "ANSWER_NOT_PROVIDED"
    # else:
        # raise ValueError("Answer extraction failed.")

    return answer


def get_5_percent_range(row):
    if row["split"] == DatasetSplit.BAR.value:
        if "count" in row["question_type"]:
            return 0.05 * row["x_range"]
        return 0.05 * row["y_range"]
    elif row["split"] == DatasetSplit.PIE.value:
        return 5
    elif row["split"] == DatasetSplit.SCATTER.value:
        if "x_" in row["question_type"]:
            return 0.05 * row["x_range"]
        elif "y_" in row["question_type"]:
            return 0.05 * row["y_range"]
    return None

def get_accuracies(row):
    correct = False
    leniently_correct = False 
    liniently_correct_5_range = False
    model_answer = extract_answer(row["model_formatted_output"])
    correct_answer = row["correct_answer"]
    # print("model output is: ", row["model_formatted_output"])
    # print("extracted answer is: ", model_answer)
    # print("correct answer is: ", correct_answer)
    # print("correct answer type is: ", type(correct_answer))
    # print("answer type is: ", type(model_answer))
    try:
        if convert_text_to_number(model_answer) is not None:
            model_answer = float(convert_text_to_number(model_answer))
        else:
            model_answer = float(model_answer)
        # model_answer = float(convert_text_to_number(answer) if convert_text_to_number(answer) is not None else answer)
        correct_answer_float = float(correct_answer)
        l_bound, u_bound = correct_answer_float*0.95, correct_answer_float*1.05
        lower_bound = min(l_bound, u_bound)
        upper_bound = max(l_bound, u_bound)
        # print(f"The lower bound it:  {lower_bound} The upper bound is:  {upper_bound}")
        bound_5 = get_5_percent_range(row)
        if bound_5 is not None: 
            lower_bound_5 = min(correct_answer_float - bound_5, correct_answer_float + bound_5)
            upper_bound_5 = max(correct_answer_float - bound_5, correct_answer_float + bound_5)
            # print(f"The lower 5 bound it:  {lower_bound_5} The upper 5 bound is:  {upper_bound_5}")
        else:
            lower_bound_5, upper_bound_5 = lower_bound, upper_bound
        
        correct = bool(model_answer == correct_answer_float)
        leniently_correct = bool(lower_bound <= model_answer <= upper_bound)
        liniently_correct_5_range = bool(lower_bound_5 <= model_answer <= upper_bound_5)
    except ValueError:
        # print("--->", model_answer, type(model_answer), correct_answer, type(correct_answer))
        if type(model_answer) == str and type(correct_answer) == str:
            if model_answer.lower() == correct_answer.lower():
                correct = True
                leniently_correct = True
                liniently_correct_5_range = True
        # print(f"returning {correct} {leniently_correct} {liniently_correct_5_range} because of value errors can not convert")
        # Handle cases where conversion to float fails
        pass
    # print("returning correct and leniently correct and 5 range liniently correct as ", correct, leniently_correct, liniently_correct_5_range)
    return (correct, leniently_correct, liniently_correct_5_range)