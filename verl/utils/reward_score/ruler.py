# def compute_score(solution_str, ground_truth: list) -> float: 
#     def compute_score_single(solution_str, ground_truth) -> float:
#         ground_truth = ground_truth.lower()

#         retval = 0.
#         try:
#             string_in_last_boxed = last_boxed_only_string(solution_str)
#             if string_in_last_boxed is not None:
#                 answer = remove_boxed(string_in_last_boxed)
#                 if is_equiv(answer, ground_truth):
#                     retval = 1.
#         except Exception as e:
#             print(e)
#         return retval
#     solution_str = solution_str[-300:].lower()
#     return max(compute_score_single(solution_str, gt) for gt in ground_truth)
import os
import re
import string
from collections import Counter


def compute_score_ruler(solution_str, ground_truth: list) -> float: 
    retval = 0.
    try:
        
        solution_str, _ = extract_solution(solution_str)
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            answer = remove_boxed(string_in_last_boxed)
            retval = string_match_all(answer, ground_truth)
    except Exception as e:
        print(e)
    return retval
    
    

def compute_score_qa(solution_str, ground_truth: list) -> float: 
    ground_truth = ground_truth[0].lower()

    retval = 0.
    try:
        solution_str, _ = extract_solution(solution_str)
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            answer = remove_boxed(string_in_last_boxed)
            retval = exact_match_score(answer, ground_truth)
    except Exception as e:
        print(e)
    return retval
    
    # return max(compute_score_single(solution_str, gt) for gt in ground_truth)
# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    # remove spaces
    string = string.replace(" ", "")

    return string

### From RULER
def string_match_all(pred, ref):
    return sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref)




# def calc_metrics(predictions, goldens):
#     assert len(predictions) == len(goldens)
#     metrics = {'sub_em': 0, 'total_num': 0}
#     for pred, gold in zip(predictions, goldens):
#         metrics['sub_em'] += string_match_all(pred, gold)
#     metrics['total_num'] = len(goldens)
#     for k, _ in metrics.items():
#         if k == 'total_num':
#             continue
#         metrics[k] = round((metrics[k]/metrics['total_num']), 2)
#     return metrics

# def calc_qa_metrics(predictions, goldens):
#     assert len(predictions) == len(goldens)
#     metrics = {'f1': 0, 'prec': 0, 'recall': 0, 'em': 0, 'sub_em': 0, 'total_num': 0}
#     for pred, gold in zip(predictions, goldens):
#         update_answer(metrics, pred, gold)
#     for k, _ in metrics.items():
#         if k == 'total_num':
#             continue
#         metrics[k] = round((metrics[k]/metrics['total_num']), 2)
#     return metrics


def extract_solution(solution_str):
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
  
    # Extract final answer using XML-style tags
    if "</think>" not in solution_str:
        return solution_str, solution_str
        
    final_answer = solution_str.split("</think>")[-1].strip()
    return final_answer, solution_str


def extract_answer(response):
    response = response.replace('*', '')

    if "the answer is" in response:
        ans = response.rsplit("the answer is", 1)[-1].strip().replace("<｜Assistant｜>", '').replace("<｜end▁of▁sentence｜>", '').strip().strip('.').strip()
    else:
        ans = None

    return ans

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def sub_exact_match_score(prediction, ground_truth):
    ground_truth = normalize_answer(ground_truth)
    prediction = normalize_answer(prediction) 
    return (ground_truth in prediction) or (prediction in ground_truth)

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))