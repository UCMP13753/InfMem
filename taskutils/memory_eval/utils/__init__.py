import os
import re
import string
from collections import Counter

def get_client():
    import openai
    # client = openai.OpenAI(
    #     api_key=os.getenv('VERIFIER_API'),
    #     base_url=os.getenv('VERIFIER_URL'),
    # )

    api_version = "2024-03-01-preview"
    return openai.AzureOpenAI(
        azure_endpoint=os.getenv('VERIFIER_URL'),
        api_version=api_version,
        api_key=os.getenv('VERIFIER_API'),
    )

def extract_solution(solution_str):
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
  
    # Extract final answer using XML-style tags
    if "</think>" not in solution_str:
        if not os.environ.get("FORCE_THINK"):
            return solution_str, solution_str
        else:
            print("[Error] No valid answer tags found")
            return None, solution_str 
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

def update_answer(metrics, prediction, gold):
    em = exact_match_score(prediction, gold)
    subem = sub_exact_match_score(prediction, gold)

    f1, prec, recall = f1_score(prediction, gold)
    metrics['sub_em'] += subem
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    metrics['total_num'] += 1
    return em, prec, recall

#### add 
# def update_answer_traj(metrics, conv_traj, gold):
#     preserve_in_mem = True
#     preserve_in_think = True
#     start  = False
#     for i in range(len(conv_traj)-1):
#         try:
#             res = conv_traj[i]
#             mem = res[1].get('content', '').strip()
#             final_mem, smem_str = extract_solution(mem)
#             in_mem = sub_exact_match_score(final_mem, gold)
#             in_think = sub_exact_match_score(smem_str, gold)
#             if in_think or in_mem:
#                 if not start:
#                     start = True
#             preserve_in_mem = not (start & (not in_mem)) & preserve_in_mem
#             preserve_in_think = not (start & (not in_mem)) & preserve_in_mem

                    
#         except Exception as e:
#             print(e)
#             preserve_in_mem = 0
#             preserve_in_think = 0
    
    
#     metrics['preserve_memory'] += float(preserve_in_mem) if start else 0.0
#     metrics['preserve_memory_in_think'] += float(preserve_in_think) if start else 0.0
#     metrics['total_num'] += 1
    
#     return preserve_in_mem, preserve_in_think

def update_answer_traj(metrics, conv_traj, gold):
    found_in_mem = False
    found_in_think =False
    preserve_in_mem = False
    preserve_in_think = False
    for i in range(len(conv_traj)-1):
        try:
            res = conv_traj[i]
            q = res[0].get('content', '').strip()
            mem = res[1].get('content', '').strip()
            if "✿FUNCTION✿" not in q:
                final_mem, smem_str = extract_solution(mem)
                in_mem = sub_exact_match_score(final_mem, gold)
                in_think = sub_exact_match_score(smem_str, gold)
                
                found_in_mem = found_in_mem or in_mem
                found_in_think = found_in_think or in_think
                preserve_in_mem = in_mem
                preserve_in_think = in_think

                    
        except Exception as e:
            print(e)
            preserve_in_mem = 0
            preserve_in_think = 0
    
    metrics['found_in_memory'] += float(found_in_mem)
    metrics['found_in_think'] += float(found_in_think)
    metrics['preserve_memory'] += float(preserve_in_mem)
    metrics['preserve_memory_in_think'] += float(preserve_in_think)
    metrics['total_num'] += 1
    
    return preserve_in_mem, preserve_in_think

#### end


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

def extract_boxed_answer(string):
    s = last_boxed_only_string(string)
    if s is None:
        return None
    else:
        if "\\boxed " in s:
            left = "\\boxed "
            assert s[:len(left)] == left
            return s[len(left):]

        left = "\\boxed{"

        assert s[:len(left)] == left
        assert s[-1] == "}"

        return s[len(left):-1]