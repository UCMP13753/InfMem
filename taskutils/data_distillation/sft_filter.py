import json
import os
import random
from pathlib import Path
from typing import List
from tqdm import tqdm
import re
import copy

def clean_answer_keep_think(content):
    """
    保留 <think>...</think> 原样，
    只清洗其后部分的 () [] {} "" '' “ ” ‘ ’ 等符号。
    """
    # 找 think block
    m = re.search(r'(<think>.*?</think>)', content, flags=re.DOTALL)
    
    if not m:
        # 没有思考块 => 全部清洗
        return re.sub(r'[\(\)\[\]\{\}]', '', content)
    
    think_block = m.group(1)
    after_think = content[m.end():]  # think 后面的部分
    
    # 清洗括号/引号
    after_clean = re.sub(r'[\(\)\[\]\{\}]', '', after_think)

    return think_block + after_clean

def normalize_answer_text(ans: str):
    if ans is None:
        return ""

    ans = ans.strip()

    # remove starting and ending quotes/stars/backticks in any number
    ans = re.sub(r'^[\'"*`]+', '', ans)
    ans = re.sub(r'[\'"*`]+$', '', ans)

    # remove trailing punctuation garbage such as */"}).,;:
    ans = re.sub(r'[\s\'"*\)}\]\.]+$', '', ans)

    # collapse whitespace
    ans = re.sub(r'\s+', ' ', ans).strip()

    return ans

def _to_list_of_str(x):
    """Normalize answers/pred/response to a list[str]."""
    if x is None:
        return []
    if isinstance(x, list):
        # stringify defensively
        return [str(v) if v is not None else "" for v in x]
    # string or anything else -> single-element list
    return [str(x)]
def process_datasets_for_sft(
    input_jsonl_list: List[str],
    output_jsonl: str,
    filter_threshold: float = 1.0,
    score_key: str = "judge_em",
    seed: int = 42,
):
    """
    Merge multiple JSONL datasets, filter by `score_key` and `filter_threshold`,
    rebalance per-item conversations to 1:1 (generate-memory : answer-question),
    then shuffle (seed=42 by default) and write to a single output JSONL.

    Args:
        input_jsonl_list: List of input JSONL file paths
        output_jsonl: Path to output JSONL file
        filter_threshold: Keep only items where item[score_key] >= threshold
        score_key: Which metric to filter by (default "judge_em")
        seed: Shuffle seed (default 42)
    """
    processed_data = []
    total_items = 0
    filtered_items = 0

    # Precompute total lines for a nice global progress bar
    total_lines_all = 0
    for fp in input_jsonl_list:
        with open(fp, 'r', encoding='utf-8') as f:
            total_lines_all += sum(1 for _ in f)

    pbar = tqdm(total=total_lines_all, desc="Processing all files")
    for input_jsonl in input_jsonl_list:
            # Read all lines first
            with open(input_jsonl, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # RULERS synthetic
            if "ruler_" in input_jsonl:
                lines = lines[:len(lines) // 8]
                score_key = "judge_sub_em"
            else:
                score_key = "judge_em"
                # SQUAD
                if "squad" in input_jsonl:
                    lines = lines[:len(lines) // 1]
                # HOTPOT QA
                else:
                    lines = lines[:len(lines) // 1]
            # Process selected lines
            for line in lines:
                pbar.update(1)
                total_items += 1
                item = json.loads(line)

                # Filter: keep only items with score_key >= threshold
                if item.get(score_key, 0) < filter_threshold:
                    filtered_items += 1
                    continue

                metadata = {
                    'judge_f1': item.get('judge_f1',0.0),
                    'judge_em': item.get('judge_em',0.0),
                    'judge_sub_em': item.get('judge_sub_em',0.0),
                    'answer': _to_list_of_str(item.get('answer')),
                    'pred': item.get('pred'),
                    'response': item.get('response'),
                }

                conversations = item.get('conversation', [])
                if not conversations:
                    # Skip if no conversation content
                    continue

                # Assume last conversation is the question-answer one
                qa_conversation = conversations[-1]
                # Rebalance: duplicate QA conversation to match other conversations
                # (your original logic kept (len(conversations)-1)/2)
                num_generate_memory = int((len(conversations) - 1) / 2)
                balanced_conversations = conversations.copy()

                # for _ in range(num_generate_memory):
                #     balanced_conversations.append(qa_conversation)
                                
                for i in range(num_generate_memory):
                    # make a deep copy so we don't modify original
                    conv_copy = copy.deepcopy(qa_conversation)

                    use_boxed = (i % 2 == 0)  # even -> boxed, odd -> no boxed
                    # ----------------------
                    # 1. Modify user prompt only if using boxed mode
                    # ----------------------
                    if use_boxed and len(conv_copy) > 0 and conv_copy[0]["role"] == "user":
                        pattern = r'Please answer the problem based on the previous memory and format your response as follows\s*"Therefore, the answer is \(insert answer here\)".'
                        replacement = 'Please answer the problem based on the previous memory and put the answer in \\boxed{}.'
                        conv_copy[0]["content"] = re.sub(pattern, replacement, conv_copy[0]["content"])

                    # ----------------------
                    # 2. Normalize assistant output BOTH in boxed / non-boxed mode
                    # ----------------------
                    if len(conv_copy) > 1 and conv_copy[1]["role"] == "assistant":
                        old = conv_copy[1]["content"]

                        # --- NEW: Only operate on text AFTER </think> ---
                        parts = old.rsplit("</think>", 1)
                        if len(parts) == 2:
                            thinking_part, after_think = parts[0] + "</think>", parts[1]
                        else:
                            thinking_part, after_think = "", old  # no <think>, safe fallback

                        # Find LAST "Therefore, the answer is" **ONLY inside after_think**
                        matches = list(re.finditer(
                            r'(Therefore,\s*the\s*answer\s*is)(\s*:)?\s*(.*)$',
                            after_think, flags=re.IGNORECASE | re.DOTALL
                        ))

                        if matches:
                            m = matches[-1]
                            head = after_think[:m.start()]     # content after </think>, before final answer line
                            prefix = m.group(1)
                            raw_answer = m.group(3).strip()

                            normalized = normalize_answer_text(raw_answer)

                            if use_boxed:
                                last_line = f"{prefix} \\boxed{{{normalized}}}."
                            else:
                                last_line = f"{prefix} {normalized}."

                            new_after = head + last_line

                        else:
                            # No "Therefore" found after </think>
                            normalized = normalize_answer_text(after_think.strip().splitlines()[-1])
                            if use_boxed:
                                new_after = after_think + f"\n\nTherefore, the answer is \\boxed{{{normalized}}}."
                            else:
                                new_after = after_think + f"\n\nTherefore, the answer is {normalized}."

                        # merge thinking + processed tail
                        conv_copy[1]["content"] = thinking_part + new_after

                    # Append to new conversation list
                    balanced_conversations.append(conv_copy)

                # Flatten each conversation into an SFT sample
                for conv_idx, single_conversation in enumerate(balanced_conversations):
                    if isinstance(single_conversation, dict):
                        messages = [single_conversation]
                    elif isinstance(single_conversation, list):
                        messages = single_conversation
                    else:
                        # Unexpected format; skip
                        continue

                    sft_item = {
                        'messages': messages,
                        'metadata': {
                            **metadata,
                            'conversation_idx': conv_idx,
                            'total_conversations': len(balanced_conversations),
                            'original_total_conversations': len(conversations),
                            'is_balanced_copy': conv_idx >= len(conversations)  # mark duplicates
                        }
                    }
                    processed_data.append(sft_item)
    pbar.close()

    # Shuffle with fixed seed
    rng = random.Random(seed)
    rng.shuffle(processed_data)

    # Save
    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\nProcessing complete:")
    print(f"  Total items read: {total_items}")
    print(f"  Filtered out ({score_key} < {filter_threshold}): {filtered_items}")
    print(f"  Final samples: {len(processed_data)}")
    print(f"  Output saved to: {output_jsonl}")


def validate_trl_format(jsonl_path: str, num_samples: int = 3):
    """Validate and display sample data in TRL format"""
    print(f"\n=== Validating TRL format ===")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            item = json.loads(line)
            if len(item.get('messages', [])) > 1:
                print(f"  Last message: {item['messages'][-1]}")



def validate_trl_format(jsonl_path: str, num_samples: int = 3):
    """Validate and display sample data in TRL format"""
    print(f"\n=== Validating TRL format ===")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            item = json.loads(line)
            if len(item.get('messages', [])) > 1:
                print(f"  Last message: {item['messages'][-1]}")

