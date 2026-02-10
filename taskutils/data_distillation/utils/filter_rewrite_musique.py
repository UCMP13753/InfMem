import json
import os
import random
from pathlib import Path
from typing import List
from tqdm import tqdm
import re
import copy
from utils.envs import URL, API_KEY, RECURRENT_CHUNK_SIZE, RECURRENT_MAX_NEW, RECURRENT_MAX_CONTEXT_LEN
from transformers import AutoTokenizer
from utils.aio import get_async_client

import asyncio
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-4B", 
    use_fast=True
)

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



def after_think(solution_str):
    if "</think>" not in solution_str:
        return solution_str, None 
    final_answer = solution_str.split("</think>")[-1].strip()
    return solution_str.split("</think>")[0].strip(), final_answer


REWRITE_THINKING_BUDGET_TOKENS = 300
REWRITE_MAX_NEW_TOKENS = 420  # 输出长度上限，适当略大于 300，避免被截断
REWRITE_TEMPERATURE = 0.0
REWRITE_TOP_P = 1.0

def _serialize_history(messages: List[Dict[str, str]], max_turns: int = 6, max_chars: int = 20000) -> str:
    """
    把对话历史压缩成纯文本，避免 prompt 过长。
    默认取最近 max_turns 条消息（user+assistant 都算）。
    """
    tail = messages[-max_turns:] if len(messages) > max_turns else messages
    chunks = []
    for m in tail:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        chunks.append(f"{role.upper()}:\n{content}")
    s = "\n\n".join(chunks)
    
    return s

def _replace_last_occurrence(text: str, old: str, new: str) -> Optional[str]:
    """只替换最后一次出现的 old（更稳：通常 old_answer 在末尾）"""
    if not old:
        return None
    idx = text.rfind(old)
    if idx < 0:
        return None
    return text[:idx] + new + text[idx + len(old):]

def _should_rewrite_assistant(content: str) -> bool:
    """
    只在可能存在“长 thinking”的情况下触发重写，节省预算。
    你也可以改成永远 True。
    """
    if not content:
        return False
    # 常见思维标签/提示
    token_len = len(tokenizer.encode(content, add_special_tokens=False))
    pattern = "Considering the limited time, I must stop thinking now"
    if pattern.lower() in content.lower() or token_len > 512:
        return True
    return False






THINK_RE = re.compile(r"(?s)<think>(.*?)</think>")  # DOTALL, non-greedy

def split_think_and_memory(content: str):
    """
    返回 (prefix, think_text, memory_suffix)
    - prefix: <think> 前的任何内容（通常为空，但保留以防模板变化）
    - think_text: <think>...</think> 中间的内容
    - memory_suffix: </think> 后面的所有内容（updated memory + 其他），必须逐字保留
    """
    m = THINK_RE.search(content or "")
    if not m:
        return None, None, None
    prefix = content[:m.start()]
    think_text = m.group(1)
    memory_suffix = content[m.end():]
    return prefix, think_text, memory_suffix


async def async_rewrite_assistant_answer(
    session,
    model: str,
    history_messages: List[Dict[str, str]],
    old_assistant_content: str,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 320,   # 目标 <=300 tokens trajectory，给点余量
) -> str:
    """
    只重写 <think>trajectory</think>，并保证 </think> 后的 updated memory 完全不变。
    """
    parsed = split_think_and_memory(old_assistant_content)
    if parsed == (None, None, None):
        # 没有 <think> 标签就不改
        return old_assistant_content

    prefix, old_think, memory_suffix = parsed

    # 没有 memory_suffix 也没意义（你说格式是 </think> 后是 updated memory）
    if memory_suffix is None or memory_suffix.strip() == "":
        return old_assistant_content

    # 给模型的上下文（可选，但建议保留最近几轮避免乱写）
    history_text = _serialize_history(history_messages, max_turns=6, max_chars=6000)

    # 只让模型输出“新的 think 内容”，不要输出标签、不要输出 memory
    target_ratio = 0.33

    user_prompt = f"""You are rewriting ONLY the text inside <think>...</think>.

    Goal:
    - Compress the original think text to about {int(target_ratio*100)}% of its original length (roughly one third).
    - Preserve the key reasoning steps and decisions.

    Constraints:
    - Do NOT change meaning or final conclusion.
    - No backtracking or multiple alternative branches.
    - Do NOT output any memory or anything after </think>.
    - Output ONLY the rewritten think text (no <think> tags).
    - Keep it substantive: do not over-shorten to a few sentences. Aim for a similar structure but shorter.

    Conversation context (for understanding only):
    {history_text}

    Original think text:
    {old_think}
    """

    async with session.post(
        url=URL + "/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json=dict(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You rewrite reasoning trajectories to be short. "
                        "Return ONLY the rewritten think text, without tags, and never output memory."
                    )
                },
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        ),
    ) as resp:
        if resp.status != 200:
            return old_assistant_content
        data = await resp.json()
        new_think = (data["choices"][0]["message"]["content"] or "").strip()

    # 清理：防止模型把标签也输出了
    new_think = re.sub(r"(?s)</?think>", "", new_think).strip()

    # 拼回去：memory_suffix 必须逐字不变
    new_content = f"{prefix}<think>{new_think}</think>{memory_suffix}"

    # 保险校验：确保 suffix 完全一致（逐字不变）
    # 1) 再拆一次新内容
    p2, t2, suffix2 = split_think_and_memory(new_content)
    if suffix2 != memory_suffix:
        return old_assistant_content  # 出现任何偏差直接回退

    return new_content











async def async_rewrite_conversation_messages(
    messages: List[Dict[str, str]],
    rewrite_model: str,
) -> List[Dict[str, str]]:
    """
    对一段 messages（user+assistant 交错）：
    - 仅重写 assistant 的 content（只改 answer 部分，memory 保持不动）
    """
    session = await get_async_client()
    async with session:
        new_messages = []
        for t, m in enumerate(messages):
            if m.get("role") != "assistant":
                new_messages.append(m)
                continue

            content = m.get("content") or ""
            if not _should_rewrite_assistant(content):
                new_messages.append(m)
                
                return new_messages, False

            history = messages[:t]  # 给模型看的上下文
            try:
                new_content = await async_rewrite_assistant_answer(
                    session=session,
                    model=rewrite_model,
                    history_messages=history,
                    old_assistant_content=content,
                )
                new_m = dict(m)
                new_m["content"] = new_content
                new_messages.append(new_m)
            except Exception:
                # 出错则原样保留
                new_messages.append(m)
        return new_messages, True
    

def process_datasets_for_sft(
    input_jsonl_list: List[str],
    output_jsonl: str,
    filter_threshold: float = 1.0,
    score_key: str = "judge_em",
    seed: int = 42,
    rewrite_with_llm: bool = False,
    rewrite_model: Optional[str] = None,
):
    """
    Merge multiple JSONL datasets, filter, flatten to SFT.
    如果 rewrite_with_llm=True，会重写 conversation 中 assistant 的回答（thinking 更短，memory 不变）。
    """
    processed_data = []
    total_items = 0
    filtered_items = 0

    # Precompute total lines
    total_lines_all = 0
    for fp in input_jsonl_list:
        with open(fp, 'r', encoding='utf-8') as f:
            total_lines_all += sum(1 for _ in f)

    pbar = tqdm(total=total_lines_all, desc="Processing all files")

    for input_jsonl in input_jsonl_list:
        with open(input_jsonl, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # RULERS synthetic
        if "ruler_" in input_jsonl:
            lines = lines[:len(lines) // 1]
            score_key_local = "judge_em"
        else:
            score_key_local = "judge_em"
            if "squad" in input_jsonl:
                lines = lines[:len(lines) // 1]
            else:
                lines = lines[:len(lines) // 1]
        
        assistant_token_counts = []
        filtered_assistant_token_counts = []
        for line in lines:
            pbar.update(1)
            total_items += 1
            item = json.loads(line)

            if item.get(score_key_local, 0) < filter_threshold:
                filtered_items += 1
                continue

            metadata = {
                'judge_f1': item.get('judge_f1', 0.0),
                'judge_em': item.get('judge_em', 0.0),
                'judge_sub_em': item.get('judge_sub_em', 0.0),
                'answer': _to_list_of_str(item.get('answer')),
                'pred': item.get('pred'),
                'response': item.get('response'),
            }

            conversations = item.get('conversation', [])
            if not conversations:
                continue
            for conv_idx, single_conversation in enumerate(conversations):
                if isinstance(single_conversation, dict):
                    messages = [single_conversation]
                elif isinstance(single_conversation, list):
                    messages = single_conversation
                else:
                    continue

        #         for msg in messages:
        #             if msg.get("role") == "assistant":
        #                 text = msg.get("content", "")
        #                 if not text:
        #                     continue

        #                 token_len = len(tokenizer.encode(text, add_special_tokens=False))
        #                 if token_len < 1024:
        #                     assistant_token_counts.append(token_len)

        # avg_tokens = sum(assistant_token_counts) / max(len(assistant_token_counts), 1)

        # print(f"total number: {len(assistant_token_counts)}, assistant 平均 token 数: {avg_tokens:.2f}, min/max {min(assistant_token_counts)}, {max(assistant_token_counts):.2f}")
    #             # ========== 你原来的 placeholder：single_conversation = ***********************
                # 在这里重写 assistant 回复（可选）
                if rewrite_with_llm:
                    if not rewrite_model:
                        raise ValueError("rewrite_with_llm=True but rewrite_model is None")
                    # 注意：这里是同步函数里跑 async。若你在已有 event loop 内调用，需要改成 async 版本。
                    messages, rewritten = asyncio.run(async_rewrite_conversation_messages(messages, rewrite_model=rewrite_model))
                else:
                    rewritten = False
                token_len = 0
                # ========== end rewrite
                for msg in messages:
                    if msg.get("role") == "assistant":
                        text = msg.get("content", "")
                        if not text:
                            continue
                        
                        token_len = len(tokenizer.encode(text, add_special_tokens=False))
                        assistant_token_counts.append(token_len)
                
                if 5 < token_len <= 1536:

                    sft_item = {
                        'messages': messages,
                        'metadata': {
                            **metadata,
                            'conversation_idx': conv_idx,
                            'original_total_conversations': len(conversations),
                            "rewritten": rewritten,
                            'source_file': input_jsonl,
                            
                        }
                    }
                    filtered_assistant_token_counts.append(token_len)
                    processed_data.append(sft_item)
        avg_tokens = sum(assistant_token_counts) / max(len(assistant_token_counts), 1)
        
        avg_tokens_filtered = sum(filtered_assistant_token_counts) / max(len(filtered_assistant_token_counts), 1)
        print(f"total number: {len(assistant_token_counts)}, assistant 平均 token 数: {avg_tokens:.2f}, min/max {min(assistant_token_counts)}, {max(assistant_token_counts):.2f}")
        print(f"The dataset we select: total number: {len(filtered_assistant_token_counts)}, assistant 平均 token 数: {avg_tokens_filtered:.2f}, min/max {min(filtered_assistant_token_counts)}, {max(filtered_assistant_token_counts):.2f}")
        
    pbar.close()

    rng = random.Random(seed)
    rng.shuffle(processed_data)

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

# def process_datasets_for_sft(
#     input_jsonl_list: List[str],
#     output_jsonl: str,
#     filter_threshold: float = 1.0,
#     score_key: str = "judge_em",
#     seed: int = 42,
# ):
#     """
#     Merge multiple JSONL datasets, filter by `score_key` and `filter_threshold`,
#     rebalance per-item conversations to 1:1 (generate-memory : answer-question),
#     then shuffle (seed=42 by default) and write to a single output JSONL.

#     Args:
#         input_jsonl_list: List of input JSONL file paths
#         output_jsonl: Path to output JSONL file
#         filter_threshold: Keep only items where item[score_key] >= threshold
#         score_key: Which metric to filter by (default "judge_em")
#         seed: Shuffle seed (default 42)
#     """
#     processed_data = []
#     total_items = 0
#     filtered_items = 0

#     # Precompute total lines for a nice global progress bar
#     total_lines_all = 0
#     for fp in input_jsonl_list:
#         with open(fp, 'r', encoding='utf-8') as f:
#             total_lines_all += sum(1 for _ in f)

#     pbar = tqdm(total=total_lines_all, desc="Processing all files")
#     for input_jsonl in input_jsonl_list:
#             # Read all lines first
#             with open(input_jsonl, 'r', encoding='utf-8') as f:
#                 lines = f.readlines()
            
#             # RULERS synthetic
#             if "ruler_" in input_jsonl:
#                 # lines = lines[:len(lines) // 8]
#                 score_key = "judge_sub_em"
#             else:
#                 score_key = "judge_em"
#                 # SQUAD
#                 # if "squad" in input_jsonl:
#                 #     lines = lines[:len(lines) // 16]
#                 # # HOTPOT QA
#                 # else:
#                 #     lines = lines[:len(lines) // 8]
#             # Process selected lines
#             for line in lines:
#                 pbar.update(1)
#                 total_items += 1
#                 item = json.loads(line)

#                 # Filter: keep only items with score_key >= threshold
#                 if item.get(score_key, 0) < filter_threshold:
#                     filtered_items += 1
#                     continue

#                 metadata = {
#                     'judge_f1': item.get('judge_f1',0.0),
#                     'judge_em': item.get('judge_em',0.0),
#                     'judge_sub_em': item.get('judge_sub_em',0.0),
#                     'answer': _to_list_of_str(item.get('answer')),
#                     'pred': item.get('pred'),
#                     'response': item.get('response'),
#                 }

#                 conversations = item.get('conversation', [])
#                 if not conversations:
#                     # Skip if no conversation content
#                     continue

#                 # Assume last conversation is the question-answer one
#                 qa_conversation = conversations[-1]
#                 cleaned = clean_answer_keep_think(qa_conversation[-1]["content"])
#                 qa_conversation[-1]["content"] = cleaned
#                 # Rebalance: duplicate QA conversation to match other conversations
#                 # (your original logic kept (len(conversations)-1)/2)
#                 # num_generate_memory = int((len(conversations) - 1) / 2)
#                 # balanced_conversations = conversations.copy()

#                 # # for _ in range(num_generate_memory):
#                 # #     balanced_conversations.append(qa_conversation)
                                
#                 # for i in range(num_generate_memory):
#                 #     # make a deep copy so we don't modify original
#                 #     conv_copy = copy.deepcopy(qa_conversation)

#                 #     # 50% chance or deterministic half-split
#                 #     if i % 2 == 0:
#                 #         # Change instruction
#                 #         if len(conv_copy) > 0 and conv_copy[0]["role"] == "user":
#                 #             # Replace only the specific instruction sentence, preserving all other content
#                 #             pattern = r'Please answer the problem based on the previous memory and format your response as follows\s*"Therefore, the answer is \(insert answer here\)".'
#                 #             replacement = r'Please answer the problem based on the previous memory and put the answer in \\boxed{}.'
#                 #             conv_copy[0]["content"] = re.sub(pattern, replacement, conv_copy[0]["content"])

#                 #         # Change assistant output formatting
#                 #         if len(conv_copy) > 1 and conv_copy[1]["role"] == "assistant":
#                 #             old_answer = conv_copy[1]["content"]

#                 #             old = old_answer  # full original response including thinking

#                 #             # Find the LAST occurrence so earlier mentions aren't touched
#                 #             matches = list(re.finditer(r'(Therefore,\s*the\s*answer\s*is)(\s*:)?\s*(.*)$',
#                 #                                     old, flags=re.IGNORECASE | re.DOTALL))
#                 #             if matches:
#                 #                 m = matches[-1]
#                 #                 head = old[:m.start()]                   # everything before the final answer line (thinking etc.)
#                 #                 prefix = m.group(1)                      # "Therefore, the answer is"
#                 #                 answer_text = m.group(3).strip()         # the extracted answer payload
#                 #                 boxed_line = f"{prefix} \\boxed{{{answer_text}}}"
#                 #                 new_answer = head + boxed_line
#                 #             else:
#                 #                 # Fallback: no marker found—append a boxed line without destroying the original
#                 #                 new_answer = old + "\n\nTherefore, the answer is \\boxed{}"
#                 #             conv_copy[1]["content"] = new_answer

#                 #     # Append to new conversation list
#                 #     balanced_conversations.append(conv_copy)

#                 # # Flatten each conversation into an SFT sample
#                 # for conv_idx, single_conversation in enumerate(balanced_conversations):
#                 #     if isinstance(single_conversation, dict):
#                 #         messages = [single_conversation]
#                 #     elif isinstance(single_conversation, list):
#                 #         messages = single_conversation
#                 #     else:
#                 #         # Unexpected format; skip
#                 #         continue

#                 sft_item = {
#                     'messages': qa_conversation,
#                     'metadata': {
#                         **metadata,
#                         # 'conversation_idx': conv_idx,
#                         # 'total_conversations': len(balanced_conversations),
#                         # 'original_total_conversations': len(conversations),
#                         # 'is_balanced_copy': conv_idx >= len(conversations)  # mark duplicates
#                     }
#                 }
#                 processed_data.append(sft_item)
#     pbar.close()

#     # Shuffle with fixed seed
#     rng = random.Random(seed)
#     rng.shuffle(processed_data)

#     # Save
#     Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)
#     with open(output_jsonl, 'w', encoding='utf-8') as f:
#         for item in processed_data:
#             f.write(json.dumps(item, ensure_ascii=False) + '\n')

#     print(f"\nProcessing complete:")
#     print(f"  Total items read: {total_items}")
#     print(f"  Filtered out ({score_key} < {filter_threshold}): {filtered_items}")
#     print(f"  Final samples: {len(processed_data)}")
#     print(f"  Output saved to: {output_jsonl}")


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



def filter_by_len(input_file, output_file, seed: int = 42):
    processed_data = []
    total_items = 0
    filtered_items = 0

    # Precompute total lines
    total_lines_all = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines_all += sum(1 for _ in f)

    pbar = tqdm(total=total_lines_all, desc="Processing all files")

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    assistant_token_counts = []
    for line in lines:
        pbar.update(1)
        total_items += 1
        item = json.loads(line)

        messages = item["messages"]
        for msg in messages:
            if msg.get("role") == "assistant":
                text = msg.get("content", "")
                if not text:
                    continue

                token_len = len(tokenizer.encode(text, add_special_tokens=False))
                if token_len<=1436:
                    assistant_token_counts.append(token_len)
                    sft_item = item
                    processed_data.append(sft_item)
                    break

    avg_tokens = sum(assistant_token_counts) / max(len(assistant_token_counts), 1)

    print(f"total number: {len(assistant_token_counts)}, assistant average token: {avg_tokens:.2f}, min/max {min(assistant_token_counts)}, {max(assistant_token_counts):.2f}")
        
    pbar.close()

    rng = random.Random(seed)
    rng.shuffle(processed_data)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\nProcessing complete:")
    print(f"  Total items read: {total_items}")
    print(f"  Final samples: {len(processed_data)}")
    print(f"  Output saved to: {output_file}")





import json
from pathlib import Path

MARKER = "✿FUNCTION✿"

def last_msg_content(item) -> str:
    msgs = item.get("messages", [])
    if not msgs:
        return ""
    last = msgs[-1]
    return last.get("content", "") if isinstance(last, dict) else ""


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] JSON decode failed: {path}:{line_no} ({e})")
                continue



def main():
    file_to_shorter = "/mnt/nfs/datasets/mem_agent/retriever_sft/combined_retriever_sfted_hqa_sqd_musique_qwen3_32B_only_em_rewrite_to_shorter.jsonl"
    file_only_em   = "/mnt/nfs/datasets/mem_agent/retriever_sft/combined_retriever_sfted_hqa_sqd_musique_qwen3_32B_only_em.jsonl"

    output_file = "/mnt/nfs/datasets/mem_agent/retriever_sft/combined_retriever_sfted_hqa_sqd_musique_qwen3_32B_only_em_rewrite_mixed_trajectory.jsonl"

    kept = []

    # 1) For *_to_shorter: keep items whose last message DOES NOT contain ✿FUNCTION✿
    in1 = 0
    keep1 = 0
    for item in read_jsonl(file_to_shorter):
        in1 += 1
        c = last_msg_content(item)
        if MARKER not in c:
            kept.append(item)
            keep1 += 1

    # 2) For only_em.jsonl: keep items whose last message DOES contain ✿FUNCTION✿
    in2 = 0
    keep2 = 0
    for item in read_jsonl(file_only_em):
        in2 += 1
        c = last_msg_content(item)
        if MARKER in c:
            kept.append(item)
            keep2 += 1

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as w:
        for item in kept:
            w.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[DONE] wrote: {out_path}")
    print(f"  to_shorter: {keep1}/{in1} kept (NO {MARKER})")
    print(f"  only_em:    {keep2}/{in2} kept (HAS {MARKER})")
    print(f"  total:      {len(kept)}")


# if __name__ == "__main__":
#     input_file = ["/mnt/nfs/datasets/mem_agent/retriever_sft/MemAgent_hotpotqa_final_pair.jsonl","/mnt/nfs/datasets/mem_agent/retriever_sft/MemAgent_musique_final_pair_gt.jsonl","/mnt/nfs/datasets/mem_agent/retriever_sft/MemAgent_squad_final_pair.jsonl"]
#     output_file = "/mnt/nfs/datasets/mem_agent/retriever_sft/combined_retriever_sfted_hqa_sqd_musique_qwen3_32B_only_em_rewrite_to_shorter.jsonl"
#     # process_datasets_for_sft(
#     #     input_jsonl_list=input_file,
#     #     output_jsonl=output_file,
#     #     filter_threshold=1.0,     # keep items with judge_em >= 1.0 (adjust as needed)
#     #     score_key="judge_sub_em",     # or "judge_sub_em" if you prefer
#     #     seed=42,
#     #     rewrite_with_llm=True,
#     #     rewrite_model="next",
#     # )

#     # filter_by_len( "/mnt/nfs/datasets/mem_agent/distilled_musique_cleaned_shorten.jsonl",  "/mnt/nfs/datasets/mem_agent/distilled_musique_cleaned_shorten_1024.jsonl")
#     # Validate output
#     # validate_trl_format(output_file, 50)
#     main()

    