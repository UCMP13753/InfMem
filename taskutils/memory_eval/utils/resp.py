import json
import nltk
import numpy as np
import traceback
import re
from collections import defaultdict

from rank_bm25 import BM25Okapi
from openai import AsyncOpenAI

from .aio import get_async_client
from utils import extract_solution
from .envs import URL, API_KEY, MAX_INPUT_LEN, MAX_OUTPUT_LEN

nltk.download('punkt', quiet=True)

ENABLE_THINK = True
RETRIEVE_CHUNK_SIZE = 2000 
RETRIEVE_CHUNK_OVERLAP = 400
TOP_K = 6
MAX_ITERATIONS = 5  
THINKING_BUDGET = 1024
BM25_MAX_PER_CHUNK = 2
BM25_DEDUP_JACCARD = 0.8
BM25_CAND_MULTIPLIER = 6

# ================= ReSP Prompt Template =================

PROMPT_JUDGE = """Judging based solely on the current known information and without allowing for inference, are you able to completely and accurately respond to the question {question}? \nKnown information: {combined_memory}.
\nIf you can, please reply with "Yes" directly; if you cannot and need more information, please reply with "No" directly."""

PROMPT_REASONER = """You serve as an intelligent assistant, adept at facilitating users through complex, multi-hop reasoning across multiple documents. Please understand the information gap between the currently known information and the target problem.Your task is to generate one thought in the form of question for next retrieval step directly. DON’T generate the whole thoughts at once!\n DON’T generate thought which has been retrieved.\n [Known information]: {combined_memory}\n[Targetquestion]:{question}\n[YouThought]:"""

PROMPT_GLOBAL_EVIDENCE = """Passages: {docs}\nYour job is to act as a professional writer. You will write a good quality passage that can support the given prediction about the question only based on the information in the provided supporting passages. Now, let’s start. After you write, please write [DONE] to indicate you are done. Do not write a prefix (e.g., "Response:") while writing a passage.\nQuestion:{question}\nPassage:"""

PROMPT_LOCAL_PATHWAY = """Judging based solely on the current known information and without allowing for inference, are you able to respond completely and accurately to the question {sub_question}? \nKnown information: {combined_memory}.
If yes, please reply with "Yes", followed by an accurate response to the question {sub_question}, without restating the question; if no, please reply with "No" directly."""

PROMPT_GENERATOR = """Answer the question based on the given reference.\nThe following are given reference: {combined_memory}\nQuestion: {question} Please answer the problem based on the given reference and format your response as follows "Therefore, the answer is (insert answer here)"""

# ================= BM25 =================

def tokenize_legacy(text: str):
    return text.lower().split()

def tokenize(text: str):
    return tokenize_legacy(text)

def tokenize_mixed(text: str):
    lower_text = text.lower()
    tokens = re.findall(r"[a-z0-9_]+", lower_text)
    zh_chars = [ch for ch in lower_text if '\u4e00' <= ch <= '\u9fff']
    if zh_chars:
        tokens.extend(zh_chars)
        tokens.extend(
            zh_chars[i] + zh_chars[i + 1]
            for i in range(len(zh_chars) - 1)
        )
    return tokens

def _jaccard(tokens_a, tokens_b):
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    union_size = len(set_a | set_b)
    if union_size == 0:
        return 0.0
    return len(set_a & set_b) / union_size

def _build_retrieve_corpus_enhanced(input_ids, tokenizer):
    chunk_size = RETRIEVE_CHUNK_SIZE
    overlap = min(RETRIEVE_CHUNK_OVERLAP, chunk_size - 1)
    step = max(1, chunk_size - overlap)

    retrieve_docs = []
    retrieve_meta = []
    p = 0
    while p < len(input_ids):
        rs = p
        re = min(len(input_ids), rs + chunk_size)
        sub_ids = input_ids[rs:re]
        retrieve_docs.append(tokenizer.decode(sub_ids))
        retrieve_meta.append({
            "source_idx": rs // chunk_size,
            "start": rs,
            "end": re,
        })
        if re >= len(input_ids):
            break
        p = min(len(input_ids), rs + step)
    return retrieve_docs, retrieve_meta

def bm25_retrieve_for_recurrent_chunk(
    query: str,
    bm25_model: BM25Okapi,
    retrieve_docs,
    top_k: int,
    retrieve_doc_tokens=None,
    retrieve_meta=None,
):
    q_tokens = tokenize_mixed(query)
    if not q_tokens:
        q_tokens = tokenize(query)
    scores = bm25_model.get_scores(q_tokens)
    if len(scores) == 0:
        return []

    candidate_count = min(len(scores), max(40, top_k * BM25_CAND_MULTIPLIER))
    candidate_idx = np.argpartition(-scores, candidate_count - 1)[:candidate_count]
    candidate_idx = candidate_idx[np.argsort(-scores[candidate_idx])]

    results = []
    selected_tokens = []
    per_chunk = defaultdict(int)

    for idx in candidate_idx:
        if retrieve_meta is not None:
            source_idx = retrieve_meta[idx]["source_idx"]
            if per_chunk[source_idx] >= BM25_MAX_PER_CHUNK:
                continue

        if retrieve_doc_tokens is not None:
            candidate_tokens = retrieve_doc_tokens[idx]
            if any(_jaccard(candidate_tokens, chosen_tokens) > BM25_DEDUP_JACCARD for chosen_tokens in selected_tokens):
                continue
            selected_tokens.append(candidate_tokens)

        results.append(retrieve_docs[idx])
        if retrieve_meta is not None:
            per_chunk[source_idx] += 1
        if len(results) >= top_k:
            break

    if len(results) >= top_k:
        return results

    ranked = np.argsort(-scores)
    for idx in ranked:
        text = retrieve_docs[idx]
        if text in results:
            continue
        results.append(text)
        if len(results) >= top_k:
            break
    return results

async def _async_call_llm(prompt: str, model: str, tokenizer, temperature=0.7, top_p=0.95, stop=None, max_new_tokens=MAX_OUTPUT_LEN):
    session = await get_async_client()
    max_len = MAX_INPUT_LEN
    
    input_ids = tokenizer.encode(prompt)
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
        prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
        
    async with session:
        try:
            return await call_llm_text(
                session=session,
                model=model,
                text=prompt,
                temperature=temperature,
                top_p=top_p,
                max_len=max_new_tokens,
                stop=stop,
                enable_thinking=ENABLE_THINK,
                thinking_budget=THINKING_BUDGET,
            )
        except Exception as e:
            traceback.print_exc()
            return ""

async def call_llm_text(session, model, text, temperature, top_p, max_len, stop=None, enable_thinking=True, thinking_budget=1024):
    """
    Two-stage decoding with optional thinking budget control.
    Stage-1 uses `thinking_budget`; Stage-2 uses `max_len`.
    """
    try:
        if enable_thinking:
            payload1 = {
                "model": model,
                "prompt": text,
                "max_tokens": thinking_budget,
                "temperature": temperature,
                "top_p": top_p,
            }
            if stop is not None:
                payload1["stop"] = stop

            async with session.post(
                url=URL + "/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json=payload1,
            ) as resp1:
                if resp1.status != 200:
                    print(f"Stage1 HTTP {resp1.status} {await resp1.text()}")
                    return ""
                data1 = await resp1.json()
                think_part = data1["choices"][0]["text"]
        else:
            payload = {
                "model": model,
                "prompt": text,
                "max_tokens": max_len,
                "temperature": temperature,
                "top_p": top_p,
            }
            if stop is not None:
                payload["stop"] = stop
            async with session.post(
                url=URL + "/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json=payload,
            ) as resp:
                if resp.status != 200:
                    print(f"HTTP {resp.status} {await resp.text()}")
                    return ""
                data = await resp.json()
                return data["choices"][0]["text"]

        if "</think>" not in think_part:
            think_part = think_part.rstrip() + \
                "\n\nConsidering the limited time, I must stop thinking now.\n</think>\n\n"

        final_budget = max(1, max_len)
        followup_prompt = text + think_part
        payload2 = {
            "model": model,
            "prompt": followup_prompt,
            "max_tokens": final_budget,
            "temperature": temperature,
            "top_p": top_p,
        }
        if stop is not None:
            payload2["stop"] = stop

        async with session.post(
            url=URL + "/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json=payload2,
        ) as resp2:
            if resp2.status != 200:
                print(f"Stage2 HTTP {resp2.status} {await resp2.text()}")
                return think_part

            data2 = await resp2.json()
            answer_part = data2["choices"][0]["text"].strip()

        return (think_part + answer_part).strip()
    except Exception:
        traceback.print_exc()
        return ""

def _strip_thinking_content(text: str) -> str:
    """
    Keep only the visible answer part when `</think>` is present.
    """
    if not isinstance(text, str):
        return ""
    clean_text = text.strip()
    if not clean_text:
        return ""
    if "</think>" in clean_text:
        return clean_text.rsplit("</think>", 1)[-1].strip()
    if clean_text.startswith("<think>"):
        return clean_text[len("<think>"):].strip()
    return clean_text


async def async_query_llm(item, model, tokenizer, temperature=0.7, top_p=0.95, stop=None):
    context = item["context"]
    question = item['input'].strip()
    conversation = []

    input_ids = tokenizer.encode(context)
    if len(input_ids) > MAX_INPUT_LEN:
        input_ids = input_ids[:MAX_INPUT_LEN//2] + input_ids[-MAX_INPUT_LEN//2:]

    retrieve_docs, retrieve_meta = _build_retrieve_corpus_enhanced(
        input_ids=input_ids,
        tokenizer=tokenizer,
    )
    retrieve_doc_tokens = [tokenize_mixed(doc) for doc in retrieve_docs]
    bm25 = BM25Okapi(retrieve_doc_tokens)

    global_evidence_memory = []
    local_pathway_memory = []

    for iteration in range(MAX_ITERATIONS):
        # Combined memory queues
        combined_memory = "\n".join(global_evidence_memory + local_pathway_memory)
        if not combined_memory.strip():
            combined_memory = "None"

        # --- Module 1: Judge ---
        judge_p = PROMPT_JUDGE.format(question=question, combined_memory=combined_memory)
        judge_raw = await _async_call_llm(judge_p, model, tokenizer, temperature=0.1, top_p=top_p, stop=stop)
        judge_res = _strip_thinking_content(judge_raw)
        conversation.append([{"role": "user", "content": "JUDGE:\n" + judge_p}, {"role": "assistant", "content": judge_raw}])
        
        if judge_res.lower().startswith("yes"):
            break 

        # --- Module 2: Reasoner (Plan) ---
        reasoner_p = PROMPT_REASONER.format(combined_memory=combined_memory, question=question)
        sub_question_raw = await _async_call_llm(reasoner_p, model, tokenizer, temperature=0.7, top_p=top_p, stop=stop)
        sub_question = _strip_thinking_content(sub_question_raw)
        conversation.append([{"role": "user", "content": "REASONER:\n" + reasoner_p}, {"role": "assistant", "content": sub_question_raw}])
        
        if not sub_question:
            break

        # --- Retrieval ---
        chunks = bm25_retrieve_for_recurrent_chunk(
            query=sub_question,
            bm25_model=bm25,
            retrieve_docs=retrieve_docs,
            top_k=TOP_K,
            retrieve_doc_tokens=retrieve_doc_tokens,
            retrieve_meta=retrieve_meta,
        )
        docs_str = "\n\n".join(chunks)

        # --- Module 3: Global Evidence Summarizer ---
        global_p = PROMPT_GLOBAL_EVIDENCE.format(docs=docs_str, question=question)
        global_raw = await _async_call_llm(global_p, model, tokenizer, temperature=0.3, top_p=top_p, stop=stop)
        global_res = _strip_thinking_content(global_raw)
        clean_global_res = global_res.replace("[DONE]", "").strip()
        if clean_global_res:
            global_evidence_memory.append(f"Global Evidence: {clean_global_res}")
        conversation.append([{"role": "user", "content": "GLOBAL SUMMARIZER:\n" + global_p}, {"role": "assistant", "content": global_raw}])

        combined_memory_updated = "\n".join(global_evidence_memory + local_pathway_memory)

        # --- Module 4: Local Pathway Summarizer ---
        local_p = PROMPT_LOCAL_PATHWAY.format(sub_question=sub_question, combined_memory=combined_memory_updated)
        local_raw = await _async_call_llm(local_p, model, tokenizer, temperature=0.1, top_p=top_p, stop=stop)
        local_res = _strip_thinking_content(local_raw)
        
        if local_res.lower().startswith("yes"):
            answer = local_res[3:].strip() 
            if answer.startswith(","): answer = answer[1:].strip()
            local_pathway_memory.append(f"Q: {sub_question} -> A: {answer}")
        else:
            local_pathway_memory.append(f"Q: {sub_question} -> A: Need more information.")
            
        conversation.append([{"role": "user", "content": "LOCAL PATHWAY:\n" + local_p}, {"role": "assistant", "content": local_raw}])

    # --- Module 5: Generator (Response Generation) ---
    final_combined_memory = "\n".join(global_evidence_memory + local_pathway_memory)
    if not final_combined_memory.strip():
        final_combined_memory = "None"
        
    generator_p = PROMPT_GENERATOR.format(combined_memory=final_combined_memory, question=question)
    final_generation_raw = await _async_call_llm(generator_p, model, tokenizer, temperature=temperature, top_p=top_p, stop=stop)
    final_generation = _strip_thinking_content(final_generation_raw)
    conversation.append([{"role": "user", "content": "GENERATOR:\n" + generator_p}, {"role": "assistant", "content": final_generation_raw}])

    final_answer, _ = extract_solution(final_generation_raw) if final_generation_raw else ("", "")
    if not final_answer:
        final_answer = final_generation

    return {'conversation': conversation, 'final': final_answer}
