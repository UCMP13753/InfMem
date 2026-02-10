
from .aio import get_async_client
from utils import extract_solution
from .envs import URL, API_KEY, RECURRENT_CHUNK_SIZE, RECURRENT_MAX_NEW, RECURRENT_MAX_CONTEXT_LEN,ENABLE_THINK,EARLY_STOP

from openai import AsyncOpenAI

# Initialize OpenAI async client (only do this once)
client = AsyncOpenAI(api_key=API_KEY)
from rank_bm25 import BM25Okapi
import nltk
import numpy as np
from qwen_agent.llm.fncall_prompts.qwen_fncall_prompt import QwenFnCallPrompt, FN_STOP_WORDS

import json
# from utils.document_chunks

nltk.download('punkt')
NO_MEMORY = "No previous memory"

RETRIEVE_CHUNK_SIZE = 500      



TEMPLATE_CALL_OR_ANSWER = """
You are a Retrieval Planner.  

Your ONLY task is to decide whether to perform another retrieval using `retrievesearch`, or STOP retrieving.

You MUST NOT answer the QUESTION.  
Another model will use MEMORY to answer later.

Guidelines:
- Retrieval is cheap. Unless MEMORY clearly contains all essential information, you are encouraged to retrieve.
- You may retrieve multiple times. At each step, refine your search direction.
- Avoid repeating any previous queries in RETRIEVAL_HISTORY (unless meaningfully refined).

When deciding whether to retrieve again:
1. Break the QUESTION into specific sub-questions or information needs.
2. Compare these needs with what MEMORY already contains.
3. Identify which facts are still missing, uncertain, or incomplete.
4. If something important is missing, design a NEW search query focused only on that missing information.
   - You may explore related clues hinted in MEMORY.
   - Queries should be concise, specific, and actionable.
5. If MEMORY already contains all necessary information, choose to STOP.

If you choose retrieval, you MUST output a function call to `retrievesearch` with:
- a new `query` (different from RETRIEVAL_HISTORY unless refined),
- and a `top_k` suited to your confidence (small: focused; large: broad exploration).
- In early retrieval steps, you may exlore more documents.
- In later steps, focus on refining MEMORY.


ORIGINAL QUESTION:
{prompt}

<retrieval_history>
{retrieval_history}
</retrieval_history>



CURRENT MEMORY:
{memory}

"""



TEMPLATE_RETREIVE_RECURRENT = """You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. Please read the provided section carefully and update the memory with the new information that helps to answer the problem.

the given section have two parts. One is reetrieved chunk, which are retrieved by given question. another is the recurrent chunk which are given recurrently. Both chunks might contain useful information, the retrieved chunk might have higher chance.



<problem>
{prompt}
</problem>

<retrieved_chunk>
{retrieve}
</retrieved_chunk>

<recurrent_chunk>
{chunk}
</recurrent_chunk>

<memory>
{memory}
</memory>


Updated memory:
"""


TEMPLATE_RETRIEVE = """You are presented with the final question and a the previous memory. Please according to the given memory, output the most important element you need to retrieve. 
Your output will be used as the query for BM25, make sure you output the straightforward element directly. You are allowed to list multiple key elements.

<problem>
{prompt}
</problem>

<memory>
{memory}
</memory>
Elements need to be retrieved:
"""


ORIGINAL_TEMPLATE = """You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. Please read the provided section carefully and update the memory with the new information that helps to answer the problem. Be sure to retain all relevant details from the previous memory while adding any new, useful information.

<problem> 
{prompt}
</problem>

<memory>
{memory}
</memory>

<section>
{chunk}
</section>

Updated memory:
"""



ORIGINAL_TEMPLATE_FINAL = """You are presented with a problem and a previous memory. Please answer the problem based on the previous memory and format your response as follows "Therefore, the answer is (insert answer here)".

<problem> 
{prompt}
</problem>

<memory>
{memory}
</memory>

Your answer:
"""
TEMPLATE_FINAL = """You are presented with a problem and a previous memory. Please answer the problem based on the previous memory and format your response as follows "Therefore, the answer is (insert answer here)".

<problem> 
{prompt}
</problem>

<memory>
{memory}
</memory>

Your answer:
"""


# Function call special tokens (from Qwen-Agent)
FN_NAME = '✿FUNCTION✿'
FN_ARGS = '✿ARGS✿'
FN_RESULT = '✿RESULT✿'
FN_EXIT = '✿RETURN✿'


def clip_long_string(string, max_length=2000):
    """Clip long string to a maximum length."""
    # assert max_length > 50, "max_length must be greater than 50"
    if not len(string) > max_length:
        return string
    target_len = max_length - len('\n\n...(truncated)\n\n')
    return string[target_len//3:target_len] + '\n\n...(truncated)\n\n' + string[-target_len//2:]


def extract_answer(item):
    # Prefer item["outputs"]
    if "outputs" in item and item["outputs"] is not None:
        outputs = item["outputs"]

        # Ensure it's a list
        if not isinstance(outputs, list):
            outputs = [outputs]

        # If all answers are the same → keep only the first
        unique = list(dict.fromkeys(outputs))  # preserves order

        if len(unique) == 1:
            return unique[0]

        # Otherwise join them
        return ", ".join(unique)

    # Fallback: item["answers"] (string)
    if "answers" in item and item["answers"]:
        ans = item["answers"]
        if isinstance(ans, list):
            return ", ".join(map(str, ans))
        return str(ans)

    return ""


########## tools ################
def tokenize(text: str):
    return text.lower().split()

def bm25_retrieve_for_recurrent_chunk(
    query: str,
    bm25_model: BM25Okapi,
    retrieve_docs,
    retrieve_meta,
    top_k: int,
    exclude_recurrent_idx: int | None = None,
):
    q_tokens = tokenize(query)
    scores = bm25_model.get_scores(q_tokens)
    ranked = np.argsort(-scores)  # 降序

    results = []
    for idx in ranked:
        if exclude_recurrent_idx is not None:
            if retrieve_meta[idx]["recurrent_idx"] == exclude_recurrent_idx:
                
                continue
        results.append(retrieve_docs[idx])
        if len(results) >= top_k:
            break
    return results



def _process_messages_to_text(tokenizer, messages, functions=None, enable_thinking=True):
    """
    Process messages to text format suitable for Qwen model input.
    Extracted from _complete_qwen for reuse in generation.py
    """
    # Convert messages to proper format for Qwen processing
    from qwen_agent.llm.schema import Message, ContentItem
    qwen_messages = []
    for msg in messages: # [{"role": ..., "content": ...}, ...]
        content = msg["content"]
        if isinstance(content, str):
            content = [ContentItem(text=content)]
        elif isinstance(content, list):
            # Handle mixed content types
            content_items = []
            for item in content:
                if isinstance(item, str):
                    content_items.append(ContentItem(text=item))
                elif isinstance(item, dict) and "text" in item: # If the list item is a dictionary and has a "text" key, then extract value
                    content_items.append(ContentItem(text=item["text"]))
                else:
                    content_items.append(ContentItem(text=str(item)))
            content = content_items
        else:
            content = [ContentItem(text=str(content))]

        qwen_msg = Message(role=msg["role"], content=content)
        qwen_messages.append(qwen_msg)

    # Preprocess messages with function calling format
    if functions: # If tool functions are available
        processed_messages = QwenFnCallPrompt.preprocess_fncall_messages(
            messages=qwen_messages,
            functions=functions,
            lang='en',  # Using English
            parallel_function_calls=True,
            function_choice='auto'
        )
    else:
        processed_messages = qwen_messages

    # Convert back to dict format for tokenizer
    dict_messages = []
    for msg in processed_messages:
        content_text = ""
        for content_item in msg.content:
            content_text += content_item.text
        dict_messages.append({
            "role": msg.role,
            "content": content_text
        })

    # Apply chat template
    text = tokenizer.apply_chat_template( # Convert the list of dialogue messages into a single, suitable string as model input
        dict_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking # Determine whether to enable think prompt in the chat template
    )

    return text


def _parse_response(response):
    """
    Parse the response from Qwen model to extract function calls and text content.
    Returns a list of assistant messages with either content or function_call.
    """
    function_calls, remaining_text, stop = _parse_function_calls_from_text(response)

    messages = []

    if function_calls:
        # Add main content message if there's remaining text
        if remaining_text.strip():
            messages.append({
                "role": "assistant",
                "content": "",
                "reasoning_content": remaining_text.strip().strip("<think>").strip("</think>").strip()
            })

        # Add function call messages
        for func_call in function_calls:
            messages.append({
                "role": "assistant",
                "content": "",
                "function_call": {
                    "name": func_call["name"],
                    "arguments": func_call["arguments"]
                }
            })

    else:
        # No function calls - parse thinking and final response
        # Format: <think>...</think>\n\nfinal_response
        thinking_pattern = r'<think>(.*?)</think>\s*(.*)'
        match = re.search(thinking_pattern, response, re.DOTALL)

        if match:
            thinking_content = match.group(1).strip()
            final_content = match.group(2).strip()

            messages.append({
                "role": "assistant",
                "content": final_content,
                "reasoning_content": thinking_content
            })

        else:
            # No thinking tags found, treat entire response as content
            # Remove any incomplete thinking tags
            clean_response = response.strip()
            if clean_response.startswith('<think>') and '</think>' not in clean_response:
                # Incomplete thinking tag at start
                clean_response = clean_response[7:].strip()  # Remove '<think>'

            messages.append({
                "role": "assistant",
                "content": clean_response,
                "reasoning_content": ""
            })

    # If no content and no function calls, add an empty content message
    if not messages:
        messages.append({
            "role": "assistant",
            "content": ""
        })

    return messages,stop

def _parse_function_calls_from_text(text: str):
    """
    Parse function calls from raw model output text.
    Returns (function_calls, remaining_text).
    Adapted from raw_token_function_calling.py
    """
    function_calls = []
    if "STOP" in text:
        return function_calls, "", True
    # Find all function call patterns
    pattern = f'{re.escape(FN_NAME)}:\\s*([^\\n]+)\\s*{re.escape(FN_ARGS)}:\\s*([^✿]+?)(?={re.escape(FN_RESULT)}|{re.escape(FN_EXIT)}|{re.escape(FN_NAME)}|$)'

    matches = re.finditer(pattern, text, re.DOTALL)

    for match in matches:
        func_name = match.group(1).strip()
        func_args = match.group(2).strip()

        # Clean up arguments
        func_args = _remove_trailing_comment_of_fn_args(func_args)

        function_calls.append({
            'name': func_name,
            'arguments': func_args
        })

    # Extract text before first function call
    first_fn_pos = text.find(FN_NAME)
    if first_fn_pos >= 0:
        remaining_text = text[:first_fn_pos].strip()
    else:
        remaining_text = text.strip()

    # Remove incomplete special tokens
    remaining_text = _remove_incomplete_special_tokens(remaining_text)

    return function_calls, remaining_text, False

def _remove_incomplete_special_tokens(text: str) -> str:
    """Remove incomplete special tokens from the end of text."""
    special_tokens = (FN_NAME, FN_ARGS, FN_RESULT, FN_EXIT)
    text = text.rstrip()

    if text.endswith(special_tokens):
        for s in special_tokens:
            if text.endswith(s):
                text = text[:-len(s)]
                break
    else:
        trail_start = text.rfind('✿')
        if trail_start >= 0:
            trail_token = text[trail_start:]
            for s in special_tokens:
                if s.startswith(trail_token):
                    text = text[:trail_start]
                    break

    text = text.lstrip('\n').rstrip()
    return text


def _remove_trailing_comment_of_fn_args(fn_args: str) -> str:
    """Remove trailing comments from function arguments."""
    fn_args = fn_args.strip()

    if fn_args.startswith('{'):
        k = fn_args.rfind('}')
        if k > 0:
            fn_args = fn_args[:k + 1]

    if fn_args.startswith('```'):
        k = fn_args.rfind('\n```')
        if k > 0:
            fn_args = fn_args[:k + 4]

    return fn_args


########## end tools ################



from typing import List, Dict, Tuple
import json
import os
import openai
import uuid
import math
import re
import numpy as np
from collections import Counter, defaultdict
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi



def get_bm25search_tool_schema():
    
    return {
        "type": "function",
        "function": {
            "name": "retrievesearch",
            "description": "Search over pre-indexed context chunks using BM25.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Concise search query focusing on key entities / concepts."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of chunks to retrieve (3-20).",
                        "minimum": 3,
                        "maximum": 20,
                        "default": 8
                    },
                },
                "required": ["query"]
            },
        },
    }








async def call_llm_text(session, model, text, temperature, top_p, max_len, stop=None, enable_thinking=True, thinking_budget=1024):   # 你可以自由调整

    """
    control think or not, if enable think, set thinking budget and garantee the memory budget
    """

    try:
        # -------------------- Stage 1: Thinking generation --------------------
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
            # No thinking stage: one pass only
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

        # -----------------------------------
        # Stage 1 done
        # -----------------------------------
        if "</think>" not in think_part:
            # ---------- insert closing tag ----------
            think_part = think_part.rstrip() + \
                "\n\nConsidering the limited time, I must stop thinking now.\n</think>\n\n"

        # -------------------- Stage 2: Answer generation --------------------
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
                return think_part  # at least return the thinking

            data2 = await resp2.json()
            answer_part = data2["choices"][0]["text"].strip()

        # -------------------- Final concatenation --------------------
        return (think_part + answer_part).strip()

    except Exception:
        import traceback
        traceback.print_exc()
        return ""







async def async_query_llm_multi_turn(item, model, tokenizer, temperature=0.7, top_p=0.95, stop=None):
    # model="gpt-4o-mini"
    idx = item["_id"]
    context = item["context"].strip() 
    
    # print(decomp)
    # register bm25 search
    bm25_tool = get_bm25search_tool_schema()
    functions = [bm25_tool["function"]]  


    conversation = []
    session = await get_async_client()
    async with session:


        max_len = RECURRENT_MAX_CONTEXT_LEN
        input_ids = tokenizer.encode(context)
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
        memory = NO_MEMORY



        #### build recurrent #####
        recurrent_chunks = []  # list of dict: {'r_idx', 'start', 'end'}
        pos = 0
        while pos < len(input_ids):
            start = pos
            end = min(len(input_ids), pos + RECURRENT_CHUNK_SIZE)
            recurrent_chunks.append({
                "r_idx": len(recurrent_chunks),
                "start": start,
                "end": end,
            })
            pos = end

        
        
        retrieve_docs = []   
        retrieve_meta = []   

        for rc in recurrent_chunks:
            r_idx = rc["r_idx"]
            start, end = rc["start"], rc["end"]
            p = start
            while p < end:
                rs = p
                re = min(end, p + RETRIEVE_CHUNK_SIZE)
                sub_ids = input_ids[rs:re]
                text = tokenizer.decode(sub_ids)
                retrieve_docs.append(text)
                retrieve_meta.append({
                    "recurrent_idx": r_idx,
                    "start": rs,
                    "end": re,
                })
                p = re

        # === Step 2.3: build bm25 index） ===
        corpus_tokens = [tokenize(doc) for doc in retrieve_docs]
        bm25 = BM25Okapi(corpus_tokens)
        


        def bm25search_impl(query: str, top_k: int = 8):
            
           
            chunks = bm25_retrieve_for_recurrent_chunk(
                query=query,
                bm25_model=bm25,
                retrieve_docs=retrieve_docs,
                retrieve_meta=retrieve_meta,
                top_k=4,
                exclude_recurrent_idx=None,
            )
            
            return {
                "top_k": top_k,
                "query": query,
                "results": [
                    {
                        "rank": i + 1,
                        "text": txt,
                        
                    }
                    for i, txt in enumerate(chunks)
                ]
            }




        prompt = item['input'].strip()


        TOP_K = 8 
        stop_time = 0
        history=[]

        max_retrieve_steps = len(recurrent_chunks)

        for retrieve_step in range(len(recurrent_chunks)):
            rc = recurrent_chunks[retrieve_step]
            start, end = rc["start"], rc["end"]
            chunk_ids = input_ids[start:end]
            chunk_text = tokenizer.decode(chunk_ids)
 
            retrieval_history_str = (
                f"Current retrieval step: {retrieve_step+1}  Maximum allowed retrieval steps: {max_retrieve_steps}\n"
                + "\n".join(
                    f"Step {i+1}: query={h['query']!r}, top_k={h['top_k']}"
                    for i, h in enumerate(history)
                )
            )
            
            msg_to_rewrite = TEMPLATE_CALL_OR_ANSWER.format(
                prompt=prompt,
                memory=memory,
                retrieval_history=retrieval_history_str or "None yet.",
            )
            
            message = [{"role": "user", "content": msg_to_rewrite}]
            text = _process_messages_to_text(tokenizer, message, functions, ENABLE_THINK)

            function_call_raw = await call_llm_text(
                session=session,
                model=model,
                text=text,
                temperature=temperature,
                top_p=top_p,
                max_len=RECURRENT_MAX_NEW,
                enable_thinking=ENABLE_THINK
            )

            parsed_messages,stop = _parse_response(function_call_raw)

            retrieved_block = ""
            tool_calls = []
            answer_msg = None 
            conversation.append([
                {"role": "user", "content": msg_to_rewrite},
                {"role": "assistant", "content":function_call_raw},
            ])
            
            for msg in parsed_messages:
                if "function_call" in msg:
                    tool_calls.append(msg)
                elif msg.get("content"): 
                    answer_msg = msg
                
            if not stop:
                stop_time = 0
            # === Case A: stop, already have answer in memory ===
            if stop or not tool_calls:
                stop_time+=1
                if stop_time>=EARLY_STOP:


                    msg_final_to_teacher = TEMPLATE_FINAL.format(
                        prompt=prompt,
                        memory=memory,
                    )
                    msg_final_original_temp = ORIGINAL_TEMPLATE_FINAL.format(
                        prompt=prompt,
                        memory=memory,
                    )
                    
                    message = [{"role": "user", "content": msg_final_to_teacher}]
                    text = _process_messages_to_text(tokenizer, message, functions=None, enable_thinking=ENABLE_THINK)
                    generation = await call_llm_text(session, model, text, temperature, top_p, RECURRENT_MAX_NEW, enable_thinking=ENABLE_THINK)
        
                    final_answer, _ = extract_solution(generation)
                    conversation.append([
                        {"role": "user", "content": msg_final_original_temp},
                        {"role": "assistant", "content": final_answer}
                    ])

                    return {"conversation": conversation, "final": final_answer, "step": retrieve_step+1,"max_step":max_retrieve_steps}


            # === Case B: have function_call ===
            for msg in tool_calls:
                fn_name = msg["function_call"]["name"]
                raw_args = msg["function_call"]["arguments"]
                
                if fn_name == "retrievesearch":
                    try:
                        args = json.loads(raw_args)
                        query = args.get("query",prompt)
                        top_k = args.get("top_k",TOP_K)
                        method = args.get("method","bm25") # 比如默认让它用 bge
                    except Exception:
                        print("function call invalid, use default value")
                        args = {}
                        query = prompt
                        top_k = TOP_K
                        method = "bm25"
                    history.append({"method": method, "query": query, "top_k": top_k})
                    
                    ret = bm25search_impl(query=query, top_k=top_k)
                    
                    retrieved_block = "\n\n".join(
                        f"[Retrieved #{r['rank']}] {r['text']}"
                        for r in ret["results"]
                    )


            msg_to_teacher = TEMPLATE_RETREIVE_RECURRENT.format(
                prompt=prompt,
                retrieval_history=retrieval_history_str,
                retrieve=retrieved_block,
                chunk=chunk_text,
                memory=memory,
            )
            message = [{"role": "user", "content": msg_to_teacher}]
            text = _process_messages_to_text(tokenizer, message, enable_thinking=ENABLE_THINK)
            generation = await call_llm_text(
                session=session,
                model=model,
                text=text,
                temperature=temperature,
                top_p=top_p,
                max_len=RECURRENT_MAX_NEW,
                enable_thinking=ENABLE_THINK
            )
            

            conversation.append([
                {"role": "user", "content": msg_to_teacher},
                {"role": "assistant", "content": generation}
            ])

            memory, _ = extract_solution(generation)

            
        msg_final_to_teacher = TEMPLATE_FINAL.format(prompt=prompt, memory=memory)
        msg_final_original_temp = ORIGINAL_TEMPLATE_FINAL.format(prompt=prompt, memory=memory)

        message = [{"role": "user", "content": msg_final_to_teacher}]
        text = _process_messages_to_text(tokenizer, message, functions=None, enable_thinking=ENABLE_THINK)
        generation = await call_llm_text(session, model, text, temperature, top_p, RECURRENT_MAX_NEW, enable_thinking=ENABLE_THINK)
        final_answer, _ = extract_solution(generation)
        conversation.append([{"role": "user", "content": msg_final_original_temp},{"role": "assistant", "content": final_answer}])

        return {'conversation': conversation, 'final': final_answer,"step": max_retrieve_steps, "max_step":max_retrieve_steps}

