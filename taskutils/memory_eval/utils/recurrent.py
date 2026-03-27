from .aio import get_async_client
from utils import extract_solution
from .envs import URL, API_KEY, RECURRENT_CHUNK_SIZE, RECURRENT_MAX_NEW, RECURRENT_MAX_CONTEXT_LEN, ENABLE_THINK
import time

from openai import AsyncOpenAI

# Initialize OpenAI async client (only do this once)
client = AsyncOpenAI(api_key=API_KEY)

NO_MEMORY = "No previous memory"
ENABLE_THINK = False
if ENABLE_THINK:
    RECURRENT_MAX_NEW = 1024
    
TEMPLATE = """You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. Please read the provided section carefully and update the memory with the new information that helps to answer the problem. Be sure to retain all relevant details from the previous memory while adding any new, useful information.

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



def clip_long_string(string, max_length=2000):
    """Clip long string to a maximum length."""
    # assert max_length > 50, "max_length must be greater than 50"
    if not len(string) > max_length:
        return string
    target_len = max_length - len('\n\n...(truncated)\n\n')
    return string[target_len//3:target_len] + '\n\n...(truncated)\n\n' + string[-target_len//2:]

async def call_llm_text(session, model, text, temperature, top_p, max_len, stop=None, enable_thinking=False):
    """
    直接用已经 apply_chat_template 过的 text 调 vLLM 的 /v1/completions
    不再传 messages / tools / functions
    """
    # if model=="next_2":
    #     URL = f"http://{os.getenv('SERVE_HOST', '127.0.0.1')}:8002/v1"
    try:
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
            url=URL + "/completions",   # 注意这里用的是 /completions
            headers={"Authorization": f"Bearer {API_KEY}"},
            json=payload,
        ) as resp:
            status = resp.status
            if status != 200:
                print(f"{status=}, {model=}")
                print(await resp.text())
                return ""

            data = await resp.json()
            # /completions 返回的是 choices[0].text
            first_gen = data["choices"][0]["text"]
            
            if enable_thinking and "</think>" not in first_gen:
                sentence_to_mask = "\n\nConsidering the limited time by the user, I have to stop thinking now.\n</think>\n\n"
                followup_prompt = text + first_gen + sentence_to_mask

                payload2 = {
                    "model": model,
                    "prompt": followup_prompt,
                    "max_tokens": max_len,
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

                    status = resp2.status
                    if status != 200:
                        print(f"{status=}, {model=}")
                        print(await resp2.text())
                        return first_gen  # return first output at least

                    data2 = await resp2.json()
                    second_gen = data2["choices"][0]["text"].strip()

                # -------- concatenate two generations --------
                return (first_gen + sentence_to_mask + second_gen).strip()

            # -------- If no need for second call --------
            return first_gen
    except KeyboardInterrupt:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        return ""






def _process_messages_to_text(tokenizer, messages, functions=None, enable_thinking=False):
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

async def async_query_llm_multi_turn(item, model, tokenizer, temperature=0.7, top_p=0.95, stop=None):
    # model="gpt-4o-mini"
    perf_sample_start = time.perf_counter()
    loop_perf = []

    idx = item["_id"]
    context = item["context"].strip()
    prompt = item['input'].strip()
    answer = extract_answer(item)
    conversation = []
    session = await get_async_client()
    async with session:
        max_len = RECURRENT_MAX_CONTEXT_LEN
        input_ids = tokenizer.encode(context)
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
        memory = NO_MEMORY
        for i in range(0, len(input_ids), RECURRENT_CHUNK_SIZE):
            loop_start = time.perf_counter()
            chunk = input_ids[i:i+RECURRENT_CHUNK_SIZE]
            msg_to_teacher = TEMPLATE.format(prompt=prompt, chunk=tokenizer.decode(chunk), memory=memory)
            msg_original_temp = ORIGINAL_TEMPLATE.format(prompt=prompt, chunk=tokenizer.decode(chunk), memory=memory)
            # if idx == 0:
            #     print(f"{'--'*100}\nchunk_{i} user:\n\n{msg_to_teacher}")
                # print(msg_to_teacher)
            message = [{"role": "user", "content": msg_to_teacher}]
            text = _process_messages_to_text(tokenizer, message, functions=None, enable_thinking=ENABLE_THINK)


            generation = await call_llm_text(
                session=session,
                model=model,
                text=text,
                temperature=temperature,
                top_p=top_p,
                max_len=RECURRENT_MAX_NEW,
                enable_thinking=ENABLE_THINK
            )
            conversation.append([{"role": "user", "content": msg_original_temp},{"role": "assistant", "content": generation}])
            memory, _ = extract_solution(generation)
            loop_perf.append({
                "loop_idx": len(loop_perf) + 1,
                "loop_total_s": time.perf_counter() - loop_start,
                "retrieval_s": 0.0,
                "retrieval_calls": 0,
                "retrieval_mem_delta_mb": 0.0,
            })


        msg_final_to_teacher = TEMPLATE_FINAL.format(prompt=prompt, memory=memory)
        msg_final_original_temp = ORIGINAL_TEMPLATE_FINAL.format(prompt=prompt, memory=memory)


        message = [{"role": "user", "content": msg_final_to_teacher}]
        text = _process_messages_to_text(tokenizer, message, functions=None, enable_thinking=ENABLE_THINK)
        generation = await call_llm_text(session, model, text, temperature, top_p, RECURRENT_MAX_NEW, enable_thinking=ENABLE_THINK)
        final_answer, _ = extract_solution(generation)
        conversation.append([{"role": "user", "content": msg_final_original_temp},{"role": "assistant", "content": final_answer}])

        sample_total_s = time.perf_counter() - perf_sample_start
        avg_loop_s = (
            sum(loop["loop_total_s"] for loop in loop_perf) / len(loop_perf)
            if loop_perf
            else 0.0
        )
        return {
            'conversation': conversation,
            'final': final_answer,
            'perf': {
                "sample_total_s": sample_total_s,
                "avg_loop_s": avg_loop_s,
                "retrieval_total_s": 0.0,
                "retrieval_calls": 0,
                "retrieval_mem_deltas_mb": [],
                "retrieval_mem_delta_avg_mb": 0.0,
                "retrieval_mem_delta_max_mb": 0.0,
                "loops": loop_perf,
            },
        }
