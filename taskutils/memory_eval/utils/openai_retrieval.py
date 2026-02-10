
from .aio import get_async_client
from utils import extract_solution
from .envs import URL, API_KEY,MAX_INPUT_LEN,MAX_OUTPUT_LEN
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
ENABLE_THINK=True


RETRIEVE_CHUNK_SIZE = 2000 
TOP_K = 6
########## tools ################






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



def tokenize(text: str):
    return text.lower().split()

def bm25_retrieve_for_recurrent_chunk(
    query: str,
    bm25_model: BM25Okapi,
    retrieve_docs,
    top_k: int,
):
    q_tokens = tokenize(query)
    scores = bm25_model.get_scores(q_tokens)
    ranked = np.argsort(-scores)  # 降序

    results = []
    for idx in ranked:
        results.append(retrieve_docs[idx])
        if len(results) >= top_k:
            break
    return results


template_0shot = """Please read the following text and answer the question below.

<text>
$DOC$
</text>

$Q$

Format your response as follows: "Therefore, the answer is (insert answer here)"."""
from .aio import get_async_client
async def async_query_llm(item, model, tokenizer, temperature=0.7, top_p=0.95, stop=None):
    max_input_tokens=MAX_INPUT_LEN
    max_new_tokens=MAX_OUTPUT_LEN

    context = item["context"]
    question = item['input'].strip()

    conversation = []



    input_ids = tokenizer.encode(context)
    if len(input_ids) > max_input_tokens:
        input_ids = input_ids[:max_input_tokens//2] + input_ids[-max_input_tokens//2:]

    retrieve_docs = []  
    pos = 0
    while pos < len(input_ids):
        start = pos
        end = min(len(input_ids), pos + RETRIEVE_CHUNK_SIZE)
        sub_ids = input_ids[pos:end]
        text = tokenizer.decode(sub_ids)
        retrieve_docs.append(text)
        pos = end
    corpus_tokens = [tokenize(doc) for doc in retrieve_docs]
    bm25 = BM25Okapi(corpus_tokens)



    chunks = bm25_retrieve_for_recurrent_chunk(
        query=question,
        bm25_model=bm25,
        retrieve_docs=retrieve_docs,
        top_k=TOP_K,
    )

    retrieved_block = "\n\n".join(chunks)
    prompt = template_0shot.replace('$DOC$', retrieved_block).replace('$Q$', question)
    session = await get_async_client()
    async with session:
        max_len = max_input_tokens
        input_ids = tokenizer.encode(prompt)
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
            prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
        try:
            async with session.post(
                url=URL + "/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json=dict(model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_new_tokens
                )
            ) as resp:
                status = resp.status
                if status!= 200:
                    print(f"{status=}, {model=}")
                    return {'conversation': conversation, 'final': ''}
                data = await resp.json()
                generation = data['choices'][0]['message']['content']
                final_answer, _ = extract_solution(generation)
                conversation.append([{"role": "user", "content": prompt},{"role": "assistant", "content": generation}])
                return {'conversation': conversation, 'final': final_answer}
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            import traceback
            traceback.print_exc()
        return {'conversation': conversation, 'final': ''}


