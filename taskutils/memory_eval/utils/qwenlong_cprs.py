import time
import argparse
import json
import random
import os
from tqdm import tqdm
from openai import OpenAI
import traceback
from .aio import get_async_client
from utils import extract_solution

from .envs import URL, API_KEY, MAX_INPUT_LEN, MAX_OUTPUT_LEN

from openai import AsyncOpenAI

template_0shot = """Please read the following text and answer the question below.

<text>
$DOC$
</text>

$Q$

Format your response as follows: "Therefore, the answer is (insert answer here)"."""


CPRS_PROMPT="You are an expert for information extraction, your task is to extract some sentences from the documents as the supporting facts of the user's question.\n## tagging rule:\n- tag the supporting facts with 'fact'"


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
def compress_api_call_local(messages: list):

    import requests
    retry_cnt = 0
    while retry_cnt < 2:
        try:
            data = {
                'header':{
                    'request_id': "1111abca"
                },
                'payload':{
                    'input':{
                        'messages':messages
                    },
                    'parameters':{
                        "min_keyword_len":1,
                        "complete_sentence":False,
                        "batch_size": 1,
                        'chunk_size': 8192
                    }
                }
            }

            url = 'http://0.0.0.0:8091/qwen_long_compress_server'
            payload = json.dumps(data)
            returns = requests.request("POST", url, data=payload)
            returns = returns.json()

            return returns['payload']['output']['text']

        except:
            retry_cnt += 1

            time.sleep(2)
            print("FAILED!",returns)
            continue
    # raise ValueError
    return []

def openai_call(messages: list, model: str,streaming=False):
    client = OpenAI(
        api_key=os.environ.get("LLM_APIKEY", None),
        base_url=os.environ.get("LLM_APIURL", None)
    )
    retry_cnt = 0
    while retry_cnt < 2:
        try:
            if streaming == False:
                response = client.chat.completions.create(
                    # model="deepseek-reasoner",
                    model=model,
                    messages=messages,
                    stream=False
                )
                print(response)
                output=response.choices[0].message.content.strip()
                return output
            else:
                response = client.chat.completions.create(
                    # model="deepseek-reasoner",
                    model=model,
                    messages=messages,
                )


                reasoning = ""
                output = ""
                for chunk in response:
                    chunk_message = chunk.choices[0].delta
                    # print(chunk.choices[0].delta)
                    if chunk_message.reasoning_content is not None:
                        reasoning += chunk_message.reasoning_content
                    
                    if chunk_message.content is not None:
                        output += chunk_message.content
                ans = output
                print(ans)
                return ans
        except:
            retry_cnt += 1
            traceback.print_exc()

            time.sleep(2)
            print("FAILED!",response)
            continue
    # raise ValueError
    return ""
    

# def process_item(d, args):
#     # from .prompts import prompt

#     try:
#         # Choose the appropriate prompt version
#         if args.use_compress:
#             messages_for_compress = [
#                 {
#                     'role': 'system',
#                     'content': args.cprs_prompt,
#                 },
#                 {
#                     'role': 'user',
#                     'content': d['query']
#                 },
#                 {
#                     'role': 'context',
#                     'content': d['context']
#                 }
#             ]
#             cprs_res = compress_api_call_local(messages_for_compress)
#             doc_content = "Doc content:\n\n" + '\n'.join(cprs_res)
#             d['cprs_preds'] = cprs_res
#         else:
#             doc_content = d['context']

#         messages = [
#             {'role': 'system', 'content': doc_content},
#             {'role': 'user', 'content': d['query']},
#         ]

#         ans = openai_call(messages=messages, model=args.model, streaming=args.streaming)

#         if ans != "":
#             d['llm_preds'] = ans
#             return  d
#         else:
#             return None

#     except Exception as e:
#         # Handle exceptions gracefully
#         print(f"Error processing id {d.get('id', 'N/A')}: {e}")
#         return None



async def async_query_llm(item, model, tokenizer, temperature=0.7, top_p=0.95, stop=None, use_compress=True):
    max_input_tokens=MAX_INPUT_LEN
    max_new_tokens=MAX_OUTPUT_LEN
    context = item["context"]
    question=item['input'].strip()
    conversation = []
    try:
        # Choose the appropriate prompt version
        if use_compress:
            messages_for_compress = [
                {
                    'role': 'system',
                    'content': CPRS_PROMPT,
                },
                {
                    'role': 'user',
                    'content': question
                },
                {
                    'role': 'context',
                    'content': context
                }
            ]
            cprs_res = compress_api_call_local(messages_for_compress)
            doc_content = "Doc content:\n\n" + '\n'.join(cprs_res)
            
            messages_for_compress.append({"role": "assistant", "content": doc_content})
            conversation.append(messages_for_compress)
            # d['cprs_preds'] = cprs_res
        else:
            doc_content = context

        
    except Exception as e:
        # Handle exceptions gracefully
        print(f"Error processing question '{question}': {e}")
        return {'conversation': conversation, 'final': ''}
    

    prompt = template_0shot.replace('$DOC$', doc_content.strip()).replace('$Q$', question)
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
                final_answer = data['choices'][0]['message']['content']
                conversation.append([{"role": "user", "content": prompt}, {"role": "assistant", "content": final_answer}])
                return {'conversation': conversation, 'final': final_answer}
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            import traceback
            traceback.print_exc()
        return {'conversation': conversation, 'final': ''}

        



# def main():
    
#     random.seed(10)

#     parser = argparse.ArgumentParser()

#     parser.add_argument("--model", type=str, default='gpt-4-1106-preview')
#     parser.add_argument('--input_path', type=str, required=True)
#     parser.add_argument('--output_path', type=str, required=True)
#     parser.add_argument('--start', type=int, default=0)
#     parser.add_argument('--end', type=int, default=99999)
#     parser.add_argument('--cprs_prompt', type=str, required=True)
#     parser.add_argument('--use_compress', type=str, default='False')
#     parser.add_argument('--streaming', type=str, default='False')
#     args = parser.parse_args()

#     def str2bool(text):
#         if text.lower() in ['true', 'yes', '1']:
#             return True
#         else:
#             return False
    
#     args.use_compress = str2bool(args.use_compress)
#     args.streaming = str2bool(args.streaming)

#     # Ensure the output directory exists
#     output_dir = os.path.dirname(args.output_path)
#     if output_dir:
#         os.makedirs(output_dir, exist_ok=True)

#     # Load existing IDs to skip

#     import json

#     # Load existing IDs to skip
#     existed_ids = set()
#     if os.path.exists(args.output_path):
#         with open(args.output_path, 'r', encoding='utf-8') as f_out:
#             for line in f_out:
#                 try:
#                     existed_ids.add(json.loads(line).get('id'))
#                 except json.JSONDecodeError:
#                     continue  # Skip malformed lines
   
#     # Load and filter input data
#     with open(args.input_path, 'r', encoding='utf-8') as f_in:
#         all_lines = f_in.readlines()

#     total_lines = len(all_lines)
#     print(f"Total input lines: {total_lines}")


#     filtered_data = []
#     for idx, line in enumerate(all_lines):
#         if idx < args.start or idx >= args.end:
#             continue
#         try:
#             data = json.loads(line)
#             if data['id'] not in existed_ids:
#                 filtered_data.append(data)
#         except json.JSONDecodeError:
#             continue  # Skip malformed lines
    

#     print(f"Data to process after filtering: {len(filtered_data)}")


#     # Use partial to fix the args parameter
#     gen_cnt = 0
#     with open(args.output_path, 'a', encoding='utf-8') as fw:
#         # Use imap_unordered for better performance with large datasets
#         for d in tqdm(filtered_data, total=len(filtered_data), desc="Processing"):
#             result= process_item(d, args)
#             if result is not None:
#                 print(f'=============={result["id"]}==============')
#                 print(result['query'])
#                 print('--------------------')
#                 print(result['llm_preds'])
#                 fw.write(json.dumps(result, ensure_ascii=False) + '\n')
#                 fw.flush()
#             else:
#                 print('GOT UNEXPECTED SUM!!!!!!!!!!!')
#             gen_cnt += 1


#     fw.close()


# if __name__ == "__main__":
#     main()