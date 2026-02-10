# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os, csv, json
import argparse
import time
from tqdm import tqdm
from datasets import load_dataset
import re
from transformers import AutoTokenizer
import tiktoken
import torch.multiprocessing as mp
import string
from collections import Counter
from datasets import load_dataset, concatenate_datasets,Dataset

from utils import extract_solution,update_answer, update_answer_traj
from utils.envs import DATAROOT


def calc_metrics(predictions, goldens):
    assert len(predictions) == len(goldens)
    metrics = {'f1': 0, 'prec': 0, 'recall': 0, 'em': 0, 'sub_em': 0, 'total_num': 0}
    for pred, gold in zip(predictions, goldens):
        update_answer(metrics, pred, gold)
    for k, _ in metrics.items():
        if k == 'total_num':
            continue
        metrics[k] = round((metrics[k]/metrics['total_num']), 2)
    return metrics


### add 
def calc_preserve_metric(conv_traj, goldens):
    assert len(conv_traj) == len(goldens)
    metrics = {'found_in_memory': 0, 'found_in_think': 0, 'preserve_memory': 0, 'preserve_memory_in_think': 0,'total_num': 0}
    for pred, gold in zip(conv_traj, goldens):
        update_answer_traj(metrics, pred, gold)
    for k, _ in metrics.items():
        if k == 'total_num':
            continue
        metrics[k] = round((metrics[k]/metrics['total_num']), 2)
    return metrics



def get_pred_with_conversation_trace(data, args, out_file):
    model = args.model
    if "gpt" in model or "o1" in model or "o3" in model or "o4" in model or "gemini" in model or "claude" in model:
        tokenizer = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if args.api == "recurrent":
        from utils.recurrent import async_query_llm_multi_turn as async_query_llm
        from utils import extract_answer
    elif args.api == "infmem":
        from utils.infmem import async_query_llm_multi_turn as async_query_llm
        from utils import extract_answer
    elif args.api == "openai":
        from utils.openai_api import async_query_llm
        from utils import extract_answer
    elif args.api == "CPRS":
        from utils.qwenlong_cprs import async_query_llm as async_query_llm
        from utils import extract_answer
    elif args.api == "rag":
        from utils.openai_retrieval import async_query_llm as async_query_llm
        from utils import extract_answer
    else:
        print(f"Invalid API: {args.api}")
        raise ValueError
    coros = []
    for item in data:
        coro = async_query_llm(item, model, tokenizer, temperature=0, top_p=1)
        coros.append(coro)
    from utils.aio import async_main, close_async_client
    import uvloop
    outputs = uvloop.run(async_main(coros, args.n_proc))
    uvloop.run(async_main([close_async_client()]))
    from collections import defaultdict
    scores = defaultdict(list)
    fout = open(out_file, 'w' if args.force else 'a', encoding='utf-8')
    steps = []
    max_steps = []
    for i, (output, item) in enumerate(zip(outputs, data)):
        if output == '':
            continue
        # assume output is a dict from async_query_llm
        response = output.get('final', '').strip()
        conversation_trace = output.get('conversation', [])
        stop_step  = output.get('step', None)
        
        max_step = output.get('max_step', None)
        if stop_step is not None:
            steps.append(stop_step)
        if max_step is not None:
            max_steps.append(max_step)
        pred, _ = extract_solution(response)
        item['response'] = response
        item['answer'] = item["answers"][0]

        item['pred'] = extract_answer(pred) if pred else extract_answer(response)
        item['judge_f1'] = calc_metrics([item["pred"]], [item["answer"]])['f1'] if item["pred"] else 0
        item['judge_em'] = calc_metrics([item["pred"]], [item["answer"]])['em'] if item["pred"] else 0
        item['judge_sub_em'] = calc_metrics([item["pred"]], [item["answer"]])['sub_em'] if item["pred"] else 0


        item['found_in_memory'] = calc_preserve_metric([conversation_trace], [item["answer"]])['found_in_memory'] if item["pred"] else 0
        item['found_in_think'] = calc_preserve_metric([conversation_trace], [item["answer"]])['found_in_think'] if item["pred"] else 0
        
        item['preserve_memory'] = calc_preserve_metric([conversation_trace], [item["answer"]])['preserve_memory'] if item["pred"] else 0
        item['preserve_memory_in_think'] = calc_preserve_metric([conversation_trace], [item["answer"]])['preserve_memory_in_think'] if item["pred"] else 0
        
        
        scores['f1'].append(item['judge_f1'])
        scores['em'].append(item['judge_em'])
        scores['sub_em'].append(item['judge_sub_em'])
        scores['found_in_memory'].append(item['found_in_memory'])
        scores['found_in_think'].append(item['found_in_think'])
        scores['preserve_memory'].append(item['preserve_memory'])
        scores['preserve_memory_in_think'].append(item['preserve_memory_in_think'])
        item.pop('context');fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        if i == 0:
            print("="*40 + "New Item Start" + "="*40)
            print(item['response'])
            print("-"*80)
            print(item['pred'])
            print("-"*80)
            print(item['answer'])
            print("-"*80)
            print(item['judge_sub_em'])
            print("="*40 + "New Item End" + "="*40)
    print(f"ruler_hqa [{args.length}]")
    for k, v in scores.items():
        print(f"{k}: {round(sum(v) * 100 /len(v), 2)}")
    print(f"Total: {len(data)}")
    try:
        avg_step = sum(steps) / len(steps) if len(steps) > 0 else 0.0
        print(
            f"avg={avg_step:.2f}, "
            f"min={min(steps)}, "
            f"max={max(steps)}, "
            f"count={len(steps)}"
        )

        avg_maxstep = sum(max_steps) / len(max_steps) if len(max_steps) > 0 else 0.0
        print(
            "MAX:"
            f"avg={avg_maxstep:.2f}, "
            f"min={min(max_steps)}, "
            f"max={max(max_steps)}, "
            f"count={len(max_steps)}"
        )
    except Exception as e:
        print(e)



def main():
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)
    out_file = os.path.join(args.save_dir, args.save_file + ".jsonl")
    
    path = f"{DATAROOT}/eval_{args.length}.json"

    with open(path, "r", encoding="utf-8") as f:
        data_list = json.load(f)  # list of dicts

    dataset = Dataset.from_list(data_list)
    dataset = concatenate_datasets([dataset])

    print(f"Loaded {len(dataset)} items")
    print(f"original data len {len(dataset)}")
    
    import copy
    dataset = [copy.deepcopy(item) for _ in range(args.sampling) for item in dataset]
    print(f"sampling data len {len(dataset)}")

    data_all = []
    for idx, item in enumerate(dataset):
        item["_id"] = idx 
        data_all.append(item)

    print(data_all[0]["_id"])
    print(data_all[-1]["_id"])
    #for test
    # data_all = data_all[:1]
    
    # cache
    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            has_data = {json.loads(line)["_id"]: 0 for line in f}
    data = []
    for item in data_all:
        if item["_id"] not in has_data or args.force:
            data.append(item)
        elif args.force:
            data.append(item)

    get_pred_with_conversation_trace(data, args, out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=50, choices=[50, 100, 200, 400, 800, 1600, 3200, 6400])
    parser.add_argument("--save_dir", "-s", type=str, default="results/ruler_hqa")
    parser.add_argument("--save_file", "-f", type=str, default="MemoryAgent-7B-recurrent")
    parser.add_argument("--model", "-m", type=str, default="BytedTsinghua-SIA/RL-MemoryAgent-7B")
    parser.add_argument("--tokenizer", "-t", type=str, default="BytedTsinghua-SIA/RL-MemoryAgent-7B")
    parser.add_argument("--n_proc", "-n", type=int, default=64)
    parser.add_argument("--api", "-a", type=str, default="recurrent")
    parser.add_argument("--sampling", "-p", type=int, default=1)
    parser.add_argument('--force', action='store_true', help='force to overrite')
    args = parser.parse_args()
    main()