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

### From RULER
def string_match_all(pred, ref):
    return sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref)


### add
def update_answer_traj_niah(metrics, conv_traj, gold_refs):
    """
    gold_refs: list of required strings / spans
    conv_traj: list of tuples of (tool_call, llm_response)
    """

    found_in_mem = False
    found_in_think = False
    preserve_in_mem = False
    preserve_in_think = False

    for i in range(len(conv_traj) - 1):
        try:
            res = conv_traj[i]
            mem = res[1].get('content', '').strip()
            final_mem, smem_str = extract_solution(mem)

            # match any ref span in mem/think
            in_mem = string_match_all(final_mem, gold_refs) > 0
            in_think = string_match_all(smem_str, gold_refs) > 0

            found_in_mem = found_in_mem or in_mem
            found_in_think = found_in_think or in_think

            # check preservation only at last thinking turn
            if i == len(conv_traj) - 2:
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


def calc_metrics(predictions, goldens):
    assert len(predictions) == len(goldens)
    metrics = {'sub_em': 0, 'total_num': 0}
    for pred, gold in zip(predictions, goldens):
        metrics['sub_em'] += string_match_all(pred, gold)
    metrics['total_num'] = len(goldens)
    for k, _ in metrics.items():
        if k == 'total_num':
            continue
        metrics[k] = round((metrics[k]/metrics['total_num']), 2)
    return metrics

def calc_qa_metrics(predictions, goldens):
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
def calc_niah_preserve_metric(conv_trajs, gold_refs_list):
    """
    conv_trajs: list of conversation trajectories
    gold_refs_list: list of list[str] references (per example)
    """
    assert len(conv_trajs) == len(gold_refs_list)
    
    metrics = {
        'found_in_memory': 0,
        'found_in_think': 0,
        'preserve_memory': 0,
        'preserve_memory_in_think': 0,
        'total_num': 0
    }

    for conv, gold_refs in zip(conv_trajs, gold_refs_list):
        update_answer_traj_niah(metrics, conv, gold_refs)

    for k in metrics:
        if k == 'total_num':
            continue
        metrics[k] = round(metrics[k] / metrics['total_num'], 2)
    
    return metrics

def calc_qa_preserve_metric(conv_traj, goldens):
    assert len(conv_traj) == len(goldens)
    metrics = {'found_in_memory': 0, 'found_in_think': 0, 'preserve_memory': 0, 'preserve_memory_in_think': 0,'total_num': 0}
    for pred, gold in zip(conv_traj, goldens):
        update_answer_traj(metrics, pred, gold)
    for k, _ in metrics.items():
        if k == 'total_num':
            continue
        metrics[k] = round((metrics[k]/metrics['total_num']), 2)
    return metrics





def get_pred_with_conversation_trace(data, args, out_file, n_rollout=1):
    model = args.model
    if "gpt" in model or "o1" in model or "o3" in model or "o4" in model or "gemini" in model or "claude" in model:
        tokenizer = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if args.api == "recurrent":
        from utils.recurrent import async_query_llm_multi_turn as async_query_llm
        from utils import extract_answer
    
    elif args.api == "openai":
        from utils.openai_api import async_query_llm
        from utils import extract_answer
    elif args.api == "infmem":
        from utils.infmem import async_query_llm_multi_turn as async_query_llm
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


    expanded_data = []
    for item in data:
        for r in range(n_rollout):
            new_item = item.copy()
            new_item["rollout_id"] = r
            expanded_data.append(new_item)
    for item in expanded_data:
        coro = async_query_llm(
            item,
            model,
            tokenizer,
            temperature=0.0, 
            top_p=0.95
        )
        coros.append(coro)

    from utils.aio import async_main, close_async_client
    import uvloop
    outputs = uvloop.run(async_main(coros, args.n_proc))
    uvloop.run(async_main([close_async_client()]))



    from collections import defaultdict
    scores = defaultdict(list)
    grouped = defaultdict(list)
    fout = open(out_file, 'w' if args.force else 'a', encoding='utf-8')
    
    steps = []
    max_steps = []
    for i, (output, item) in enumerate(zip(outputs, expanded_data)):
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
        item['answer'] = item.pop("outputs")
        item['pred'] = extract_answer(pred) if pred else extract_answer(response)
        if "qa" in args.split:
            if item['pred']:
                metrics = calc_qa_metrics([item["pred"]], [item["answer"][0]])
                preserved_metric = calc_qa_preserve_metric([conversation_trace], [item["answer"][0]])
            else:
                metrics = {'f1': 0, 'prec': 0,'recall': 0, 'em': 0,'sub_em': 0, 'total_num': 0}
                preserved_metric = {'found_in_memory': 0, 'found_in_think': 0, 'preserve_memory': 0, 'preserve_memory_in_think': 0,'total_num': 0}



            item['judge_sub_em'] = metrics['sub_em']
            item['judge_em'] = metrics['em']
            item['judge_f1'] = metrics['f1']
            
            item['found_in_memory'] = preserved_metric['found_in_memory']
            item['found_in_think'] = preserved_metric['found_in_think']
            item['preserve_memory'] = preserved_metric['preserve_memory']
            item['preserve_memory_in_think'] = preserved_metric['preserve_memory_in_think']
            
            
            scores['f1'].append(item['judge_f1'])
            scores['em'].append(item['judge_em'])
            scores['sub_em'].append(item['judge_sub_em'])
            scores['found_in_memory'].append(item['found_in_memory'])
            scores['found_in_think'].append(item['found_in_think'])
            scores['preserve_memory'].append(item['preserve_memory'])
            scores['preserve_memory_in_think'].append(item['preserve_memory_in_think'])

            
        else:
            if item['pred']:
                metrics = calc_metrics([item["pred"]], [item["answer"]])
                preserved_metric = calc_niah_preserve_metric([conversation_trace], [item["answer"]])
            else:
                metrics = {'sub_em': 0, 'total_num': 0}
                preserved_metric = {'found_in_memory': 0, 'found_in_think': 0, 'preserve_memory': 0, 'preserve_memory_in_think': 0,'total_num': 0}

            item['judge_sub_em'] = metrics['sub_em']
            item['found_in_memory'] = preserved_metric['found_in_memory']
            item['found_in_think'] = preserved_metric['found_in_think']
            item['preserve_memory'] = preserved_metric['preserve_memory']
            item['preserve_memory_in_think'] = preserved_metric['preserve_memory_in_think']


            scores['sub_em'].append(item['judge_sub_em'])
            scores['found_in_memory'].append(item['found_in_memory'])
            scores['found_in_think'].append(item['found_in_think'])
            scores['preserve_memory'].append(item['preserve_memory'])
            scores['preserve_memory_in_think'].append(item['preserve_memory_in_think'])
        item.pop('context');fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        grouped[item["_id"]].append(item)
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
    # print(f"Total: {len(data)}")
    print(f"Total: {len(expanded_data)}")
    print(f"Total questions: {len(grouped)}")
    sub_em_rates = {}

    for qid, items in grouped.items():
        total = len(items)
        hit = sum(it.get("judge_sub_em", 0) for it in items)
        sub_em_rates[qid] = hit / total if total > 0 else 0.0
    
    at_least_one_hit = sum(
        1 for items in grouped.values()
        if any(it.get("judge_sub_em", 0) == 1 for it in items)
    )

    print(f"Any-hit rate: {at_least_one_hit / len(grouped):.4f}")
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
























# Read SQuAD QA dataset
def read_squad(file):
    with open(file) as f:
        data = json.load(f)
        
    total_docs = [p['context'] for d in data['data'] for p in d['paragraphs']]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}

    total_qas = []
    for d in data['data']:
        more_docs = [total_docs_dict[p['context']] for p in d['paragraphs']]
        for p in d['paragraphs']:
            for qas in p['qas']:
                if not qas['is_impossible']:
                    total_qas.append({
                        'query': qas['question'],
                        'outputs': [a['text'] for a in qas['answers']],
                        'context': [total_docs_dict[p['context']]],
                        'more_context': [idx for idx in more_docs if idx != total_docs_dict[p['context']]]
                    })
                        
    return total_qas, total_docs

# Read Hotpot QA dataset
def read_hotpotqa(file):
    with open(file) as f:
        data = json.load(f)

    total_docs = [f"{t}\n{''.join(p)}" for d in data for t, p in d['context']]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}
    
    total_qas = []
    for d in data:
        total_qas.append({
            'query': d['question'],
            'outputs': [d['answer']],
            'context': [total_docs_dict[f"{t}\n{''.join(p)}"] for t, p in d['context']],
        })
        
    return total_qas, total_docs
import json

def load_jsonl(path):
    items = []
    with open(path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

def read_musique(file):
    data = load_jsonl(file)

    total_docs = [f"{paragraph.get('title','')}\n{paragraph.get('paragraph_text','')}" for d in data for paragraph in d['paragraphs']]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}
    
    total_qas = []
    for d in data:
        total_qas.append({
            'query': d['question'],
            'outputs': [d['answer']],
            'context': [total_docs_dict[f"{paragraph.get('title','')}\n{paragraph.get('paragraph_text','')}"] for paragraph in d['paragraphs']],
        })
        
    return total_qas, total_docs

import pandas as pd
def read_2wiki(file):
    
    data = pd.read_parquet(file)
    total_docs = [
        f"{t}\n{''.join(p)}"
        for _, row in data.iterrows()
        for t, p in zip(row["context"]["title"],
                        row["context"]["sentences"])
    ]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}
    
    total_qas = []
    for _, row in data.iterrows():
        total_qas.append({
            'query': row['question'],
            'outputs': [row['answer']],
            'context': [total_docs_dict[f"{t}\n{''.join(p)}"] for t, p in zip(row["context"]["title"], row["context"]["sentences"])],
        })
        
    return total_qas, total_docs
DOCS = None
def set_context(item):
    global DOCS
    if DOCS is None:
        if args.split == "qa_1":
            _, DOCS = read_squad("../memory_data/squad.json")
        elif args.split == "qa_2":
            _, DOCS = read_hotpotqa("../memory_data/hotpotqa_dev.json")
        elif args.split == "qa_3":
            _, DOCS = read_musique("../memory_data/musique_ans_v1.0_dev.jsonl")
        elif args.dataset == '2wiki':
            _, DOCS = read_2wiki("../memory_data/2wiki_dev.parquet")
        else:
            raise ValueError
    all_docs = [DOCS[idx] for idx in item['context']]
    DOCUMENT_PROMPT = "Document {i}:\n{document}"
    context = '\n\n'.join([DOCUMENT_PROMPT.format(i=i+1, document=d) for i, d in enumerate(all_docs)])
    item['context'] = context
    return item

def main():
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)
    out_file = os.path.join(args.save_dir, args.save_file + ".jsonl")

    dataset = concatenate_datasets([
            load_dataset("json", data_files=f"{DATAROOT}/eval_{args.split}_{args.length}.json", split="train"),
        ])
    if isinstance(dataset[0]['context'], list):
        dataset = [[set_context(item) for item in dataset]]
    print(f"original data len {len(dataset)}")
    # 通过深拷贝生成新数据集
    import copy
    dataset = [copy.deepcopy(item) for _ in range(args.sampling) for item in dataset]
    print(f"sampling data len {len(dataset)}")
    data_all = []
    for idx, item in enumerate(dataset):
        item["_id"] = idx  # 现在每个 item 是独立对象
        data_all.append(item)

    print(data_all[0]["_id"])
    print(data_all[-1]["_id"])

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
    # data = data_all[:5]
    get_pred_with_conversation_trace(data, args, out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="niah_single_1", choices=["niah_single_1","niah_single_2","niah_single_3","niah_multikey_1","niah_multikey_2",
    "niah_multikey_3","niah_multivalue","niah_multiquery","vt","cwe","fwe","qa_1","qa_2","qa_3","qa_4"], help="split of the dataset")
    parser.add_argument("--length", type=int, default=8192, choices=[8192,16384,32768,65536,131072,262144,524288,1048576,1048576*2, 1048576*4, 10000000],)
    parser.add_argument("--save_dir", "-s", type=str, default="results/ruler_general")
    parser.add_argument("--save_file", "-f", type=str, default="Qwen2.5-7B-Instruct-recurrent")
    parser.add_argument("--model", "-m", type=str, default="Qwen2.5-7B-Instruct")
    parser.add_argument("--tokenizer", "-t", type=str, default="/mnt/hdfs/hongli/model/Qwen2.5-7B-Instruct")
    parser.add_argument("--n_proc", "-n", type=int, default=64)
    parser.add_argument("--api", "-a", type=str, default="recurrent")
    parser.add_argument("--sampling", "-p", type=int, default=1)
    parser.add_argument('--force', action='store_true', help='force to overrite')
    args = parser.parse_args()
    main()