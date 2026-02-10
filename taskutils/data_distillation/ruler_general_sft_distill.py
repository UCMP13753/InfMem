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



def get_pred(data,  out_file):
    model = "qwen32b"
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    from utils.infmem import async_query_llm_multi_turn as async_query_llm
    from utils import extract_answer


    coros = []
    for item in data:
        coro = async_query_llm(item, model, tokenizer, temperature=0.7, top_p=0.95)
        coros.append(coro)
    from utils.aio import async_main, close_async_client
    import uvloop
    outputs = uvloop.run(async_main(coros, 64))
    uvloop.run(async_main([close_async_client()]))




    from collections import defaultdict
    fout = open(out_file, 'w', encoding='utf-8')


    for i, (output, item) in enumerate(zip(outputs, data)):
        if output == '':
            continue

        # assume output is a dict from async_query_llm
        response = output.get('final', '').strip()
        conversation_trace = output.get('conversation', [])


        pred, _ = extract_solution(response)
        item['response'] = response
        item['answer'] = item.pop("outputs")
        item['pred'] = extract_answer(pred) if pred else extract_answer(response)
        item['conversation'] = conversation_trace

        
        
        if item['pred']:
            metrics = calc_qa_metrics([item["pred"]], [item["answer"][0]])
        else:
            metrics = {'f1': 0, 'prec': 0,'recall': 0, 'em': 0,'sub_em': 0, 'total_num': 0}
        item['judge_sub_em'] = metrics['sub_em']
        item['judge_em'] = metrics['em']
        item['judge_f1'] = metrics['f1']
        # scores['em'].append(item['judge_em'])
        # scores['f1'].append(item['judge_f1'])
        # scores['sub_em'].append(item['judge_sub_em'])
        
        
        item.pop('context')
        fout.write(json.dumps(item, ensure_ascii=False) + '\n')



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

DOCS = None
def set_context(item):
    global DOCS
    if DOCS is None:
        _, DOCS = read_squad("../memory_data/downloaded_data/squad_train.json")

    all_docs = [DOCS[idx] for idx in item['context']]
    DOCUMENT_PROMPT = "Document {i}:\n{document}"
    context = '\n\n'.join([DOCUMENT_PROMPT.format(i=i+1, document=d) for i, d in enumerate(all_docs)])
    item['context'] = context
    return item

def distill_squad(path, out_file,num_samples):
    
    with open(path, "r", encoding="utf-8") as f:
        data_list = json.load(f)  # list of dicts

    dataset = Dataset.from_list(data_list)
    dataset = concatenate_datasets([dataset])

    print(f"Loaded {len(dataset)} items")

    if isinstance(dataset[0]['context'], list):
        dataset = [[set_context(item) for item in dataset]]
    print(f"original data len {len(dataset)}")
    # 通过深拷贝生成新数据集
    import copy
    dataset = [copy.deepcopy(item) for _ in range(1) for item in dataset]
    print(f"sampling data len {len(dataset)}")
    data_all = []
    for idx, item in enumerate(dataset):
        item["_id"] = idx  # 现在每个 item 是独立对象
        data_all.append(item)

    print(data_all[0]["_id"])
    print(data_all[-1]["_id"])

    # cache
    has_data = {}
    # if os.path.exists(out_file):
    #     with open(out_file, encoding='utf-8') as f:
    #         has_data = {json.loads(line)["_id"]: 0 for line in f}
    data = []
    for item in data_all:
        data.append(item)
    # ✅ Only process the first data item for quick testing
    
    data = data[:num_samples]
    get_pred(data, out_file)


