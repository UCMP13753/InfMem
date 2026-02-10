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

from utils import extract_solution,update_answer
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


def get_pred(data, out_file):
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
    scores = defaultdict(list)
    fout = open(out_file, 'w', encoding='utf-8')
    valid_conversations = []

    for i, (output, item) in enumerate(zip(outputs, data)):
        if not output or output == '':
            continue

        # assume output is a dict from async_query_llm
        response = output.get('final', '').strip()
        conversation_trace = output.get('conversation', [])

        pred, _ = extract_solution(response)
        item['response'] = response
        item['answer'] = item["answers"][0]
        item['pred'] = extract_answer(pred) if pred else extract_answer(response)
        item['conversation'] = conversation_trace

        print(conversation_trace)
        metrics = calc_metrics([item["pred"]], [item["answer"]])
        item['judge_f1'] = metrics['f1']
        item['judge_em'] = metrics['em']
        item['judge_sub_em'] = metrics['sub_em']
        # ✅ keep conversation only if correct
        # if metrics['sub_em'] > 0:
        #     valid_conversations.append({
        #         "conversation": conversation_trace,
        #         "final_answer": item['pred'],
        #         "ground_truth": item['answer'],
        #         "metrics": metrics

        #     })

        # optional: write per-item as before
        item.pop('context')
        fout.write(json.dumps(item, ensure_ascii=False) + '\n')


def distill_hotpot(path, out_file,num_samples):
    
    

    with open(path, "r", encoding="utf-8") as f:
        data_list = json.load(f)  # list of dicts

    dataset = Dataset.from_list(data_list)
    dataset = concatenate_datasets([dataset])

    print(f"Loaded {len(dataset)} items")
    print(f"original data len {len(dataset)}")
    # 通过深拷贝生成新数据集
    import copy
    dataset = [copy.deepcopy(item) for _ in range(1) for item in dataset]
    print(f"sampling data len {len(dataset)}")

    data_all = []
    for idx, item in enumerate(dataset):
        item["_id"] = idx  # 现在每个 item 是独立对象
        data_all.append(item)
        

    data = []
    for item in data_all:
            data.append(item)
    data = data[:num_samples]
    get_pred(data, out_file)

