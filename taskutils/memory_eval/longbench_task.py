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

dataset2prompt={
    "narrativeqa": "answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\n \n\nQuestion: {input}",
    "hotpotqa": "{input}",
    "2wikimqa": "{input}",
    "musique": "{input}",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.{input}",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences\n\nQuery: {input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. \n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences\n\n{input}",
   
}

from utils.metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
    qa_sub_em_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        try:
            score = 0.
            if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
                prediction = prediction.lstrip('\n').split('\n')[0]
            for ground_truth in ground_truths:
                score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
            total_score += score
        except:
            continue
    return round(100 * total_score / len(predictions), 2)

def sub_em(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        try:
            score = 0.
            if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
                prediction = prediction.lstrip('\n').split('\n')[0]
            for ground_truth in ground_truths:
                score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
            total_score += score
        except:
            continue
    return round(total_score / len(predictions), 2)



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
    # for item in data:
    #     coro = async_query_llm(item, model, tokenizer, temperature=0, top_p=1)
    #     coros.append(coro)
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
            top_p=1
        )
        coros.append(coro)
    from utils.aio import async_main, close_async_client
    import uvloop
    outputs = uvloop.run(async_main(coros, args.n_proc))
    uvloop.run(async_main([close_async_client()]))



    from collections import defaultdict
    grouped = defaultdict(list)
    total_score = 0.
    fout = open(out_file, 'w' if args.force else 'a', encoding='utf-8')    
    for i, (output, item) in enumerate(zip(outputs, expanded_data)):
    # for i, (output, item) in enumerate(zip(outputs, data)):
        if output == '':
            continue
        # assume output is a dict from async_query_llm
        response = output.get('final', '').strip()
        conversation_trace = output.get('conversation', [])

        pred, _ = extract_solution(response)
        item['response'] = response
        item['answer'] = item.pop("answers")
        item['pred'] = extract_answer(pred) if pred else extract_answer(response)
        
        item['conversation'] = conversation_trace
        score = scorer(args.split, [item['pred']], [item['answer']], item['all_classes'])
        item['judge_f1'] = score/100
        item['judge_sub_em'] = sub_em(args.split, [item['pred']], [item['answer']], item['all_classes'])
        
        total_score+=score
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
            print(score)
            print("="*40 + "New Item End" + "="*40)
    print(f"longbench [{args.split}]")
    
    # print(f" {round(total_score /len(data), 2)}")
    print(f" {round(total_score /len(expanded_data), 2)}")
    # print(f"Total: {len(data)}")
    
    print(f"Total: {len(expanded_data)}")
    print(f"Total questions: {len(grouped)}")
    sub_em_rates = {}
    best_samples = []

 
    for qid, items in grouped.items():
        total = len(items)
        hit = sum(it.get("judge_sub_em", 0) for it in items)
        sub_em_rates[qid] = hit / total if total > 0 else 0.0
        # 选 F1 最高的 rollout
        best_item = max(
            items,
            key=lambda it: it.get("judge_f1", 0)
        )

        best_samples.append(best_item)
    
    at_least_one_hit = sum(
        1 for items in grouped.values()
        if any(it.get("judge_sub_em", 0) == 1 for it in items)
    )

    print(f"Any-hit rate: {at_least_one_hit / len(grouped):.4f}")





def main():
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)
    out_file = os.path.join(args.save_dir, args.save_file + ".jsonl")
    data_path = f"{DATAROOT}/data/{args.split}.jsonl"
    
    
    dataset = load_dataset("json", data_files=data_path, split="train")
    data_all = []
    for idx, item in enumerate(dataset):
        item["_id"] = idx  # 现在每个 item 是独立对象
        data_all.append(item)

    import copy
    dataset = [copy.deepcopy(item) for _ in range(args.sampling) for item in dataset]
    print(f"sampling data len {len(dataset)}")
    print(data_all[0]["_id"])
    print(data_all[-1]["_id"])
    # prompt_format = dataset2prompt[args.split]
    # cache
    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            has_data = {json.loads(line)["_id"]: 0 for line in f}
    data = []
    for item in data_all:
        if item["_id"] not in has_data or args.force:
            # print(item.get("input",""))
            # item["input"] = prompt_format.format(input=item.get("input",""))
            data.append(item)
        elif args.force:
            data.append(item)
    # ✅ Only process the first data item for quick testing
    # if len(data) > 0:
        # data = [data_all[0]]

    get_pred_with_conversation_trace(data, args, out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="niah_single_1", choices=["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", 
                    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", 
                    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"], help="split of the dataset")
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