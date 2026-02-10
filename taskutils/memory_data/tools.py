import json

def analyze_dataset(path):
    total = 0
    em_1 = 0
    user_contains = 0
    both = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)
            total += 1

            # ---- Ground truth ----
            gt = obj.get("answers", [""])[0].strip().lower()

            # ---- Extract user content ----
            user_content = ""
            for turn in obj.get("conversation", []):
                if turn.get("role") == "user":
                    user_content += " " + turn.get("content", "")

            user_content = user_content.lower()

            # ---- Check whether user content contains the answer ----
            contains = gt in user_content
            if contains:
                user_contains += 1

            # ---- Check judge_em ----
            if obj.get("judge_em", 0) == 1:
                em_1 += 1

            # ---- Check both conditions ----
            if contains and obj.get("judge_em", 0) == 1:
                both += 1

    # ---- Compute ratios ----
    ratio_user_contains = user_contains / total if total > 0 else 0
    ratio_judge_em = em_1 / total if total > 0 else 0
    ratio_both = both / total if total > 0 else 0

    return {
        "total": total,
        "user_contains_answer_count": user_contains,
        "judge_em_count": em_1,
        "both_count": both,
        "ratio_user_contains": ratio_user_contains,
        "ratio_judge_em": ratio_judge_em,
        "ratio_both": ratio_both
    }

def process_jsonl(input_path_list, output_path):
    fout = open(output_path, "w", encoding="utf-8")
    new_index = 0  # 用于重新编号 index
    for input_file in input_path_list:
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                obj = json.loads(line)


                # ---- Ground truth ----
                gt = obj.get("answer", "")
                if isinstance(gt, list):
                    gt = gt[0]
                gt = gt.strip().lower()
                question = {}
                # ---- Collect user content ----
                for turn in obj.get("conversation", []):
                    if turn.get("role") == "user":
                        question = turn
                        break

                # ---- Skip if gt not in user content ----
                if gt not in question.get("content","").lower():
                    continue

                # ---- Add synthetic assistant answer ----


                synthetic = {
                    "role": "assistant",
                    "content": f"Therefore, the answer is {gt}"
                }

                # conversation += synthetic answer
                new_conversation = [question, synthetic]
                obj["conversation"] = new_conversation

                 # ---- Construct new object ----
                new_obj = {
                    "index": new_index,
                    "input": obj.get("input", ""),
                    "old_response": obj.get("response", ""),
                    "correct_answer": gt,
                    "messages": new_conversation,
                }

                new_index += 1

                # ---- Write out ----
                fout.write(json.dumps(new_obj, ensure_ascii=False) + "\n")

    fout.close()
    print(f"Processed file saved to {output_path}")

if __name__ == "__main__":
    input_path_list = ["/mnt/nfs/datasets/mem_agent/final_round_qa_pairs/MemAgent_hotpot_final_pair.jsonl", "/mnt/nfs/datasets/mem_agent/final_round_qa_pairs/MemAgent_squad_final_pair.jsonl"] # 修改成你的文件名
    onput_path = "/mnt/nfs/datasets/mem_agent/final_round_qa_pairs/MemAgent_hotpot+squad_final_pair_synthetic.jsonl"  # 修改成你的文件名
    process_jsonl(input_path_list, onput_path)

    # stats = analyze_dataset(path)
    # for k, v in stats.items():
    #     print(f"{k}: {v}")