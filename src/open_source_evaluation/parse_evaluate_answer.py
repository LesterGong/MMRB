import json
import re
import os
from collections import defaultdict


def extract_predicted_answer(cot_answer):
    if not isinstance(cot_answer, str):
        cot_answer = str(cot_answer)

    strict_match = re.search(r'Answer\s*[\s*([A-Za-z])\s*]', cot_answer, re.IGNORECASE)
    if strict_match:
        return strict_match.group(1).strip().upper()

    strict_match = re.search(r'Answer\s*[\s*(.*?)\s*]', cot_answer, re.IGNORECASE)
    if strict_match:
        return strict_match.group(1).strip().upper()

    boxed_match = re.search(r'\\boxed\{(.*?)\}', cot_answer, re.IGNORECASE)
    if boxed_match:
        return boxed_match.group(1).strip().upper()

    match = re.match(r'^[A-Z]$', cot_answer)
    if match:
        answer = match.group(0)
        return answer

    relaxed_patterns = [
        r'Final\s+answer\s*(?:is)?\s*[:：]?\s*\**\s*(\w+)',
        r'\*\*Final Answer\*\*\s*[:：]?\s*(\w+)',
        r'The correct answer is\s+(\**\w+\**)',
        r'answer is\s+(\**\w+\**)',
        r'The correct choice is\s+(\**\w+\**)',
        r'The answer is\s+(\**\w+\**)',
        r'Answer\s*[:：]?\s*(\w+)$',
        r'\\boxed\{([A-Z])\}',
        r'Answer\s*[:：]?\s*<([A-Z])>',
        r'Answer\s*[:：]?\s*<([A-Z])<',
        r'Answer[\s*([A-Z])\s*]',
        r'Answer[\s*(【A-Z】)\s*]',
        r'Answer\\[\s*([A-Z])\s*\\]',
        r'Answer\s*[:：]?\s*\(([A-Z])\)',
        r'Answer\s*[:：]?\s*(.+?)(?:\s|$)',
        r'<\s*([\w\.]+)\s*>',
        r'Point\s+([A-H])\b',
        r'^([A-D])\s*[\s*([A-D])\s*$',
        r'^([A-D])\s*【\s*([A-D])\s*】$',
        r'^([A-D])\s*【\s*([A-D])?$',
        r'<\s*([A-D])\s*$',
        r'<\s*(\d+(?:\.\d+)?)<',
        r'<\s*(\d+(?:\.\d+)?)$',
        r'[Oo]ption\s+([A-D])\b',
        r'^([A-D])\s*[\s*([A-D])\s*$',
        r'^([A-D])\s*[\s*\d+\s*$',
        r'<\s*([A-Z]{1,5})\s*<',
        r'<\s*([A-D])\s*$',
    ]

    for pattern in relaxed_patterns:
        match = re.search(pattern, cot_answer, re.IGNORECASE)
        if match:
            ans = match.groups()[-1].strip().strip('*')
            return ans.upper() if len(ans) == 1 and ans.isalpha() else ans.lower()
        
    match = re.match(r'^([A-H])[\.\):：[]', cot_answer)
    if match:
        answer = match.group(1)
        return answer
    
    return None

def evaluate_predictions(data):
    task_correct = defaultdict(int)
    task_total = defaultdict(int)
    
    extracted_count = 0
    unmatched_count = 0

    for item in data:
        gold = str(item.get("answer", "")).strip().upper()
        pred = extract_predicted_answer(item.get("CoT_answer", ""))
        
        if pred is not None:
            pred = pred.upper()
            extracted_count += 1
        else:
            unmatched_count += 1
        subtask = str(item.get("subtask", "unknown")).strip()
        if pred is not None:
            if pred == gold or pred in gold or gold in pred:
                task_correct[subtask] += 1
            task_total[subtask] += 1

    results = {}
    total_correct = sum(task_correct.values())
    total_total = sum(task_total.values())

    for task in task_total:
        acc = task_correct[task] / task_total[task] if task_total[task] > 0 else 0.0
        results[task] = {
            "correct": task_correct[task],
            "total": task_total[task],
            "accuracy": round(acc * 100, 1)
        }

    overall_accuracy = round((total_correct / total_total) * 100, 2) if total_total > 0 else 0.0
    results["Overall"] = {
        "correct": total_correct,
        "total": total_total,
        "accuracy": overall_accuracy,
        "extracted_count": extracted_count,
        "unmatched_count": unmatched_count
    }

    return results

def process_json_folder(folder_path):
    total_accuracy = 0.0
    file_count = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"❌ can not read {filename}：{e}")
                continue
                  
            results = evaluate_predictions(data)
            overall = results.get("Overall", {})
            print(f"📄 file：{filename[:]}, Acc:{overall.get('accuracy', 0.0)}%, Parse:{round(overall.get('extracted_count', 0)/len(data) * 100, 1)}%")
            print("-" * 50)
            total_accuracy += overall.get('accuracy', 0.0)
            file_count += 1

    if file_count > 0:
        average_accuracy = total_accuracy / file_count
        print(f"📊 the average accuracy across all files is: {average_accuracy:.2f}%")


folder_path = "benchmark_results/OpenSource_noneReasoning/full_results/with_answer_prompt"
process_json_folder(folder_path)

folder_path = "benchmark_results/OpenSource_noneReasoning/full_results/with_cot_answer_prompt"
process_json_folder(folder_path)
