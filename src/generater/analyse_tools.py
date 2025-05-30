import json
from tqdm import tqdm
from typing import List, Dict, Any
from collections import Counter, defaultdict


def process_and_filter_json_data(
    input_data_path: str,
    output_file_name: str,
    target_source: str,
    target_subtask: str,
    image_path_prefix: str = "./eval_images_fix/",
    question_suffix_to_remove: str = "</s>",
    start_index_in_input: int = 0
    ) -> None:

    try:
        with open(input_data_path, "r", encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_data_path}'.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_data_path}'. Please check the file's content.")
        return

    generated_data: List[Dict[str, Any]] = []
    output_index = -1

    for i, row in enumerate(tqdm(data[start_index_in_input:], desc=f"Processing {input_data_path}")):
        if not isinstance(row, dict) or "metadata" not in row or "conversations" not in row or "image" not in row:
            print(f"Warning: Skipping malformed row at original index {i}. Missing expected keys.")
            continue

        if not isinstance(row["metadata"], dict) or "dataset" not in row["metadata"]:
            print(f"Warning: Skipping row at original index {i}. Missing 'dataset' in 'metadata'.")
            continue
        
        taskname = row["metadata"]["dataset"]
        if taskname != target_subtask:
            continue
        
        output_index += 1 

        question_value = row["conversations"][0]["value"]
        if question_value.endswith(question_suffix_to_remove):
            question = question_value[:-len(question_suffix_to_remove)] + "."
        else:
            question = question_value + "." 


        answer = row["conversations"][1]["value"]
        image_paths = [f"{image_path_prefix}{path}" for path in row["image"]] 

        generated_data.append({
            "source": target_source,
            "subtask": target_subtask,
            "index": output_index,
            "question": question,
            "options": [], 
            "answer": answer,
            "image_paths": image_paths,
            "reasoning_steps": []
        })

    try:
        with open(output_file_name, "w", encoding='utf-8') as f:
            json.dump(generated_data, f, ensure_ascii=False, indent=2)
        print(f"\nSuccessfully generated {len(generated_data)} entries to '{output_file_name}'.")
    except IOError as e:
        print(f"Error: Could not write to output file '{output_file_name}': {e}")


def count_datasets_in_json(file_path: str):

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{file_path}'. Please ensure it's a valid JSON file.")
        return

    dataset_counter = Counter()

    for item in data:
        if isinstance(item, dict) and 'metadata' in item and isinstance(item['metadata'], dict):
            metadata = item['metadata']
            if 'dataset' in metadata:
                dataset = metadata['dataset']
                dataset_counter[dataset] += 1
    
    if not dataset_counter:
        print(f"No 'dataset' information found in 'metadata' in '{file_path}'.")
        return

    print(f"Dataset statistics for '{file_path}':")
    for dataset, count in dataset_counter.items():
        print(f"Dataset: {dataset}, Count: {count}")


def evaluate_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'reasoning_steps': 0})

    for item in data:
        sub_task = item.get('sub_task')
        try:
            gold_answer = item['conversations'][1]['value']
            model_answer = item['evaluate_answer']['Answer'][0]
            reasoning_steps = len(item['evaluate_answer']['reasoning steps'])

            stats[sub_task]['total'] += 1
            stats[sub_task]['reasoning_steps'] += reasoning_steps

            if gold_answer.strip() == model_answer.strip():
                stats[sub_task]['correct'] += 1
        except (KeyError, IndexError, TypeError):
            print(f"Skipping an item due to missing fields: {item}")

    for sub_task, values in stats.items():
        total = values['total']
        correct = values['correct']
        avg_reasoning = values['reasoning_steps'] / total if total > 0 else 0
        accuracy = correct / total if total > 0 else 0

        print(f"Sub-task: {sub_task}")
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  Avg Reasoning Steps: {avg_reasoning:.2f}")