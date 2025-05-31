import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import json
import sys
import warnings
import os
import gc
from collections import defaultdict
import csv

warnings.filterwarnings("ignore")

def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

model_id = "./model"

print(f"Loading model '{model_id}'...")
try:
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model.tokenizer = tokenizer
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

def load_json_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_reasoning_steps_text(steps, path_index, is_modified=False):
    for path_data in steps:
        if path_data["path_index"] == path_index:
            steps_text = []
            for step in path_data["path"]:
                if is_modified:
                    steps_text.append(f"Step {step['reasoning step']} ({step['reasoning type']}): {step['rationale_fix']}")
                else:
                    steps_text.append(f"Step {step['reasoning step']} ({step['reasoning type']}): {step['rationale']}")
            return "\n".join(steps_text)
    return ""

try:
    json_data = load_json_data("MMRB_data_reward.json")
except FileNotFoundError:
    print("Error: MMRB_data_reward.json not found.")
    sys.exit(1)

stats = {
    "total_samples": 0,
    "total_paths": 0,
    "first_choice": 0,
    "second_choice": 0,
    "error_processing": 0,
    "by_subtask": defaultdict(lambda: {"total": 0, "first": 0, "second": 0, "error": 0})
}

HD_NUM = 9

for sample_idx, sample in enumerate(json_data):
    stats["total_samples"] += 1
    
    available_path_indices = set()
    if "modified_reasoning_steps" not in sample or "raw_reasoning_steps" not in sample:
        print(f"Warning: Sample {sample_idx + 1} (ID: {sample.get('id', 'N/A')}) is missing 'modified_reasoning_steps' or 'raw_reasoning_steps'. Skipping.")
        num_potential_paths_m = len(sample.get("modified_reasoning_steps", []))
        num_potential_paths_r = len(sample.get("raw_reasoning_steps", []))
        num_potential_paths = max(num_potential_paths_m, num_potential_paths_r)
        if num_potential_paths == 0 and sample.get('id'):
             num_potential_paths = 1
        
        stats["error_processing"] += num_potential_paths
        stats["by_subtask"][sample.get("subtask", "Unknown")]["error"] += num_potential_paths
        continue

    for path_data in sample["modified_reasoning_steps"]:
        available_path_indices.add(path_data["path_index"])

    image_paths = sample.get("image_paths", [])
    valid_image_paths = []
    if image_paths:
        for p in image_paths:
            if isinstance(p, str) and os.path.exists(p):
                valid_image_paths.append(p)
            else:
                print(f"Warning: Image path '{p}' (type: {type(p)}) not found or invalid for sample {sample_idx + 1}. It will be excluded.")
        if not valid_image_paths and image_paths:
             print(f"Warning: No valid image paths found for sample {sample_idx + 1} though image_paths was provided. Proceeding without images.")
    
    original_question_text = sample.get('question', '')
    if not original_question_text:
        print(f"Warning: Sample {sample_idx + 1} (ID: {sample.get('id', 'N/A')}) is missing 'question'. Skipping paths for this sample.")
        num_potential_paths = len(available_path_indices) if available_path_indices else 1
        stats["error_processing"] += num_potential_paths
        stats["by_subtask"][sample.get("subtask", "Unknown")]["error"] += num_potential_paths
        continue
        
    image_placeholders_string = ""
    if valid_image_paths:
        for i in range(len(valid_image_paths)):
            image_placeholders_string += f"{{image#{i}}}: <ImageHere>\n"
    
    question_text_with_placeholders = image_placeholders_string + original_question_text
    options = sample.get("options", [])
    answer = sample.get("answer", "")

    if options:
        options_string = "\n".join(options) + "\n"
    else:
        options_string = ""

    question_text_with_placeholders += "\n" + options_string
    if answer:
        question_text_with_placeholders += f"Answer: {answer}\n"

    for path_idx, path_index in enumerate(sorted(list(available_path_indices))):
        stats["total_paths"] += 1
        subtask = sample.get("subtask", "Unknown")
        stats["by_subtask"][subtask]["total"] += 1
        
        print(f"\nProcessing sample {sample_idx + 1}/{len(json_data)}, Path Index: {path_index} (Subtask: {subtask})")
        if valid_image_paths:
            print(f"  Using {len(valid_image_paths)} valid image(s).")
            if image_placeholders_string:
                 print(f"  Image placeholders added to question:\n{image_placeholders_string.strip()}")
        else:
            print("  No valid images used.")


        modified_steps_text = get_reasoning_steps_text(sample["modified_reasoning_steps"], path_index, True)
        raw_steps_text = get_reasoning_steps_text(sample["raw_reasoning_steps"], path_index, False)

        if not modified_steps_text or not raw_steps_text:
            print(f"Warning: Missing modified or raw steps for sample {sample_idx + 1}, path_index {path_index}. Skipping this path.")
            stats["error_processing"] += 1
            stats["by_subtask"][subtask]["error"] += 1
            continue

        chat_modified = [
            {"role": "user", "content": question_text_with_placeholders},
            {"role": "assistant", "content": modified_steps_text}
        ]
        chat_raw = [
            {"role": "user", "content": question_text_with_placeholders},
            {"role": "assistant", "content": raw_steps_text}
        ]
        
        try:
            autocast_enabled = torch.cuda.is_available()
            current_autocast_device_type = "cuda" if autocast_enabled else "cpu"

            with torch.autocast(device_type=current_autocast_device_type, dtype=torch.float16, enabled=autocast_enabled):
                score_modified = model.get_score(chat_modified, valid_image_paths if valid_image_paths else [], hd_num=HD_NUM)
                if isinstance(score_modified, torch.Tensor):
                    score_modified = score_modified.item()

                score_raw = model.get_score(chat_raw, valid_image_paths if valid_image_paths else [], hd_num=HD_NUM)
                if isinstance(score_raw, torch.Tensor):
                    score_raw = score_raw.item()
            
            print(f"  Score for Modified (Response 1): {score_modified:.4f}")
            print(f"  Score for Raw (Response 2): {score_raw:.4f}")

            if score_modified > score_raw:
                stats["first_choice"] += 1
                stats["by_subtask"][subtask]["first"] += 1
                print("  Model evaluation: Response 1 (Modified) is better")
            else: 
                stats["second_choice"] += 1
                stats["by_subtask"][subtask]["second"] += 1
                print("  Model evaluation: Response 2 (Raw) is better (or score is equal to Response 1)")

        except Exception as e:
            print(f"  Error evaluating sample {sample_idx + 1}, path_index {path_index}: {e}")
            stats["error_processing"] += 1
            stats["by_subtask"][subtask]["error"] += 1
            exc_type, exc_obj, exc_tb = sys.exc_info()
            if exc_tb:
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(f"  Error type: {exc_type}, File: {fname}, Line number: {exc_tb.tb_lineno}")
            else:
                print(f"  Error type: {exc_type}")

        finally:
            clear_gpu_memory()

print("\nFinal Statistics:")
print(f"Total samples (attempted to process): {stats['total_samples']}")
print(f"Total paths evaluated (attempted to process): {stats['total_paths']}")
print(f"Times Response 1 (Modified) chosen: {stats['first_choice']}")
print(f"Times Response 2 (Raw) chosen (including equal scores): {stats['second_choice']}")
print(f"Times errors occurred during processing: {stats['error_processing']}")

valid_paths_evaluated = stats['first_choice'] + stats['second_choice']

if valid_paths_evaluated > 0:
    print(f"\nBased on {valid_paths_evaluated} successfully evaluated paths:")
    print(f"  Percentage Response 1 chosen: {stats['first_choice']/valid_paths_evaluated*100:.2f}%")
    print(f"  Percentage Response 2 chosen: {stats['second_choice']/valid_paths_evaluated*100:.2f}%")
if stats['total_paths'] > 0:
     print(f"  Processing error rate (based on total paths): {stats['error_processing']/stats['total_paths']*100:.2f}%")

print("\nStatistics by Subtask:")
for subtask, s_stats in stats["by_subtask"].items():
    print(f"\nSubtask: {subtask}")
    print(f"  Total paths: {s_stats['total']}")
    print(f"  Response 1 chosen: {s_stats['first']}")
    print(f"  Response 2 chosen: {s_stats['second']}")
    print(f"  Errors: {s_stats['error']}")
    
    sub_valid_paths = s_stats['first'] + s_stats['second']
    if sub_valid_paths > 0:
        print(f"    Response 1 Percentage: {s_stats['first']/sub_valid_paths*100:.2f}%")
        print(f"    Response 2 Percentage: {s_stats['second']/sub_valid_paths*100:.2f}%")
    if s_stats['total'] > 0:
        print(f"    Error Rate: {s_stats['error']/s_stats['total']*100:.2f}%")

csv_file_path = "internlm_reward_model_ixc.csv"
try:
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        writer.writerow(["Overall Statistics", "Value"])
        writer.writerow(["Total Samples Processed", stats['total_samples']])
        writer.writerow(["Total Paths Evaluated", stats['total_paths']])
        writer.writerow(["Times Response 1 Chosen", stats['first_choice']])
        writer.writerow(["Times Response 2 Chosen (incl. equal score)", stats['second_choice']])
        writer.writerow(["Times Error Processing", stats['error_processing']])
        if valid_paths_evaluated > 0:
            writer.writerow(["Percentage Response 1 Chosen (of valid)", f"{stats['first_choice']/valid_paths_evaluated*100:.2f}%"])
            writer.writerow(["Percentage Response 2 Chosen (of valid, incl. equal)", f"{stats['second_choice']/valid_paths_evaluated*100:.2f}%"])
        if stats['total_paths'] > 0:
            writer.writerow(["Percentage Error (of total paths)", f"{stats['error_processing']/stats['total_paths']*100:.2f}%"])
        writer.writerow([])

        writer.writerow(["Subtask", "Total Paths", "Response 1 Chosen", "Response 2 Chosen (incl. equal)", "Errors",
                         "Response 1 % (of valid)", "Response 2 % (of valid, incl. equal)", "Error % (of total)"])
        
        for subtask, s_stats in stats["by_subtask"].items():
            sub_valid = s_stats['first'] + s_stats['second']
            row = [
                subtask,
                s_stats['total'],
                s_stats['first'],
                s_stats['second'],
                s_stats['error']
            ]
            if sub_valid > 0:
                row.extend([
                    f"{s_stats['first']/sub_valid*100:.2f}%",
                    f"{s_stats['second']/sub_valid*100:.2f}%",
                ])
            else:
                row.extend(["N/A", "N/A"])
            if s_stats['total'] > 0:
                row.append(f"{s_stats['error']/s_stats['total']*100:.2f}%")
            else:
                row.append("N/A")
            writer.writerow(row)
            
    print(f"\nStatistics saved to: {csv_file_path}")

except IOError:
    print(f"Error: Could not write CSV file to path {csv_file_path}. Please check permissions or path.")
except Exception as e:
    print(f"An unknown error occurred while saving CSV: {e}")

print("\nScript execution finished.")