import torch
import re
import os
import numpy as np
from PIL import Image
import json
import sys
import warnings
import gc
from collections import defaultdict
import csv

from modelscope import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration, AutoConfig
from trl import AutoModelForCausalLMWithValueHead
from transformers.utils import cached_file
from safetensors import safe_open

from qwen_vl_utils import process_vision_info


MODEL_PATH = "model/skywork"
PROCESSOR_PATH = "model/skywork"
VALUE_HEAD_FILENAME = "value_head.safetensors"

DEVICE = "cuda:1"
USE_FLASH_ATTENTION_2 = True

print(f"Loading processor from '{PROCESSOR_PATH}'...")
try:
    processor = AutoProcessor.from_pretrained(PROCESSOR_PATH, trust_remote_code=True)
    print("Processor loaded successfully.")
except Exception as e:
    print(f"Error loading processor: {e}")
    sys.exit(1)

print(f"Loading model from '{MODEL_PATH}'...")
try:
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    if USE_FLASH_ATTENTION_2:
        try:
            import flash_attn
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("Attempting to use flash_attention_2.")
        except ImportError:
            print("flash_attn not installed. Falling back to default attention mechanism.")
            USE_FLASH_ATTENTION_2 = False

    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        device_map="cuda:1",
        **model_kwargs
    )
    print(f"Base model Qwen2_5_VLForConditionalGeneration loaded to {DEVICE}.")

    model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)
    print("Model wrapped with AutoModelForCausalLMWithValueHead.")

    vhead_file_path = os.path.join(MODEL_PATH, VALUE_HEAD_FILENAME)
    if not os.path.exists(vhead_file_path):
        print(f"Error: Value head file '{vhead_file_path}' not found.")
        try:
            print(f"Attempting to locate value head using cached_file for '{MODEL_PATH}' with filename '{VALUE_HEAD_FILENAME}'")
            vhead_file_path = cached_file(path_or_repo_id=MODEL_PATH, filename=VALUE_HEAD_FILENAME)
            print(f"Located value head via cached_file: {vhead_file_path}")
        except Exception as cache_e:
            print(f"Error using cached_file for value head: {cache_e}")
            print("Please ensure the value_head.safetensors file is correctly placed or MODEL_PATH is a valid Hugging Face Hub identifier if applicable.")
            sys.exit(1)
            
    print(f"Loading value head parameters from '{vhead_file_path}'...")
    with safe_open(vhead_file_path, framework="pt", device="cpu") as f:
        vhead_params = {key: f.get_tensor(key) for key in f.keys()}
    
    missing_keys, unexpected_keys = model.load_state_dict(vhead_params, strict=False)
    if unexpected_keys:
        print(f"Warning: Unexpected keys in value_head checkpoint: {unexpected_keys}")
    if any("v_head" not in k for k in missing_keys):
         print(f"Warning: Some keys were missing while loading value_head: {missing_keys}")
    else:
        print(f"Value head parameters loaded. Missing keys related to non-value head parts (expected): {missing_keys}")

    model.requires_grad_(False)
    model.eval()
    print("Model (with value head) loaded, frozen, and set to eval mode successfully.")

except Exception as e:
    print(f"Error loading model: {e}")
    exc_type, exc_obj, exc_tb = sys.exc_info()
    if exc_tb:
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(f"  Error details: Type {exc_type}, File {fname}, Line {exc_tb.tb_lineno}")
    sys.exit(1)

def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_json_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_reasoning_steps_text(steps, path_index, is_modified=False):
    for path_data in steps:
        if path_data["path_index"] == path_index:
            steps_text_list = []
            for step in path_data["path"]:
                key = 'rationale_fix' if is_modified else 'rationale'
                steps_text_list.append(f"Step {step['reasoning step']} ({step['reasoning type']}): {step[key]}")
            return "\n".join(steps_text_list)
    return ""

def get_reward_score(model_obj, processor_obj, image_paths, question_text, answer_text, current_device):
    messages = [
        {
            "role": "user",
            "content": []
        },
        {
            "role": "assistant",
            "content": answer_text,
        },
    ]
    if image_paths:
        for img_path in image_paths:
            messages[0]["content"].append({"type": "image", "image": img_path})
    messages[0]["content"].append({"type": "text", "text": question_text})

    try:
        text_for_tokenizer = processor_obj.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor_obj(
            text=[text_for_tokenizer],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(current_device)
        
        with torch.inference_mode():
            model_outputs_tuple = model_obj(**inputs, return_dict=True, use_cache=False,
                                           output_attentions=False, output_hidden_states=False)
            
            if isinstance(model_outputs_tuple, tuple):
                value_predictions = model_outputs_tuple[-1]
            elif hasattr(model_outputs_tuple, 'value') and model_outputs_tuple.value is not None:
                 value_predictions = model_outputs_tuple.value
            else:
                print("Error: Model output is not a recognized tuple or object with a .value attribute.")
                print(f"Type of model_outputs_tuple: {type(model_outputs_tuple)}")
                if isinstance(model_outputs_tuple, tuple):
                    print(f"Length of tuple: {len(model_outputs_tuple)}")
                return None
            
            if value_predictions is None:
                print("Error: value_predictions is None after attempting to extract from model output.")
                return None
            
            if not isinstance(value_predictions, torch.Tensor):
                print(f"Error: Expected value_predictions to be a torch.Tensor, but got {type(value_predictions)}")
                return None

            last_token_indices = inputs["attention_mask"].sum(dim=-1, keepdim=True) - 1
            last_token_indices = torch.clamp(last_token_indices, min=0)
            
            if value_predictions.ndim == 1:
                value_predictions = value_predictions.unsqueeze(0) 
            
            final_scores = value_predictions.gather(dim=1, index=last_token_indices)
            score = final_scores.squeeze(-1)[0].item()
            return score

    except Exception as e:
        print(f"  Error during get_reward_score for answer '{answer_text[:50]}...': {e}")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        if exc_tb:
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(f"    Error details: Type {exc_type}, File {fname}, Line {exc_tb.tb_lineno}")
        return None

try:
    json_file_path = "MMRB_data_compare.json"
    if not os.path.exists(json_file_path):
        print(f"Error: Data file '{json_file_path}' not found.")
        sys.exit(1)
    json_data = load_json_data(json_file_path)
    print(f"Successfully loaded data from '{json_file_path}'.")
except FileNotFoundError:
    print(f"Error: '{json_file_path}' not found.")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"Error: '{json_file_path}' is not a valid JSON file.")
    sys.exit(1)

stats = {
    "total_samples": 0,
    "total_paths": 0,
    "modified_preferred_count": 0,
    "raw_preferred_count": 0,
    "tied_score_count": 0,
    "score_calculation_errors": 0,
    "data_preparation_errors": 0,
    "by_subtask": defaultdict(lambda: {"total": 0, "modified_preferred": 0, "raw_preferred": 0, 
                                       "tied": 0, "score_errors": 0, "data_errors": 0})
}

for sample_idx, sample in enumerate(json_data):
    stats["total_samples"] += 1
    
    if "modified_reasoning_steps" not in sample or "raw_reasoning_steps" not in sample:
        print(f"Warning: Sample {sample_idx + 1} (ID: {sample.get('id', 'N/A')}) is missing 'modified_reasoning_steps' or 'raw_reasoning_steps'. Skipping.")
        num_potential_paths = max(len(sample.get("modified_reasoning_steps", [])), len(sample.get("raw_reasoning_steps", [])))
        if num_potential_paths == 0 and sample.get('id'): num_potential_paths = 1
        stats["data_preparation_errors"] += num_potential_paths
        stats["by_subtask"][sample.get("subtask", "Unknown")]["data_errors"] += num_potential_paths
        continue

    available_path_indices = set(pd["path_index"] for pd in sample["modified_reasoning_steps"])
    
    image_paths_from_json = sample.get("image_paths", [])
    valid_image_paths = []
    if image_paths_from_json:
        for p_img in image_paths_from_json:
            if isinstance(p_img, str) and os.path.exists(p_img):
                valid_image_paths.append(p_img)
            else:
                print(f"Warning: Image path '{p_img}' (type: {type(p_img)}) is invalid or does not exist (Sample {sample_idx + 1}). It will be excluded.")
        if not valid_image_paths and image_paths_from_json:
             print(f"Warning: Sample {sample_idx + 1} provided image paths but none were valid. Processing without images.")
    
    original_question_text = sample.get('question', '')
    original_answer_text = sample.get('answer', '')
    if not original_question_text:
        print(f"Warning: Sample {sample_idx + 1} (ID: {sample.get('id', 'N/A')}) is missing 'question'. Skipping paths for this sample.")
        num_potential_paths = len(available_path_indices) if available_path_indices else 1
        stats["data_preparation_errors"] += num_potential_paths
        stats["by_subtask"][sample.get("subtask", "Unknown")]["data_errors"] += num_potential_paths
        continue
        
    question_for_model = original_question_text
    options = sample.get("options", [])
    if options:
        if isinstance(options, list):
            question_for_model += "\n" + "\n".join(options)
        else:
            print(f"  Warning: 'options' for sample {sample_idx + 1} is not in list format. Options will be ignored.")

    for path_idx, path_index in enumerate(sorted(list(available_path_indices))):
        stats["total_paths"] += 1
        subtask = sample.get("subtask", "Unknown")
        stats["by_subtask"][subtask]["total"] += 1
        
        current_path_has_error = False
        print(f"\nProcessing Sample {sample_idx + 1}/{len(json_data)}, Path Index: {path_index} (Subtask: {subtask})")
        if valid_image_paths:
            print(f"  Using {len(valid_image_paths)} valid image(s): {', '.join(valid_image_paths)}")
        else:
            print("  No valid images used.")

        modified_steps_text = f"Answer is: {original_answer_text}, because: " + get_reasoning_steps_text(sample["modified_reasoning_steps"], path_index, True)
        raw_steps_text = f"Answer is: {original_answer_text}, because: " + get_reasoning_steps_text(sample["raw_reasoning_steps"], path_index, False)

        if not modified_steps_text or not raw_steps_text:
            print(f"Warning: Sample {sample_idx + 1}, path_index {path_index} is missing modified or raw steps text. Skipping scoring for this path.")
            stats["data_preparation_errors"] += 1
            stats["by_subtask"][subtask]["data_errors"] += 1
            continue

        print(f"  Scoring Modified Response...")
        score_modified = get_reward_score(model, processor, valid_image_paths, question_for_model, modified_steps_text, "cuda:1")
        if score_modified is not None:
            print(f"  Modified Response Score: {score_modified:.4f}")
        else:
            print(f"  Error: Could not calculate score for Modified Response.")
            current_path_has_error = True
        
        clear_gpu_memory()

        print(f"  Scoring Raw Response...")
        score_raw = get_reward_score(model, processor, valid_image_paths, question_for_model, raw_steps_text, "cuda:1")
        if score_raw is not None:
            print(f"  Raw Response Score: {score_raw:.4f}")
        else:
            print(f"  Error: Could not calculate score for Raw Response.")
            current_path_has_error = True
        
        clear_gpu_memory()

        if current_path_has_error:
            stats["score_calculation_errors"] += 1
            stats["by_subtask"][subtask]["score_errors"] += 1
            print(f"  Comparison for this path skipped due to scoring error(s).")
        elif score_modified is not None and score_raw is not None:
            if score_modified > score_raw:
                stats["modified_preferred_count"] += 1
                stats["by_subtask"][subtask]["modified_preferred"] += 1
                print("  Model Evaluation: Modified Response is better")
            elif score_raw > score_modified:
                stats["raw_preferred_count"] += 1
                stats["by_subtask"][subtask]["raw_preferred"] += 1
                print("  Model Evaluation: Raw Response is better")
            else:
                stats["tied_score_count"] += 1
                stats["by_subtask"][subtask]["tied"] += 1
                print("  Model Evaluation: Modified and Raw Response scores are tied")
        else:
            stats["score_calculation_errors"] += 1
            stats["by_subtask"][subtask]["score_errors"] += 1
            print(f"  Comparison for this path skipped due to one or more invalid scores (Modified: {score_modified}, Raw: {score_raw}).")

print("\nFinal Statistics:")
print(f"Total Samples (attempted to process): {stats['total_samples']}")
print(f"Total Paths Evaluated (attempted to process): {stats['total_paths']}")
print(f"Modified Response Preferred Count: {stats['modified_preferred_count']}")
print(f"Raw Response Preferred Count: {stats['raw_preferred_count']}")
print(f"Tied Score Count: {stats['tied_score_count']}")
print(f"Path Score Calculation Error Count: {stats['score_calculation_errors']}")
print(f"Data Preparation Error (skipped paths) Count: {stats['data_preparation_errors']}")

valid_comparisons = stats['modified_preferred_count'] + stats['raw_preferred_count'] + stats['tied_score_count']
paths_attempted_scoring = stats['total_paths'] - stats['data_preparation_errors']

if valid_comparisons > 0:
    print(f"\nBased on {valid_comparisons} successfully scored and compared paths:")
    print(f"  Modified Response Preferred Percentage: {stats['modified_preferred_count']/valid_comparisons*100:.2f}%")
    print(f"  Raw Response Preferred Percentage: {stats['raw_preferred_count']/valid_comparisons*100:.2f}%")
    print(f"  Tied Score Percentage: {stats['tied_score_count']/valid_comparisons*100:.2f}%")

if paths_attempted_scoring > 0:
    print(f"  Score Calculation Error Rate (based on paths attempted for scoring): {stats['score_calculation_errors']/paths_attempted_scoring*100:.2f}%")
if stats['total_paths'] > 0:
     print(f"  Data Preparation Error Rate (based on total paths): {stats['data_preparation_errors']/stats['total_paths']*100:.2f}%")

print("\nStatistics by Subtask:")
for subtask, s_stats in stats["by_subtask"].items():
    print(f"\nSubtask: {subtask}")
    print(f"  Total Paths: {s_stats['total']}")
    print(f"  Modified Preferred: {s_stats['modified_preferred']}")
    print(f"  Raw Preferred: {s_stats['raw_preferred']}")
    print(f"  Tied Scores: {s_stats['tied']}")
    print(f"  Score Calculation Errors: {s_stats['score_errors']}")
    print(f"  Data Preparation Errors: {s_stats['data_errors']}")
    
    sub_valid_comparisons = s_stats['modified_preferred'] + s_stats['raw_preferred'] + s_stats['tied']
    sub_paths_attempted_scoring = s_stats['total'] - s_stats['data_errors']

    if sub_valid_comparisons > 0:
        print(f"    Modified Preferred Percentage (of valid comparisons): {s_stats['modified_preferred']/sub_valid_comparisons*100:.2f}%")
        print(f"    Raw Preferred Percentage (of valid comparisons): {s_stats['raw_preferred']/sub_valid_comparisons*100:.2f}%")
        print(f"    Tied Score Percentage (of valid comparisons): {s_stats['tied']/sub_valid_comparisons*100:.2f}%")
    if sub_paths_attempted_scoring > 0:
        print(f"    Score Calculation Error Rate (based on attempted scoring): {s_stats['score_errors']/sub_paths_attempted_scoring*100:.2f}%")
    if s_stats['total'] > 0:
        print(f"    Data Preparation Error Rate (based on total paths): {s_stats['data_errors']/s_stats['total']*100:.2f}%")

output_dir = "./R1_new_reward_model_output"
os.makedirs(output_dir, exist_ok=True)
csv_file_path = os.path.join(output_dir, "skywork_stats.csv")

try:
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        writer.writerow(["Overall Statistics", "Value"])
        writer.writerow(["Total Samples Processed", stats['total_samples']])
        writer.writerow(["Total Paths Attempted", stats['total_paths']])
        writer.writerow(["Times Modified Response Preferred", stats['modified_preferred_count']])
        writer.writerow(["Times Raw Response Preferred", stats['raw_preferred_count']])
        writer.writerow(["Times Scores Tied", stats['tied_score_count']])
        writer.writerow(["Times Score Calculation Error", stats['score_calculation_errors']])
        writer.writerow(["Times Data Preparation Error (skipped paths)", stats['data_preparation_errors']])
        
        if valid_comparisons > 0:
            writer.writerow(["Percentage Modified Preferred (of valid comparisons)", f"{stats['modified_preferred_count']/valid_comparisons*100:.2f}%"])
            writer.writerow(["Percentage Raw Preferred (of valid comparisons)", f"{stats['raw_preferred_count']/valid_comparisons*100:.2f}%"])
            writer.writerow(["Percentage Tied (of valid comparisons)", f"{stats['tied_score_count']/valid_comparisons*100:.2f}%"])
        if paths_attempted_scoring > 0:
            writer.writerow(["Percentage Score Calculation Error (of paths attempted scoring)", f"{stats['score_calculation_errors']/paths_attempted_scoring*100:.2f}%"])
        if stats['total_paths'] > 0:
            writer.writerow(["Percentage Data Preparation Error (of total paths)", f"{stats['data_preparation_errors']/stats['total_paths']*100:.2f}%"])
        writer.writerow([])

        writer.writerow(["Subtask", "Total Paths", "Modified Preferred", "Raw Preferred", "Tied",
                         "Score Errors", "Data Errors",
                         "Modified Pref % (valid comp)", "Raw Pref % (valid comp)", "Tied % (valid comp)",
                         "Score Error % (attempted score)", "Data Error % (total paths)"])
        
        for subtask, s_stats in stats["by_subtask"].items():
            sub_valid_comp = s_stats['modified_preferred'] + s_stats['raw_preferred'] + s_stats['tied']
            sub_attempted_score = s_stats['total'] - s_stats['data_errors']
            row = [
                subtask, s_stats['total'], s_stats['modified_preferred'], s_stats['raw_preferred'], s_stats['tied'],
                s_stats['score_errors'], s_stats['data_errors']
            ]
            row.append(f"{s_stats['modified_preferred']/sub_valid_comp*100:.2f}%" if sub_valid_comp > 0 else "N/A")
            row.append(f"{s_stats['raw_preferred']/sub_valid_comp*100:.2f}%" if sub_valid_comp > 0 else "N/A")
            row.append(f"{s_stats['tied']/sub_valid_comp*100:.2f}%" if sub_valid_comp > 0 else "N/A")
            row.append(f"{s_stats['score_errors']/sub_attempted_score*100:.2f}%" if sub_attempted_score > 0 else "N/A")
            row.append(f"{s_stats['data_errors']/s_stats['total']*100:.2f}%" if s_stats['total'] > 0 else "N/A")
            writer.writerow(row)
            
    print(f"\nStatistics saved to: {csv_file_path}")

except IOError:
    print(f"Error: Could not write CSV file to path {csv_file_path}. Please check permissions or path.")
except Exception as e:
    print(f"An unknown error occurred while saving CSV: {e}")

print("\nScript execution finished.")