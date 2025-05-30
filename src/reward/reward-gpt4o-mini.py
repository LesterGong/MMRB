import base64
import time
import json
import csv
import os
import sys
import copy
from collections import defaultdict
from PIL import Image
from openai import OpenAI
import concurrent.futures
from tqdm import tqdm
import itertools
import threading


API_KEY = os.getenv("OPENAI_API_KEY", "sk-...")  # OpenAI API Key

# 2. API KEY pool
API_KEYS = [
# "sk-...",
# "sk-..."
]

BASE_URL = "YOUR_OPENAI_API_BASE_URL"  
JSON_DATA_PATH = "MMRB_data_compare.json"
CSV_OUTPUT_PATH = "./gpt-4o-mini-2024-07-18/reward_model_stats_gpt-4o-mini-2024-07-18.csv"

MAX_CONCURRENT_REQUESTS = 100 # maximum number of concurrent API requests
MAX_RETRIES_PER_REQUEST = 3 # maximum number of retries for each API request
RETRY_DELAY_SECONDS = 1.5 # delay before retrying


if API_KEYS:
    api_key_cycler = itertools.cycle(API_KEYS)
    api_key_lock = threading.Lock()
    print(f"API Key pool has {len(API_KEYS)} keys.")
else:
    api_key_cycler = None
    api_key_lock = None
    print(f"API Key pool is empty, using single API Key: {API_KEY[:15]}...")


def get_next_api_key():
    if api_key_cycler and api_key_lock:
        with api_key_lock:
            return next(api_key_cycler)
    return API_KEY




def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"ERROR: Image file not found {image_path}")
        return None
    except Exception as e:
        print(f"ERROR: Failed to encode image {image_path}: {e}")
        return None

def load_json_data(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: JSON file not found {json_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"ERROR: Invalid JSON file format {json_path}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load JSON file {json_path}: {e}")
        sys.exit(1)

def get_reasoning_steps_text(steps_data_list, path_index_target, is_modified=False):

    for path_data in steps_data_list:
        if path_data.get("path_index") == path_index_target:
            steps_text_parts = []
            key_to_use = 'rationale_fix' if is_modified else 'rationale'

            path_steps = path_data.get("path", [])
            if not isinstance(path_steps, list):
                return "Reasoning path data is malformed (not a list)."

            for step in path_steps:
                if not isinstance(step, dict):
                    steps_text_parts.append(f"Step (malformed data): N/A")
                    continue
                steps_text_parts.append(f"Step {step.get('reasoning step', 'N/A')} ({step.get('reasoning type', 'N/A')}): {step.get(key_to_use, 'N/A')}")
            return "\n".join(steps_text_parts)
    return "Corresponding reasoning path not found."

def get_image_contents(image_paths):

    contents = []
    for img_path in image_paths:
        encoded_image = encode_image(img_path)
        if encoded_image:
            ext = os.path.splitext(img_path)[-1].lower()
            mime_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
            contents.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{encoded_image}",
                    "detail": "low"
                }
            })
    return contents

def process_path_concurrently(sample, path_index, image_contents, api_key_to_use, base_url_to_use, sample_idx_display):

    result = {
        "status": "error",
        "choice": None,
        "subtask": sample.get("subtask", "Unknown Subtask"),
        "error_message": None,
        "sample_id": sample.get('id', 'N/A'),
        "path_index": path_index
    }

    try:
        modified_steps = get_reasoning_steps_text(sample.get("modified_reasoning_steps", []), path_index, True)
        raw_steps = get_reasoning_steps_text(sample.get("raw_reasoning_steps", []), path_index, False)

        judge_prompt = f"""
You are a visual language model evaluation assistant. Please compare the two responses to the same question based on the following four aspects, determine which one is better, and explain why:

1. Accuracy of target description
2. Accuracy of relationship description
3. Accuracy of attribute description
4. Usefulness (informativeness/helpfulness)

Please strictly choose the better response. You must choose a better answer, you can't judge them as "equally good". Try you best to select a better one.
Output in the following format:
The better response: [1]. Because...
or
The better response: [2]. Because...


[Task Content]
Image and Question:
{sample.get('question', 'N/A')}

Answer:
{sample.get('answer', 'N/A')}

Response 1:
{modified_steps}

Response 2:
{raw_steps}
"""

        messages = [
            {"role": "user", "content": [{"type": "text", "text": judge_prompt}] + image_contents}
        ]

        client = OpenAI(base_url=base_url_to_use, api_key=api_key_to_use)
        output_text = ""
        found_choice = False

        for attempt in range(MAX_RETRIES_PER_REQUEST):
            try:

                response = client.chat.completions.create(
                    model="gpt-4o-mini-2024-07-18",
                    messages=messages,
                    temperature=0.01,
                    max_tokens=512
                )
                output_text = response.choices[0].message.content.strip()

                if "The better response: [1]." in output_text:

                    result["choice"] = "first"
                    found_choice = True
                elif "The better response: [2]." in output_text:

                    result["choice"] = "second"
                    found_choice = True
                elif "Two responses are equally good." in output_text:
                    result["choice"] = "equal"
                    found_choice = True

                if found_choice:
                    result["status"] = "success"
                    break
                else:
                    result["status"] = "no_match"
                    result["error_message"] = f"Could not find a distinguishing phrase. Output: {output_text[:150]}"

            except Exception as e:
                if attempt == MAX_RETRIES_PER_REQUEST - 1:
                    result["status"] = "error"
                    result["error_message"] = f"API request failed after multiple retries: {e}"
                time.sleep(RETRY_DELAY_SECONDS)

        if not found_choice and result["status"] != "error": # if it's "no_match" or initial "error" status from result dict
             result["status"] = "no_match" 
             result["error_message"] = result.get("error_message", f"Ultimately, no distinguishing phrase was found. Final output: {output_text[:150]}")


    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        error_details = f"Internal error during path processing: {e} (Type: {exc_type}, File: {fname}, Line: {exc_tb.tb_lineno})"
        result["status"] = "error"
        result["error_message"] = error_details

    return result

def main():
    json_data = load_json_data(JSON_DATA_PATH)

    stats = {
        "total_samples_processed": 0,
        "total_paths_submitted": 0,
        "successful_evaluations": 0,
        "first_choice": 0,
        "second_choice": 0,
        "equally_good": 0,
        "no_match_found": 0,
        "api_errors": 0,
        "by_subtask": defaultdict(lambda: {
            "total_paths": 0, "first": 0, "second": 0, "equal": 0, "no_match": 0, "errors": 0
        })
    }

    tasks_to_process = []
    sample_id_set = set()

    print(f"Preparing tasks...")
    for sample_idx, sample in enumerate(json_data):
        sample_id_set.add(sample.get('id', f"generated_id_{sample_idx}"))

        image_paths = sample.get("image_paths", [])
        if not isinstance(image_paths, list):
            print(f"WARNING: Sample {sample.get('id', sample_idx)} 'image_paths' is not a list. Skipping image processing for this sample.")
            image_contents = []
        else:
            image_contents = get_image_contents(image_paths)
            if not image_contents and image_paths: 
                print(f"WARNING: Not all images for sample {sample.get('id', sample_idx)} could be loaded/encoded successfully. Proceeding without images for this sample.")
                

        modified_steps_data = sample.get("modified_reasoning_steps", [])
        if not isinstance(modified_steps_data, list):
            print(f"WARNING: Sample {sample.get('id', sample_idx)} 'modified_reasoning_steps' is invalid. Skipping this sample.")
            continue

        available_path_indices = set()
        try:
            available_path_indices = set([p_data["path_index"] for p_data in modified_steps_data if isinstance(p_data, dict) and "path_index" in p_data])
        except TypeError: 
            print(f"WARNING: Sample {sample.get('id', sample_idx)} 'modified_reasoning_steps' structure is incorrect (e.g., list contains non-dict items). Skipping this sample.")
            continue

        for path_index in sorted(list(available_path_indices)):
            current_api_key = get_next_api_key() 
            tasks_to_process.append(
                (sample, path_index, image_contents, current_api_key, BASE_URL, sample_idx + 1)
            )
            stats["total_paths_submitted"] += 1
            stats["by_subtask"][sample.get("subtask", "Unknown Subtask")]["total_paths"] += 1

    stats["total_samples_processed"] = len(sample_id_set)
    print(f"Prepared a total of {stats['total_paths_submitted']} evaluation tasks from {stats['total_samples_processed']} unique samples.")
    print(f"Starting processing with up to {MAX_CONCURRENT_REQUESTS} concurrent threads...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
        future_to_task_args = {}
        for task_args in tasks_to_process:
            future = executor.submit(process_path_concurrently, *task_args)
            future_to_task_args[future] = task_args

        for future in tqdm(concurrent.futures.as_completed(future_to_task_args), total=len(tasks_to_process), desc="Evaluation Progress"):
            try:
                result = future.result()
                subtask = result["subtask"]

                if result["status"] == "success":
                    stats["successful_evaluations"] += 1
                    if result["choice"] == "first":
                        stats["first_choice"] += 1
                        stats["by_subtask"][subtask]["first"] += 1
                    elif result["choice"] == "second":
                        stats["second_choice"] += 1
                        stats["by_subtask"][subtask]["second"] += 1
                    elif result["choice"] == "equal":
                        stats["equally_good"] += 1
                        stats["by_subtask"][subtask]["equal"] += 1
                elif result["status"] == "no_match":
                    stats["no_match_found"] += 1
                    stats["by_subtask"][subtask]["no_match"] += 1
                elif result["status"] == "error":
                    stats["api_errors"] += 1
                    stats["by_subtask"][subtask]["errors"] += 1
                    print(f"ERROR: Sample {result['sample_id']} path {result['path_index']} failed to process. Reason: {result.get('error_message', 'Unknown error')}")

            except Exception as e:
                stats["api_errors"] += 1
                task_args_for_error = future_to_task_args.get(future)
                error_subtask = "Unknown Subtask (error in future)"
                error_sample_id = "N/A"
                error_path_idx = "N/A"
                if task_args_for_error:
                    error_subtask = task_args_for_error[0].get("subtask", "Unknown Subtask")
                    error_sample_id = task_args_for_error[0].get("id", "N/A")
                    error_path_idx = task_args_for_error[1]

                stats["by_subtask"][error_subtask]["errors"] += 1
                print(f"CRITICAL ERROR: An unexpected issue occurred while retrieving result for task (Sample {error_sample_id}, Path {error_path_idx}): {e}")


    print("\n--- Final Statistics ---")
    print(f"Total Unique Samples Processed: {stats['total_samples_processed']}")
    print(f"Total Paths Submitted for Evaluation: {stats['total_paths_submitted']}")
    print(f"Successful Evaluations (choice made): {stats['successful_evaluations']}")
    print(f"Times Response 1 Chosen: {stats['first_choice']}")
    print(f"Times Response 2 Chosen: {stats['second_choice']}")
    print(f"Times Responses Equally Good: {stats['equally_good']}")
    print(f"Times No Match Found (API OK, no choice phrase): {stats['no_match_found']}")
    print(f"Times API/Processing Error: {stats['api_errors']}")

    output_dir = os.path.dirname(CSV_OUTPUT_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    try:
        with open(CSV_OUTPUT_PATH, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Overall Statistics", "Value"])
            writer.writerow(["Total Unique Samples Processed", stats['total_samples_processed']])
            writer.writerow(["Total Paths Submitted for Evaluation", stats['total_paths_submitted']])
            writer.writerow(["Successful Evaluations (choice made)", stats['successful_evaluations']])
            writer.writerow(["Times Response 1 Chosen", stats['first_choice']])
            writer.writerow(["Times Response 2 Chosen", stats['second_choice']])
            writer.writerow(["Times Responses Equally Good", stats['equally_good']])
            writer.writerow(["Times No Match Found (API ok, no choice phrase)", stats['no_match_found']])
            writer.writerow(["Times API/Processing Error", stats['api_errors']])

            successful_evals = stats['successful_evaluations']
            writer.writerow(["Percentage Response 1 Chosen (of successful)", f"{stats['first_choice']/successful_evals*100:.2f}%" if successful_evals > 0 else "N/A"])
            writer.writerow(["Percentage Response 2 Chosen (of successful)", f"{stats['second_choice']/successful_evals*100:.2f}%" if successful_evals > 0 else "N/A"])
            writer.writerow(["Percentage Equally Good (of successful)", f"{stats['equally_good']/successful_evals*100:.2f}%" if successful_evals > 0 else "N/A"])

            total_submitted = stats['total_paths_submitted']
            writer.writerow(["Percentage No Match Found (of total submitted)", f"{stats['no_match_found']/total_submitted*100:.2f}%" if total_submitted > 0 else "N/A"])
            writer.writerow(["Percentage API Errors (of total submitted)", f"{stats['api_errors']/total_submitted*100:.2f}%" if total_submitted > 0 else "N/A"])

            writer.writerow([])
            writer.writerow(["Subtask", "Total Paths Submitted", "Response 1 Chosen", "Response 2 Chosen", "Equally Good",
                             "No Match Found", "Errors",
                             "Response 1 % (of successful in subtask)", "Response 2 % (of successful in subtask)",
                             "Equally Good % (of successful in subtask)",
                             "No Match % (of total in subtask)", "Error % (of total in subtask)"])

            for subtask, s_stats in stats["by_subtask"].items():
                successful_in_subtask = s_stats['first'] + s_stats['second'] + s_stats['equal']
                total_in_subtask = s_stats['total_paths']
                row = [
                    subtask,
                    total_in_subtask,
                    s_stats['first'],
                    s_stats['second'],
                    s_stats['equal'],
                    s_stats['no_match'],
                    s_stats['errors']
                ]
                row.extend([
                    f"{s_stats['first']/successful_in_subtask*100:.2f}%" if successful_in_subtask > 0 else "N/A",
                    f"{s_stats['second']/successful_in_subtask*100:.2f}%" if successful_in_subtask > 0 else "N/A",
                    f"{s_stats['equal']/successful_in_subtask*100:.2f}%" if successful_in_subtask > 0 else "N/A",
                    f"{s_stats['no_match']/total_in_subtask*100:.2f}%" if total_in_subtask > 0 else "N/A",
                    f"{s_stats['errors']/total_in_subtask*100:.2f}%" if total_in_subtask > 0 else "N/A"
                ])
                writer.writerow(row)
        print(f"\nStatistics saved to: {CSV_OUTPUT_PATH}")
    except Exception as e:
        print(f"Error saving CSV: {e}")

if __name__ == "__main__":
    main()