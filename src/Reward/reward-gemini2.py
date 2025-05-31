import base64
import time
import json
import csv
import os
import sys
from collections import defaultdict
from openai import OpenAI 
import concurrent.futures
from tqdm import tqdm 
import itertools 
import threading 
import mimetypes 



DEFAULT_API_KEY = 'YOUR_DEFAULT_API_KEY_IF_NO_POOL' 

API_KEYS = [
   #"sk-..."
]

PROXY_BASE_URL = "Your Base URL"
MODEL_NAME_FOR_PROXY = "gemini-2.0-flash-lite-preview-02-05"


JSON_DATA_PATH = "MMRB_data_reward.json"
CSV_OUTPUT_PATH = f"./{MODEL_NAME_FOR_PROXY}_via_proxy/reward_model_stats_concurrent_en.csv"

MAX_CONCURRENT_REQUESTS = 100
MAX_RETRIES_PER_REQUEST = 3
RETRY_DELAY_SECONDS = 1.5
API_REQUEST_TIMEOUT_SECONDS = 180
MAX_IMAGES_ALLOWED = 16 # Maximum images the model accepts



if API_KEYS:
    api_key_cycler = itertools.cycle(API_KEYS)
    api_key_lock = threading.Lock()
else:
    api_key_cycler = None
    api_key_lock = None

def get_next_api_key():
    if api_key_cycler and api_key_lock:
        with api_key_lock:
            return next(api_key_cycler)
    return DEFAULT_API_KEY

def encode_image_to_base64(image_path):
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

def get_openai_image_contents(image_paths):
    contents = []
    for img_path in image_paths:
        encoded_image = encode_image_to_base64(img_path)
        if encoded_image:
            mime_type, _ = mimetypes.guess_type(img_path)
            if not mime_type:
                if img_path.lower().endswith((".jpg", ".jpeg")):
                    mime_type = "image/jpeg"
                elif img_path.lower().endswith(".png"):
                    mime_type = "image/png"
                elif img_path.lower().endswith(".webp"):
                    mime_type = "image/webp"
                elif img_path.lower().endswith(".gif"):
                    mime_type = "image/gif"
                else:
                    print(f"WARNING: Could not determine MIME type for {img_path}, defaulting to application/octet-stream.")
                    mime_type = "application/octet-stream"
            contents.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{encoded_image}",
                    "detail": "low"
                }
            })
    return contents

def process_path_concurrently_openai_proxy(sample, path_index, openai_image_contents, api_key_to_use, sample_idx_display, sampling_notification_text):
    """
    The `sampling_notification_text` is added to the prompt if images were sampled.
    """
    result = {
        "status": "error",
        "choice": None,
        "subtask": sample.get("subtask", "Unknown Subtask"),
        "error_message": None,
        "sample_id": sample.get('id', 'N/A'),
        "path_index": path_index,
        "raw_api_response_text": ""
    }

    try:
        client = OpenAI(
            base_url=PROXY_BASE_URL,
            api_key=api_key_to_use,
            timeout=API_REQUEST_TIMEOUT_SECONDS
        )

        modified_steps = get_reasoning_steps_text(sample.get("modified_reasoning_steps", []), path_index, True)
        raw_steps = get_reasoning_steps_text(sample.get("raw_reasoning_steps", []), path_index, False)

        prompt_prefix = ""
        if sampling_notification_text: 
            prompt_prefix = sampling_notification_text 

        judge_prompt = f"""{prompt_prefix}
You are a visual language model evaluation assistant. Please compare the two responses to the same question based on the following four aspects, determine which one is better, and explain why:

1. Accuracy of target description
2. Accuracy of relationship description
3. Accuracy of attribute description
4. Usefulness (informativeness/helpfulness)

Please strictly choose the better response. 
You must choose a better answer, you can't judge them as "equally good". Try you best to select a better one.
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
        message_content_parts = [{"type": "text", "text": judge_prompt}]
        if openai_image_contents:
            message_content_parts.extend(openai_image_contents)

        messages = [{"role": "user", "content": message_content_parts}]
        output_text = ""
        found_choice = False

        for attempt in range(MAX_RETRIES_PER_REQUEST):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME_FOR_PROXY,
                    messages=messages,
                    temperature=0.01,
                    max_tokens=512
                )
                output_text = response.choices[0].message.content.strip() if response.choices and response.choices[0].message else ""
                result["raw_api_response_text"] = output_text
                if "The better response: [1]." in output_text:
                    result["choice"] = "first"
                    found_choice = True
                elif "The better response: [2]." in output_text:
                    result["choice"] = "second"
                    found_choice = True
                elif "Two responses are equally good." in output_text or "equally good" in output_text.lower():
                    result["choice"] = "equal"
                    found_choice = True
                
                if found_choice:
                    result["status"] = "success"
                    break
            except Exception as e:
                print(f"Thread {threading.get_ident()}: API request via proxy failed (Sample {sample_idx_display}, Path {path_index}, Attempt {attempt+1}): {type(e).__name__} - {e}")
                result["raw_api_response_text"] = f"Error: {type(e).__name__} - {e}"
                if attempt == MAX_RETRIES_PER_REQUEST - 1:
                    result["status"] = "error"
                    result["error_message"] = f"API request via proxy failed after {MAX_RETRIES_PER_REQUEST} retries: {e}"
                time.sleep(RETRY_DELAY_SECONDS * (attempt + 1))
        
        if not found_choice and result["status"] != "error":
            result["status"] = "no_match"
            result["error_message"] = f"Could not find a distinguishing phrase. Output: {output_text[:250]}"

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        error_details = f"Internal error during path processing: {e} (Type: {exc_type}, File: {fname}, Line: {exc_tb.tb_lineno})"
        result["status"] = "error"
        result["error_message"] = error_details
        result["raw_api_response_text"] = error_details
        
    return result

def main():
    json_data = load_json_data(JSON_DATA_PATH)

    stats = {
        "total_samples_processed": 0, "total_paths_submitted": 0, "successful_evaluations": 0,
        "first_choice": 0, "second_choice": 0, "equally_good": 0, "no_match_found": 0, "api_errors": 0,
        "by_subtask": defaultdict(lambda: {
            "total_paths": 0, "first": 0, "second": 0, "equal": 0, "no_match": 0, "errors": 0
        })
    }
    tasks_to_process = []
    sample_id_set = set()

    print(f"Preparing tasks for model '{MODEL_NAME_FOR_PROXY}' via proxy '{PROXY_BASE_URL}'...")
    for sample_idx, sample in enumerate(json_data):
        sample_id = sample.get('id', f"generated_id_{sample_idx}")
        sample_id_set.add(sample_id)
        
        image_paths_original = sample.get("image_paths", [])
        images_for_api = []
        sampling_notification_text = ""

        if not isinstance(image_paths_original, list):
            print(f"WARNING: Sample {sample_id} 'image_paths' is not a list. Skipping image processing for this sample.")
        else:

            original_num_images = len(image_paths_original)
            if original_num_images > MAX_IMAGES_ALLOWED:
                print(f"INFO: Sample {sample_id} has {original_num_images} images. Sampling down to {MAX_IMAGES_ALLOWED}.")
                if MAX_IMAGES_ALLOWED == 1: 
                    sampled_indices = [0] if original_num_images > 0 else []
                elif original_num_images > 0 : 
                    sampled_indices = sorted(list(set(
                        [int(round(i * (original_num_images - 1) / (MAX_IMAGES_ALLOWED - 1))) for i in range(MAX_IMAGES_ALLOWED)]
                    )))
                else:
                    sampled_indices = []

                images_for_api = [image_paths_original[i] for i in sampled_indices if i < original_num_images] 
                

                sampling_notification_text = f"there's {original_num_images} images, The sampling has been done equally to {len(images_for_api)}.\n"
                if len(images_for_api) != MAX_IMAGES_ALLOWED and original_num_images > 0:
                     print(f"WARNING: Sample {sample_id} after sampling resulted in {len(images_for_api)} images, expected {MAX_IMAGES_ALLOWED}. Original: {original_num_images}")


            else: 
                images_for_api = image_paths_original

            openai_image_contents = get_openai_image_contents(images_for_api)
            if not openai_image_contents and images_for_api:
                print(f"WARNING: Not all images for sample {sample_id} could be processed into OpenAI format parts. Number of parts: {len(openai_image_contents)}")
        
        modified_steps_data = sample.get("modified_reasoning_steps", [])
        if not isinstance(modified_steps_data, list):
            print(f"WARNING: Sample {sample_id} 'modified_reasoning_steps' is invalid. Skipping this sample.")
            continue

        available_path_indices = set()
        try:
            available_path_indices = set([p_data["path_index"] for p_data in modified_steps_data if isinstance(p_data, dict) and "path_index" in p_data])
        except TypeError:
            print(f"WARNING: Sample {sample_id} 'modified_reasoning_steps' structure is incorrect. Skipping this sample.")
            continue

        if not available_path_indices:
            pass 

        for path_index in sorted(list(available_path_indices)):
            current_api_key = get_next_api_key()
            tasks_to_process.append(
                (sample, path_index, openai_image_contents, current_api_key, sample_idx + 1, sampling_notification_text)
            )
            stats["total_paths_submitted"] += 1
            stats["by_subtask"][sample.get("subtask", "Unknown Subtask")]["total_paths"] += 1
    
    stats["total_samples_processed"] = len(sample_id_set)
    print(f"Prepared a total of {stats['total_paths_submitted']} evaluation tasks from {stats['total_samples_processed']} unique samples.")
    print(f"Starting processing with up to {MAX_CONCURRENT_REQUESTS} concurrent threads...")

    output_dir = os.path.dirname(CSV_OUTPUT_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    all_results_for_csv = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
        future_to_task_args = {}
        for task_args in tasks_to_process:
            future = executor.submit(process_path_concurrently_openai_proxy, *task_args)
            future_to_task_args[future] = task_args

        for future in tqdm(concurrent.futures.as_completed(future_to_task_args), total=len(tasks_to_process), desc="Evaluation Progress"):
            task_sample_id = "N/A"
            task_path_idx = "N/A"
            task_subtask = "Unknown Subtask"
            try:
                original_task_args = future_to_task_args[future]
                task_sample_id = original_task_args[0].get('id', 'N/A')
                task_path_idx = original_task_args[1]
                task_subtask = original_task_args[0].get("subtask", "Unknown Subtask")

                result = future.result()
                all_results_for_csv.append(result)
                subtask = result["subtask"]

                if result["status"] == "success":
                    stats["successful_evaluations"] += 1
                    if result["choice"] == "first":
                        stats["first_choice"] += 1; stats["by_subtask"][subtask]["first"] += 1
                    elif result["choice"] == "second":
                        stats["second_choice"] += 1; stats["by_subtask"][subtask]["second"] += 1
                    elif result["choice"] == "equal":
                        stats["equally_good"] += 1; stats["by_subtask"][subtask]["equal"] += 1
                elif result["status"] == "no_match":
                    stats["no_match_found"] += 1; stats["by_subtask"][subtask]["no_match"] += 1
                elif result["status"] == "error":
                    stats["api_errors"] += 1; stats["by_subtask"][subtask]["errors"] += 1
                    print(f"ERROR: Sample {result['sample_id']} path {result['path_index']} failed via proxy. Reason: {result.get('error_message', 'Unknown error')}")
            except Exception as e:
                stats["api_errors"] += 1
                stats["by_subtask"][task_subtask]["errors"] += 1
                print(f"CRITICAL ERROR: Task for Sample {task_sample_id}, Path {task_path_idx} (via proxy) failed: {e}")
                all_results_for_csv.append({
                    "status": "critical_error", "choice": None, "subtask": task_subtask,
                    "error_message": str(e), "sample_id": task_sample_id, "path_index": task_path_idx,
                    "raw_api_response_text": f"Critical error: {e}"
                })

    print(f"\n--- Final Statistics (Model: {MODEL_NAME_FOR_PROXY} via Proxy: {PROXY_BASE_URL}) ---")
    print(f"Total Unique Samples Processed: {stats['total_samples_processed']}")
    print(f"Total Paths Submitted for Evaluation: {stats['total_paths_submitted']}")
    print(f"Successful Evaluations (choice made): {stats['successful_evaluations']}")
    print(f"  Times Response 1 Chosen: {stats['first_choice']}")
    print(f"  Times Response 2 Chosen: {stats['second_choice']}")
    print(f"  Times Responses Equally Good: {stats['equally_good']}")
    print(f"Times No Match Found (API OK, no choice phrase): {stats['no_match_found']}")
    print(f"Times API/Processing Error: {stats['api_errors']}")

    try:
        with open(CSV_OUTPUT_PATH, 'w', newline='', encoding='utf-8') as csvfile:
            if all_results_for_csv:
                detailed_fieldnames = ['sample_id', 'path_index', 'subtask', 'status', 'choice', 'error_message', 'raw_api_response_text']
                detailed_writer = csv.DictWriter(csvfile, fieldnames=detailed_fieldnames, extrasaction='ignore')
                detailed_writer.writeheader()
                for res in all_results_for_csv:
                    detailed_writer.writerow(res)
                csvfile.write("\n\n")

            summary_writer = csv.writer(csvfile)
            summary_writer.writerow([f"Overall Statistics (Model: {MODEL_NAME_FOR_PROXY} via Proxy)", "Value"])
            summary_writer.writerow(["Proxy Base URL", PROXY_BASE_URL])
            summary_writer.writerow(["Total Unique Samples Processed", stats['total_samples_processed']])
            summary_writer.writerow(["Total Paths Submitted for Evaluation", stats['total_paths_submitted']])
            summary_writer.writerow(["Successful Evaluations (choice made)", stats['successful_evaluations']])
            summary_writer.writerow(["Times Response 1 Chosen", stats['first_choice']])
            summary_writer.writerow(["Times Response 2 Chosen", stats['second_choice']])
            summary_writer.writerow(["Times Responses Equally Good", stats['equally_good']])
            summary_writer.writerow(["Times No Match Found (API ok, no choice phrase)", stats['no_match_found']])
            summary_writer.writerow(["Times API/Processing Error", stats['api_errors']])
            
            successful_evals = stats['successful_evaluations']
            summary_writer.writerow(["Percentage Response 1 Chosen (of successful)", f"{stats['first_choice']/successful_evals*100:.2f}%" if successful_evals > 0 else "N/A"])
            summary_writer.writerow(["Percentage Response 2 Chosen (of successful)", f"{stats['second_choice']/successful_evals*100:.2f}%" if successful_evals > 0 else "N/A"])
            summary_writer.writerow(["Percentage Equally Good (of successful)", f"{stats['equally_good']/successful_evals*100:.2f}%" if successful_evals > 0 else "N/A"])
            
            total_submitted = stats['total_paths_submitted']
            summary_writer.writerow(["Percentage No Match Found (of total submitted)", f"{stats['no_match_found']/total_submitted*100:.2f}%" if total_submitted > 0 else "N/A"])
            summary_writer.writerow(["Percentage API Errors (of total submitted)", f"{stats['api_errors']/total_submitted*100:.2f}%" if total_submitted > 0 else "N/A"])

            summary_writer.writerow([])
            summary_writer.writerow(["Subtask", "Total Paths Submitted", "Response 1 Chosen", "Response 2 Chosen", "Equally Good", 
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
                    s_stats['errors'],
                ]
                row.extend([
                    f"{s_stats['first']/successful_in_subtask*100:.2f}%" if successful_in_subtask > 0 else "N/A",
                    f"{s_stats['second']/successful_in_subtask*100:.2f}%" if successful_in_subtask > 0 else "N/A",
                    f"{s_stats['equal']/successful_in_subtask*100:.2f}%" if successful_in_subtask > 0 else "N/A",
                    f"{s_stats['no_match']/total_in_subtask*100:.2f}%" if total_in_subtask > 0 else "N/A",
                    f"{s_stats['errors']/total_in_subtask*100:.2f}%" if total_in_subtask > 0 else "N/A",
                ])
                summary_writer.writerow(row)

        print(f"\nStatistics and detailed results saved to: {CSV_OUTPUT_PATH}")
    except Exception as e:
        print(f"Error saving CSV: {e}")

if __name__ == "__main__":
    main()