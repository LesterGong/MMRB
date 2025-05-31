import os
import json
import base64
import time
import argparse
import asyncio
from typing import List, Dict, Any, Tuple, Optional

from tqdm.asyncio import tqdm

from openai import AsyncOpenAI, OpenAIError

API_BASE_URL = os.getenv("OPENAI_BASE_URL", "your base url")
API_KEYS = [
    "your api key 1",
    "your api key 2"
]

MODEL = "grok-3-think"
MAX_RETRIES = 2
RETRY_DELAY_S = 2
CONCURRENT_LIMIT = 100
API_TIMEOUT_S = 600
REFUSAL_ANSWER = "I'm sorry, but I can't help with that."


def get_composite_key(item: Dict[str, Any]) -> Optional[Tuple[str, str, Any]]:
    source = item.get("source")
    subtask = item.get("subtask")
    index = item.get("index") 

    if source is None or subtask is None or index is None:
        return None
    
    source_str = str(source).strip()
    subtask_str = str(subtask).strip()
    
    if not source_str or not subtask_str:
        return None

    return (source_str, subtask_str, index)


class ApiKeyManager:
    def __init__(self, keys: List[str], base_url: str):
        if not keys or not any(keys):
            raise ValueError("API key list cannot be empty or contain only empty strings.")
        self._keys_list = [key for key in keys if key]
        if not self._keys_list:
            raise ValueError("API key list contained only empty strings after filtering.")
        self._base_url = base_url
        self._current_key_idx = 0
        self._lock = asyncio.Lock()

    async def get_api_credentials(self) -> Tuple[str, str]:
        async with self._lock:
            key = self._keys_list[self._current_key_idx]
            self._current_key_idx = (self._current_key_idx + 1) % len(self._keys_list)
            return key, self._base_url


def encode_image(p: str) -> str:
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode()

def image_msgs(paths: List[str]) -> List[Dict[str, Any]]:
    if not paths:
        return []
    valid_paths = []
    for p in paths:
        if isinstance(p, str) and p.strip() and os.path.exists(p):
            valid_paths.append(p)

    return [
        {"type": "image_url",
         "image_url": {"url": f"data:image/{os.path.splitext(p)[1][1:].lower()};base64,{encode_image(p)}"},
         "detail":"low"}
        for p in valid_paths
    ]

def build_prompt(sample: Dict[str, Any]) -> str:
    if sample["question_type"] == "multi-choice":
        return (
            f"{sample['question']}\n\n"
            "Only output the option letter, without any explanation.\n"
            "Please write your answer in the format: Answer[<letter>]. "
        )
    else:  
        return (
            f"{sample['question']}\n\n"
            "Only output the final answer, without any explanation.\n"
            "Please write your answer in the format: Answer[<your_answer_here>]."
        )

async def async_process_sample(
    sample_info: Dict[str, Any], 
    api_key_manager: ApiKeyManager,
    semaphore: asyncio.Semaphore,
    model: str,
    max_retries: int,
    retry_delay_s: int,
    api_timeout_s: int
) -> Dict[str, Any]:
    composite_key = sample_info['composite_key'] 
    sample_data = sample_info['sample']

    prompt_text = build_prompt(sample_data)
    image_messages = image_msgs(sample_data.get("image_paths", []))
    current_api_key_for_error_msg = "N/A"

    async with semaphore:
        for attempt in range(max_retries):
            client: Optional[AsyncOpenAI] = None
            try:
                current_api_key, current_base_url = await api_key_manager.get_api_credentials()
                current_api_key_for_error_msg = f"...{current_api_key[-4:]}" if current_api_key else "N/A"
                
                client = AsyncOpenAI(api_key=current_api_key, base_url=current_base_url, timeout=api_timeout_s)
                
                messages_payload = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": [{"type": "text", "text": prompt_text}, *image_messages]},
                ]
                
                resp = await client.chat.completions.create(model=model, temperature=0.0, messages=messages_payload)
                cot_answer = resp.choices[0].message.content

                result_output = sample_data.copy() 
                result_output["CoT_answer"] = cot_answer
                result_output["model_used"] = model
                result_output["api_key_hint"] = current_api_key_for_error_msg
                result_output["status"] = "success"
                return result_output

            except OpenAIError as e:
                error_message = f"OpenAIError for {composite_key} (key hint {current_api_key_for_error_msg}) on attempt {attempt + 1}/{max_retries}: {type(e).__name__} - {e}"
                if attempt + 1 >= max_retries:
                    print(f"[{composite_key}] All retries failed (OpenAIError). Error: {e}")
                    failed_result = sample_data.copy()
                    failed_result.update({"error": error_message, "CoT_answer": "ERROR_MAX_RETRIES_OPENAI", "status": "failed", "model_used": model})
                    return failed_result
                else: 
                    print(f"INFO: [{composite_key}] OpenAIError on attempt {attempt + 1}/{max_retries}. Retrying in {retry_delay_s * (attempt + 1)}s... Error: {type(e).__name__}")
                await asyncio.sleep(retry_delay_s * (attempt + 1))
            except Exception as e:
                error_message = f"Generic error for {composite_key} (key hint {current_api_key_for_error_msg}) on attempt {attempt + 1}/{max_retries}: {type(e).__name__} - {e}"
                if "image_url" in str(e).lower() and ("invalid" in str(e).lower() or "could not be accessed" in str(e).lower()):
                     print(f"ERROR: [{composite_key}] Critical image error: {e}. Marking as failed without further retries.")
                     failed_result = sample_data.copy()
                     failed_result.update({"error": f"Critical image error: {e}", "CoT_answer": "ERROR_INVALID_IMAGE_DATA", "status": "failed", "model_used": model})
                     return failed_result
                if attempt + 1 >= max_retries:
                    print(f"ERROR: [{composite_key}] All retries failed (Generic Error). Error: {e}")
                    failed_result = sample_data.copy()
                    failed_result.update({"error": error_message, "CoT_answer": "ERROR_MAX_RETRIES_GENERIC", "status": "failed", "model_used": model})
                    return failed_result
                else: 
                    print(f"INFO: [{composite_key}] Generic error on attempt {attempt + 1}/{max_retries}. Retrying in {retry_delay_s * (attempt + 1)}s... Error: {type(e).__name__}")
                await asyncio.sleep(retry_delay_s * (attempt + 1))
            finally:
                if client: 
                    try:
                        await client.close()
                    except Exception as close_err:
                        print(f"DEBUG: Error closing OpenAI client for {composite_key}: {close_err}") 
        
        print(f"ERROR: [{composite_key}] Unexpectedly exhausted retries loop without returning.") 
        fallback_result = sample_data.copy()
        fallback_result.update({"error": "Exhausted retries unexpectedly", "CoT_answer": "ERROR_UNEXPECTED_RETRY_EXHAUSTION", "status": "failed", "model_used": model})
        return fallback_result

async def process_concurrently(input_json_path: str, output_json_path: str) -> None:
    try:
        with open(input_json_path, "r", encoding="utf-8") as f:
            all_rows_raw = json.load(f)
        if not isinstance(all_rows_raw, list):
            print(f"Error: Input file {input_json_path} top-level structure is not a JSON list.")
            return
    except Exception as e:
        print(f"Error reading or parsing input file {input_json_path}: {e}")
        return

    input_samples_map: Dict[Tuple[str, str, Any], Dict[str, Any]] = {}
    malformed_input_count = 0
    duplicate_input_keys = 0
    
    for idx, sample_data_item in enumerate(all_rows_raw):
        if not isinstance(sample_data_item, dict):
            malformed_input_count += 1
            continue
        
        comp_key = get_composite_key(sample_data_item)
        if comp_key is None:
            malformed_input_count += 1
            continue
        
        if comp_key in input_samples_map:
            duplicate_input_keys +=1
            continue 
        input_samples_map[comp_key] = sample_data_item
    
    if malformed_input_count > 0:
        print(f"Note: {malformed_input_count} records in input file were skipped due to format issues or inability to generate composite key.")
    if duplicate_input_keys > 0:
        print(f"Note: {duplicate_input_keys} duplicate composite keys in input file were ignored (keeping first occurrence).")
    if not input_samples_map:
        print("Error: No valid input samples after processing. Please check input file format and composite key fields.")
        return

    start_index = 0 
    print(f"DEBUG: Using start_index = {start_index}")

    considered_input_slice_keys: List[Tuple[str, str, Any]] = []
    temp_key_set_for_slice = set() 
    for item in all_rows_raw[start_index:]:
        if isinstance(item, dict):
            key = get_composite_key(item)
            if key and key not in temp_key_set_for_slice : 
                considered_input_slice_keys.append(key)
                temp_key_set_for_slice.add(key)
    print(f"DEBUG: Number of unique composite keys considered from input file (start_index={start_index}): {len(considered_input_slice_keys)}")


    output_dir = os.path.dirname(output_json_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    existing_results_map: Dict[Tuple[str, str, Any], Dict[str, Any]] = {}
    samples_to_reprocess_keys = set() 

    if os.path.exists(output_json_path) and os.path.getsize(output_json_path) > 0: 
        try:
            with open(output_json_path, "r", encoding="utf-8") as f_read:
                content = f_read.read()
                if content.strip():
                    previous_results_list = json.loads(content)
                    if isinstance(previous_results_list, list):
                        print(f"Loaded {len(previous_results_list)} previous records from {output_json_path}.")
                        malformed_output_records = 0
                        for prev_result in previous_results_list:
                            if not isinstance(prev_result, dict):
                                malformed_output_records +=1; continue
                            
                            prev_comp_key = get_composite_key(prev_result)
                            if prev_comp_key is None:
                                malformed_output_records +=1; continue
                            
                            status_val = prev_result.get("status")
                            cot_val = prev_result.get("CoT_answer", "")
                            
                            is_explicit_error = str(status_val).strip().lower() == "failed" or \
                                                (isinstance(cot_val, str) and cot_val.startswith("ERROR_"))
                            is_refusal = isinstance(cot_val, str) and cot_val.strip() == REFUSAL_ANSWER
                            
                            should_reprocess = is_explicit_error or is_refusal
                            
                            if prev_comp_key in temp_key_set_for_slice: 
                                if should_reprocess:
                                    samples_to_reprocess_keys.add(prev_comp_key)
                                else: 
                                    existing_results_map[prev_comp_key] = prev_result
                        if malformed_output_records > 0:
                             print(f"DEBUG: Skipped {malformed_output_records} malformed records in {output_json_path}.")
        except json.JSONDecodeError as e:
            print(f"Warning: JSON error while parsing existing {output_json_path}: {e}. Will treat as empty file.")
        except Exception as e: 
            print(f"Warning: Error reading or parsing existing {output_json_path}: {e}. Will treat as empty file.")

    print(f"DEBUG: (After population) existing_results_map (successful records) length: {len(existing_results_map)}")
    if existing_results_map: print(f"DEBUG: Some keys from existing_results_map (max 2): {list(existing_results_map.keys())[:2]}")
    print(f"DEBUG: (After population) samples_to_reprocess_keys (failed/refused records to retry) length: {len(samples_to_reprocess_keys)}")
    if samples_to_reprocess_keys: print(f"DEBUG: Some keys from samples_to_reprocess_keys (max 2): {list(samples_to_reprocess_keys)[:2]}")

    samples_for_processing_tasks = []
    newly_added_count = 0
    reprocessed_added_count = 0
    skipped_as_successful_count = 0

    for current_key in considered_input_slice_keys: 
        original_sample_data = input_samples_map.get(current_key) 
        if not original_sample_data:
            print(f"Warning: Composite key {current_key} in input slice but not found in deduplicated original input map. Skipping.")
            continue

        if current_key in samples_to_reprocess_keys:
            samples_for_processing_tasks.append({'composite_key': current_key, 'sample': original_sample_data})
            reprocessed_added_count += 1
        elif current_key not in existing_results_map: 
            samples_for_processing_tasks.append({'composite_key': current_key, 'sample': original_sample_data})
            newly_added_count +=1
        else: 
            skipped_as_successful_count +=1

    print(f"DEBUG: Number added to task list from samples_to_reprocess_keys: {reprocessed_added_count}")
    print(f"DEBUG: Number added to task list as new samples/not found in existing_map: {newly_added_count}")
    print(f"DEBUG: Number skipped as already in existing_results_map and considered successful: {skipped_as_successful_count}")

    if not samples_for_processing_tasks:
        print("No samples need reprocessing or new processing.")
        if existing_results_map: 
            try:
        
                with open(output_json_path, "w", encoding="utf-8") as outfile:
                    json.dump(list(existing_results_map.values()), outfile, ensure_ascii=False, indent=4)
                print(f"Written/updated {len(existing_results_map)} existing successful records to {output_json_path} (no new tasks to process)")
            except Exception as e:
                print(f"Error writing existing successful records to {output_json_path}: {e}")
        elif not os.path.exists(output_json_path) or os.path.getsize(output_json_path) == 0 :
            try:
                with open(output_json_path, "w", encoding="utf-8") as outfile:
                    json.dump([], outfile, ensure_ascii=False, indent=4)
                print(f"Output file {output_json_path} is empty or does not exist, initialized as empty list.")
            except Exception as e:
                print(f"Error initializing empty output file {output_json_path}: {e}")
        return

    print(f"Preparing to process {len(samples_for_processing_tasks)} samples (failed retries, refused retries, or new).")

    api_key_manager = ApiKeyManager(keys=API_KEYS, base_url=API_BASE_URL)
    semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)
    
    tasks_to_await = []
    for sample_info_dict in samples_for_processing_tasks:
        tasks_to_await.append(async_process_sample(
            sample_info=sample_info_dict, 
            api_key_manager=api_key_manager, semaphore=semaphore, model=MODEL,
            max_retries=MAX_RETRIES, retry_delay_s=RETRY_DELAY_S, api_timeout_s=API_TIMEOUT_S
        ))

    successful_processed_count_this_run = 0
    failed_processed_count_this_run = 0

    for future in tqdm(asyncio.as_completed(tasks_to_await), total=len(tasks_to_await), desc="Processing"):
        result = await future
        if result:
            key_of_result = get_composite_key(result)
            if key_of_result:
                existing_results_map[key_of_result] = result 
            else:
                print(f"Warning: Processing result could not generate composite key, may not save correctly: {str(result)[:200]}")

            if result.get("status") == "success":
                if result.get("CoT_answer", "").strip() == REFUSAL_ANSWER:
                    print(f"Note: Sample {key_of_result} returned refusal answer but API status is 'success'. Will be retried next run.")
                    successful_processed_count_this_run += 1 
                else:
                    successful_processed_count_this_run += 1
            else:
                failed_processed_count_this_run += 1
    
    try:
        final_output_list = list(existing_results_map.values())
        with open(output_json_path, "w", encoding="utf-8") as outfile:
            json.dump(final_output_list, outfile, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error serializing or writing final output: {e}")

    print(f"\n✓ Processing complete.")
    print(f"  Successfully processed/retried successfully this run (based on API status): {successful_processed_count_this_run} records.")
    print(f"  Failed processing/retries this run (based on API status): {failed_processed_count_this_run} records.")
    print(f"  Total {len(existing_results_map)} relevant records written/updated to {output_json_path}")

def main_cli() -> None:
    input_path = "./MMRB_data.json" 
    output_dir = "./only_answer_output"
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"DEBUG: Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error: Cannot create output directory {output_dir}: {e}")
            return
            
    output_path_json = os.path.join(output_dir, "answer_grok-3-think_only_answer.json")
    
    print(f"DEBUG: Input path: {os.path.abspath(input_path)}")
    print(f"DEBUG: Output path: {os.path.abspath(output_path_json)}")

    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} does not exist.")
        return

    if not API_KEYS or not any(key.strip() for key in API_KEYS if key): 
        print("Error: API_KEYS list is empty or contains only invalid/empty keys. Please configure valid API keys in the code.")
        return
        
    asyncio.run(process_concurrently(input_path, output_path_json))

if __name__ == "__main__":
    main_cli()