import os
import json
import base64
import time
import argparse
import asyncio
from typing import List, Dict, Any, Tuple, Optional
import math 
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI, OpenAIError

API_BASE_URL = os.getenv("OPENAI_BASE_URL", "your base url")
API_KEYS = [
"sk-...",
"sk-..."
]
MODEL = "gemini-2.5-flash-preview-04-17-thinking"
MAX_RETRIES = 1
RETRY_DELAY_S = 1
CONCURRENT_LIMIT = 30
API_TIMEOUT_S = 1000
MAX_IMAGES_TO_SAMPLE = 16

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

def image_msgs(paths: List[str]) -> Tuple[List[Dict[str, Any]], int, int]:
 
    if not paths:
        return [], 0, 0

    valid_paths = []
    for p in paths:
        if isinstance(p, str) and p.strip() and os.path.exists(p):
            valid_paths.append(p)

    original_valid_count = len(valid_paths)
    processed_paths = []

    if original_valid_count > MAX_IMAGES_TO_SAMPLE:
        print(f"DEBUG: Original image count {original_valid_count} > {MAX_IMAGES_TO_SAMPLE}, performing equidistant sampling.")
        
        indices_to_sample = []
        if MAX_IMAGES_TO_SAMPLE == 0 :
            pass
        elif MAX_IMAGES_TO_SAMPLE == 1 and original_valid_count >= 1:
             indices_to_sample.append(0)
        elif original_valid_count > 0 : 
            for i in range(MAX_IMAGES_TO_SAMPLE):
                idx_float = i * (original_valid_count - 1) / (MAX_IMAGES_TO_SAMPLE - 1) if MAX_IMAGES_TO_SAMPLE > 1 else 0
                idx_int = int(round(idx_float))
                idx_int = max(0, min(idx_int, original_valid_count - 1))
                indices_to_sample.append(idx_int)
        
        unique_indices = sorted(list(set(indices_to_sample)))
        
        for idx in unique_indices:
            processed_paths.append(valid_paths[idx])
        print(f"DEBUG: Sampled image count {len(processed_paths)}. Selected indices (unique): {unique_indices}")

    else: 
        processed_paths = valid_paths
        if valid_paths:
            print(f"DEBUG: Original image count {len(processed_paths)} <= {MAX_IMAGES_TO_SAMPLE}, using all.")

    final_image_messages = [
        {"type": "image_url",
         "image_url": {"url": f"data:image/{os.path.splitext(p)[1][1:].lower()};base64,{encode_image(p)}"},
         "detail":"low"}
        for p in processed_paths
    ]
    return final_image_messages, original_valid_count, len(processed_paths)


def build_prompt(sample: Dict[str, Any], original_image_count: int, processed_image_count: int) -> str:
    sampling_notification = ""
    if original_image_count > MAX_IMAGES_TO_SAMPLE and processed_image_count == MAX_IMAGES_TO_SAMPLE:
        sampling_notification = (
            f"Please note: The provided visual information initially contained {original_image_count} images."
            f"These images have been equidistantly sampled down to {processed_image_count} images.\n\n"
        )

    question_text = sample.get('question', '')
    
    if sample.get("question_type") == "multi-choice":
        instructions = (
                        "Please firstly reason step by step.\n"
            "Then write your final answer in the format: Answer[<letter(s)>]. "
            "Only include the option letter(s), not the content."
        )
    else:
        instructions = (
           "Please firstly reason step by step.\n"
            "Then write your final answer in the format: Answer[<your_answer_here>]."
        )
    return f"{sampling_notification}{question_text}\n\n{instructions}"

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

    image_messages_list, original_img_count, processed_img_count = image_msgs(sample_data.get("image_paths", []))
    
    prompt_text = build_prompt(sample_data, original_img_count, processed_img_count)
    
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
                    {"role": "user", "content": [{"type": "text", "text": prompt_text}, *image_messages_list]},
                ]

                resp = await client.chat.completions.create(model=model, temperature=0.0, messages=messages_payload)
                cot_answer = resp.choices[0].message.content

                result_output = sample_data.copy()
                result_output["CoT_answer"] = cot_answer
                result_output["model_used"] = model
                result_output["api_key_hint"] = current_api_key_for_error_msg
                result_output["status"] = "success"
                result_output["original_image_count"] = original_img_count 
                result_output["processed_image_count"] = processed_img_count 
                return result_output

            except OpenAIError as e:
                error_message = f"OpenAIError for {composite_key} (key hint {current_api_key_for_error_msg}) on attempt {attempt + 1}/{max_retries}: {type(e).__name__} - {e}"
                if attempt + 1 >= max_retries:
                    print(f"[{composite_key}] All retries failed (OpenAIError). Error: {e}")
                    failed_result = sample_data.copy()
                    failed_result.update({
                        "error": error_message, "CoT_answer": "ERROR_MAX_RETRIES_OPENAI", "status": "failed",
                        "model_used": model, "original_image_count": original_img_count, "processed_image_count": processed_img_count
                    })
                    return failed_result
                else:
                    print(f"INFO: [{composite_key}] OpenAIError on attempt {attempt + 1}/{max_retries}. Retrying in {retry_delay_s * (attempt + 1)}s... Error: {type(e).__name__}")
                await asyncio.sleep(retry_delay_s * (attempt + 1))
            except Exception as e:
                error_message = f"Generic error for {composite_key} (key hint {current_api_key_for_error_msg}) on attempt {attempt + 1}/{max_retries}: {type(e).__name__} - {e}"
                if "image_url" in str(e).lower() and ("invalid" in str(e).lower() or "could not be accessed" in str(e).lower()):
                    print(f"ERROR: [{composite_key}] Critical image error: {e}. Marking as failed without further retries.")
                    failed_result = sample_data.copy()
                    failed_result.update({
                        "error": f"Critical image error: {e}", "CoT_answer": "ERROR_INVALID_IMAGE_DATA", "status": "failed",
                        "model_used": model, "original_image_count": original_img_count, "processed_image_count": processed_img_count
                    })
                    return failed_result
                if attempt + 1 >= max_retries:
                    print(f"ERROR: [{composite_key}] All retries failed (Generic Error). Error: {e}")
                    failed_result = sample_data.copy()
                    failed_result.update({
                        "error": error_message, "CoT_answer": "ERROR_MAX_RETRIES_GENERIC", "status": "failed",
                        "model_used": model, "original_image_count": original_img_count, "processed_image_count": processed_img_count
                    })
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
        fallback_result.update({
            "error": "Exhausted retries unexpectedly", "CoT_answer": "ERROR_UNEXPECTED_RETRY_EXHAUSTION", "status": "failed",
            "model_used": model, "original_image_count": original_img_count, "processed_image_count": processed_img_count
        })
        return fallback_result

async def process_concurrently(input_json_path: str, output_json_path: str) -> None:
    try:
        with open(input_json_path, "r", encoding="utf-8") as f:
            all_rows_raw = json.load(f)
        if not isinstance(all_rows_raw, list):
            print(f"ERROR: The top-level structure of the input file {input_json_path} is not a JSON list.")
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
        print(f"WARNING: {malformed_input_count} records in the input file were skipped due to format issues or inability to generate a composite key.")
    if duplicate_input_keys > 0:
        print(f"WARNING: {duplicate_input_keys} duplicate composite keys in the input file were ignored (keeping the first occurrence).")
    if not input_samples_map:
        print("ERROR: No valid input samples after processing. Please check the input file format and composite key fields.")
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
                            is_failed = str(status_val).strip().lower() == "failed" or \
                                                 (isinstance(cot_val, str) and cot_val.startswith("ERROR_"))

                            if prev_comp_key in temp_key_set_for_slice:
                                if is_failed:
                                    samples_to_reprocess_keys.add(prev_comp_key)
                                else:
                                    existing_results_map[prev_comp_key] = prev_result

                        if malformed_output_records > 0:
                            print(f"DEBUG: Skipped {malformed_output_records} malformed or keyless records in {output_json_path}.")
        except json.JSONDecodeError as e:
            print(f"WARNING: JSON error parsing existing {output_json_path}: {e}. Treating as empty file.")
        except Exception as e:
            print(f"WARNING: Error reading or parsing existing {output_json_path}: {e}. Treating as empty file.")

    print(f"DEBUG: (After population) existing_results_map (successfully recorded) length: {len(existing_results_map)}")
    if existing_results_map: print(f"DEBUG: Some keys in existing_results_map (up to 2): {list(existing_results_map.keys())[:2]}")
    print(f"DEBUG: (After population) samples_to_reprocess_keys (failed records to retry) length: {len(samples_to_reprocess_keys)}")
    if samples_to_reprocess_keys: print(f"DEBUG: Some keys in samples_to_reprocess_keys (up to 2): {list(samples_to_reprocess_keys)[:2]}")

    samples_for_processing_tasks = []
    newly_added_count = 0
    reprocessed_added_count = 0
    skipped_as_successful_count = 0

    for current_key in considered_input_slice_keys:
        original_sample_data = input_samples_map.get(current_key)
        if not original_sample_data:
            print(f"WARNING: Composite key {current_key} is in the input slice but not found in the deduped original input map. Skipping.")
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
        print("No samples to reprocess or add.")
        if existing_results_map:
            try:
                with open(output_json_path, "w", encoding="utf-8") as outfile:
                    json.dump(list(existing_results_map.values()), outfile, ensure_ascii=False, indent=4)
                print(f"Wrote/updated {len(existing_results_map)} existing successful records to {output_json_path} (no new tasks processed)")
            except Exception as e:
                print(f"Error writing existing successful records to {output_json_path}: {e}")
        return

    print(f"Preparing to process {len(samples_for_processing_tasks)} samples (failed retries or new additions).")

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

    for future in tqdm(asyncio.as_completed(tasks_to_await), total=len(tasks_to_await), desc="Inferring"):
        result = await future
        if result:
            key_of_result = get_composite_key(result)
            if key_of_result:
                existing_results_map[key_of_result] = result
            else:
                print(f"WARNING: Processed result could not generate a composite key, may not be saved correctly: {str(result)[:200]}")

            if result.get("status") == "success":
                successful_processed_count_this_run += 1
            else:
                failed_processed_count_this_run += 1

    try:
        final_output_list = list(existing_results_map.values())
        with open(output_json_path, "w", encoding="utf-8") as outfile:
            json.dump(final_output_list, outfile, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error serializing or writing final output: {e}")

    print(f"\n✓ Inference complete.")
    print(f"   Successfully processed/retried this run: {successful_processed_count_this_run} records.")
    print(f"   Failed processing/retried this run: {failed_processed_count_this_run} records.")
    print(f"   A total of {len(existing_results_map)} relevant records have been written/updated to {output_json_path}")

def main_cli() -> None:
    input_path = "./MMRB_data_mini10.json"
    output_dir = "./cot_output"
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"DEBUG: Created output directory: {output_dir}")
        except OSError as e:
            print(f"ERROR: Could not create output directory {output_dir}: {e}")
            return

    output_path_json = os.path.join(output_dir, "answer_gemini-2.5-flash-preview-04-17-thinking_MMRB_data_mini10_cot.json")

    print(f"DEBUG: Input path: {os.path.abspath(input_path)}")
    print(f"DEBUG: Output path: {os.path.abspath(output_path_json)}")

    if not os.path.exists(input_path):
        print(f"ERROR: Input file {input_path} does not exist.")
        return

    if not API_KEYS or not any(key.strip() for key in API_KEYS if key):
        print("ERROR: API_KEYS list is empty or contains only invalid/empty keys. Please configure valid API keys in the code.")
        return

    asyncio.run(process_concurrently(input_path, output_path_json))

if __name__ == "__main__":
    main_cli()



