import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from modelscope import AutoModel, AutoTokenizer 
from accelerate import Accelerator

import json
import sys
import warnings
import os
import gc
from collections import defaultdict
import csv
import argparse

warnings.filterwarnings("ignore")

# --- Model Helper Functions ---
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]: # type: ignore
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_ar_w = int(max(1, target_aspect_ratio[0]))
    target_ar_h = int(max(1, target_aspect_ratio[1]))

    target_width = image_size * target_ar_w
    target_height = image_size * target_ar_h
    blocks = target_ar_w * target_ar_h

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % target_ar_w) * image_size,
            (i // target_ar_w) * image_size,
            ((i % target_ar_w) + 1) * image_size,
            ((i // target_ar_w) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image_for_new_model(image_path, input_size=448, max_num=12):
    try:
        image = Image.open(image_path).convert('RGB')
        transform = build_transform(input_size=input_size)
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

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
            steps_text = []
            for step in path_data["path"]:
                if is_modified:
                    steps_text.append(f"步骤 {step['reasoning step']} ({step['reasoning type']}): {step['rationale_fix']}")
                else:
                    steps_text.append(f"步骤 {step['reasoning step']} ({step['reasoning type']}): {step['rationale']}")
            return "\n".join(steps_text)
    return ""

def main():
    if torch.cuda.is_available():
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description='Accelerate Inference for VisualPRM-8B')
    parser.add_argument('--json_file', type=str, default="MMRB_data_reward.json", help='path to the JSON data file')
    parser.add_argument('--model_path', type=str, default='/home/localuser/xbr/data/PRMModel', help='path to the model')
    args = parser.parse_args()

    accelerator = Accelerator(mixed_precision="bf16") # bf16 can be demanding, ensure your hardware supports it well. Otherwise, "fp16" or None.

    NEW_MODEL_IMAGE_SIZE = 448
    print(f"Using NEW_MODEL_IMAGE_SIZE: {NEW_MODEL_IMAGE_SIZE}")
    NEW_MODEL_MAX_NUM_PATCHES = 12
    print(f"Using NEW_MODEL_MAX_NUM_PATCHES: {NEW_MODEL_MAX_NUM_PATCHES}")

    processed_paths = set()
    stats = {
        "total_samples": 0,
        "total_paths": 0,
        "first_choice": 0,
        "second_choice": 0,
        "processing_errors": 0,
        "by_subtask": defaultdict(lambda: {"total": 0, "first": 0, "second": 0, "errors": 0}),
        "processed_paths": []
    }

    try:
        print(f"Loading tokenizer from: {args.model_path}")
        tokenizer_new_model = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            use_fast=False
        )

        print(f"Loading model from: {args.model_path}")
        if torch.cuda.is_available():
            print(f"Available devices: {torch.cuda.device_count()} GPUs")
            max_memory_setting = {i: f"{int(torch.cuda.get_device_properties(i).total_memory * 0.8 / 1024**3)}GiB" for i in range(torch.cuda.device_count())}
        else:
            print("No CUDA GPUs available. Model will be loaded on CPU.")
            max_memory_setting = {"cpu": f"{int(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') * 0.8 / 1024**3)}GiB"} # Approximate system RAM


        # 当 device_map="auto" 时，模型会自动分配到可用设备
        # 不需要手动将 pixel_values 移动到特定设备
        model_new = AutoModel.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            device_map="auto", # "auto" 意味着 accelerate 会处理设备放置
            max_memory=max_memory_setting,
            torch_dtype=torch.bfloat16 if accelerator.mixed_precision == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 if accelerator.mixed_precision == "fp16" and torch.cuda.is_available() else torch.float32, # Adjust dtype based on availability and accelerator setting
            offload_folder="offload",
        )

        model_new.eval()
        print(f"Model loaded successfully.")
        if hasattr(model_new, 'hf_device_map'):
            print(f"Device map: {model_new.hf_device_map}")

        if not os.path.exists(args.json_file):
            print(f"Error: JSON data file not found at {args.json_file}")
            sys.exit(1)

        json_data = load_json_data(args.json_file)

        if accelerator.is_main_process:
            for sample_idx, sample in enumerate(json_data):
                try:
                    stats["total_samples"] += 1
                    available_path_indices = set()
                    for path_data in sample.get("modified_reasoning_steps", []):
                        available_path_indices.add(path_data["path_index"])

                    pixel_values = None
                    if "image_paths" in sample and sample["image_paths"]:
                        image_path_to_load = sample["image_paths"][0]
                        if not os.path.exists(image_path_to_load):
                            print(f"Warning: Image file not found: {image_path_to_load} for sample {sample_idx + 1}.")
                        else:
                            print(f"Loading image: {image_path_to_load} for sample {sample_idx + 1}")
                            try:
                                pixel_values = load_image_for_new_model(
                                    image_path_to_load,
                                    input_size=NEW_MODEL_IMAGE_SIZE,
                                    max_num=NEW_MODEL_MAX_NUM_PATCHES
                                )
                                if pixel_values is not None:
                                    # 转换数据类型以匹配模型期望 (例如 bfloat16)
                                    # 注意: torch_dtype 在模型加载时已设置，这里确保 pixel_values 匹配
                                    target_dtype = model_new.dtype # Get model's dtype
                                    pixel_values = pixel_values.to(target_dtype)

                                    # --- MODIFICATION START ---
                                    # 当使用 device_map="auto" 时，不需要手动将 pixel_values 移动到特定设备。
                                    # 模型在接收输入时，会自动处理或期望输入在正确的设备上，
                                    # 或者 accelerate/transformers 会处理设备间的传输。
                                    # 显式调用 .to(device) 或 .cuda() 可能会与 device_map 冲突。
                                    #
                                    # 原有代码块:
                                    # if hasattr(model_new, 'hf_device_map'):
                                    #     devices = set(model_new.hf_device_map.values())
                                    #     # ... (logic to find first_device) ...
                                    #     pixel_values = pixel_values.to(first_device)
                                    # else:
                                    #     if torch.cuda.is_available():
                                    #         pixel_values = pixel_values.to('cuda:0')
                                    #     else:
                                    #         pixel_values = pixel_values.cpu()
                                    # --- MODIFICATION END ---

                                    print(f"Image processed successfully, shape: {pixel_values.shape}, dtype: {pixel_values.dtype}, current device: {pixel_values.device}")
                                    # pixel_values 的设备此时通常是 'cpu'，因为它是由 ToTensor() 创建的。
                                    # 模型调用时，Hugging Face 模型会负责将数据移动到其子模块所在的正确设备。
                                else:
                                    print(f"Warning: Failed to load/process image {image_path_to_load}.")
                            except Exception as img_e:
                                print(f"Error processing image {image_path_to_load}: {img_e}")
                                import traceback
                                traceback.print_exc()
                                pixel_values = None
                    else:
                        print(f"Warning: No image_paths found for sample {sample_idx + 1}.")

                    if pixel_values is None and "image_paths" in sample and sample["image_paths"]:
                        print(f"Skipping paths for sample {sample_idx + 1} due to image loading failure.")
                        for path_index in sorted(list(available_path_indices)):
                            stats["total_paths"] += 1
                            stats["by_subtask"][sample["subtask"]]["total"] += 1
                            stats["processing_errors"] += 1
                            stats["by_subtask"][sample["subtask"]]["errors"] += 1
                            stats["processed_paths"].append({"sample_idx": sample_idx, "path_index": path_index})
                        clear_gpu_memory()
                        continue

                    question_text = sample.get('question', "No question provided.")

                    for path_idx, path_index in enumerate(sorted(list(available_path_indices))):
                        path_key = {"sample_idx": sample_idx, "path_index": path_index}
                        if path_key in stats["processed_paths"]:
                            print(f"Skipping already processed path: sample {sample_idx + 1}, path {path_index}")
                            continue

                        stats["total_paths"] += 1
                        stats["by_subtask"][sample["subtask"]]["total"] += 1

                        if path_idx > 0 and path_idx % 2 == 0:
                            clear_gpu_memory()

                        modified_steps = get_reasoning_steps_text(sample["modified_reasoning_steps"], path_index, True)
                        raw_steps = get_reasoning_steps_text(sample["raw_reasoning_steps"], path_index, False)
                        response_list_for_model = [raw_steps, modified_steps]

                        print(f"Processing sample {sample_idx + 1}/{len(json_data)}, path_index: {path_index}")

                        try:

                            with torch.no_grad(): 
                                sorted_responses_with_scores = model_new.select_best_response(
                                    tokenizer=tokenizer_new_model,
                                    question=question_text,
                                    response_list=response_list_for_model,
                                    pixel_values=pixel_values, 
                                    return_scores=True,
                                )

                            if not sorted_responses_with_scores:
                                print(f"Warning: Model returned no responses for sample {sample_idx + 1}, path_index {path_index}.")
                                stats["processing_errors"] += 1
                                stats["by_subtask"][sample["subtask"]]["errors"] += 1
                                stats["processed_paths"].append(path_key)
                                continue

                            best_response_text = sorted_responses_with_scores[0][0]
                            best_score = sorted_responses_with_scores[0][1]

                            if best_response_text == modified_steps:
                                stats["first_choice"] += 1
                                stats["by_subtask"][sample["subtask"]]["first"] += 1
                                print(f"New model chose: Response 1 (Modified). Score: {best_score:.4f}")
                            elif best_response_text == raw_steps:
                                stats["second_choice"] += 1
                                stats["by_subtask"][sample["subtask"]]["second"] += 1
                                print(f"New model chose: Response 2 (Raw). Score: {best_score:.4f}")
                            else:
                                print(f"Warning: Best response text from model does not exactly match input.")
                                score_modified = None
                                score_raw = None
                                for r_text, r_score in sorted_responses_with_scores:
                                    if r_text == modified_steps: score_modified = r_score
                                    if r_text == raw_steps: score_raw = r_score

                                if score_modified is not None and score_raw is not None:
                                    if score_modified >= score_raw:
                                        stats["first_choice"] += 1
                                        stats["by_subtask"][sample["subtask"]]["first"] += 1
                                        print(f"New model chose (by score fallback): Response 1 (Modified). Score: {score_modified:.4f}")
                                    else:
                                        stats["second_choice"] += 1
                                        stats["by_subtask"][sample["subtask"]]["second"] += 1
                                        print(f"New model chose (by score fallback): Response 2 (Raw). Score: {score_raw:.4f}")
                                else:
                                    stats["processing_errors"] += 1
                                    stats["by_subtask"][sample["subtask"]]["errors"] += 1
                                    print(f"Could not determine choice due to text mismatch and score ambiguity.")
                            stats["processed_paths"].append(path_key)

                        except Exception as model_e:
                            print(f"Error during model inference: {model_e}")
                            stats["processing_errors"] += 1
                            stats["by_subtask"][sample["subtask"]]["errors"] += 1
                            stats["processed_paths"].append(path_key)
                            import traceback
                            traceback.print_exc()
                        finally:
                            if 'sorted_responses_with_scores' in locals():
                                del sorted_responses_with_scores
                            clear_gpu_memory()

                    if pixel_values is not None:
                        del pixel_values
                    clear_gpu_memory()

                    if sample_idx % 5 == 0 and sample_idx > 0:
                        print(f"Saving interim results after processing {sample_idx} samples...")
                        try:
                            with open(f"interim_results_{sample_idx}.json", 'w') as f:
                                serializable_stats = {k: v for k, v in stats.items() if k != "processed_paths"}
                                serializable_stats["processed_paths"] = [dict(p) for p in stats["processed_paths"]]
                                serializable_stats["by_subtask"] = {k: dict(v) for k, v in stats["by_subtask"].items()}
                                json.dump(serializable_stats, f)
                        except Exception as e:
                            print(f"Error saving interim results: {e}")

                except Exception as e:
                    print(f"FATAL: Unhandled error processing sample {sample_idx + 1}: {e}")
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    if exc_tb is not None:
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print(f"Error type: {exc_type}, File: {fname}, Line: {exc_tb.tb_lineno}")

                    for path_index in sorted(list(available_path_indices)):
                        path_key = {"sample_idx": sample_idx, "path_index": path_index}
                        if path_key not in stats["processed_paths"]:
                            stats["total_paths"] += 1
                            stats["processing_errors"] += 1
                            stats["by_subtask"][sample["subtask"]]["total"] += 1
                            stats["by_subtask"][sample["subtask"]]["errors"] += 1
                            stats["processed_paths"].append(path_key)
                    clear_gpu_memory()
                    continue

            print("\n--- Final Evaluation Results ---")
            print(f"Total Samples Processed: {stats['total_samples']}")
            print(f"Total Evaluation Paths: {stats['total_paths']}")
            print(f"Choices for Response 1 (Modified): {stats['first_choice']}")
            print(f"Choices for Response 2 (Raw): {stats['second_choice']}")
            print(f"Processing Errors / No Clear Choice: {stats['processing_errors']}")

            if stats['total_paths'] > 0:
                valid_choices = stats['first_choice'] + stats['second_choice']
                if valid_choices > 0:
                    print(f"  Accuracy (Response 1 / Total Valid Choices): {stats['first_choice']/valid_choices*100:.2f}%")
                    print(f"  Accuracy (Response 2 / Total Valid Choices): {stats['second_choice']/valid_choices*100:.2f}%")
                print(f"Proportion of Response 1 Chosen (overall paths): {stats['first_choice']/stats['total_paths']*100:.2f}%")
                print(f"Proportion of Response 2 Chosen (overall paths): {stats['second_choice']/stats['total_paths']*100:.2f}%")
                print(f"Proportion of Errors (overall paths): {stats['processing_errors']/stats['total_paths']*100:.2f}%")

            print("\n--- Statistics by Subtask ---")
            for subtask, s_stats in stats["by_subtask"].items():
                print(f"\nSubtask: {subtask}")
                print(f"  Total Paths: {s_stats['total']}")
                print(f"  Choices for Response 1: {s_stats['first']}")
                print(f"  Choices for Response 2: {s_stats['second']}")
                print(f"  Errors/No Clear Choice: {s_stats['errors']}")
                if s_stats['total'] > 0:
                    sub_valid_choices = s_stats['first'] + s_stats['second']
                    if sub_valid_choices > 0:
                        print(f"    Accuracy (R1 / Valid Choices): {s_stats['first']/sub_valid_choices*100:.2f}%")
                        print(f"    Accuracy (R2 / Valid Choices): {s_stats['second']/sub_valid_choices*100:.2f}%")
                    print(f"    Proportion R1 (overall paths): {s_stats['first']/s_stats['total']*100:.2f}%")
                    print(f"    Proportion R2 (overall paths): {s_stats['second']/s_stats['total']*100:.2f}%")
                    print(f"    Proportion Errors (overall paths): {s_stats['errors']/s_stats['total']*100:.2f}%")

            csv_file_path = "visualprm_evaluation_stats_reverse.csv"
            try:
                with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["Overall Statistics", "Value"])
                    writer.writerow(["Total Samples Processed", stats['total_samples']])
                    writer.writerow(["Total Paths Evaluated", stats['total_paths']])
                    writer.writerow(["Choices for Response 1 (Modified)", stats['first_choice']])
                    writer.writerow(["Choices for Response 2 (Raw)", stats['second_choice']])
                    writer.writerow(["Processing Errors / No Clear Choice", stats['processing_errors']])
                    if stats['total_paths'] > 0:
                        valid_choices = stats['first_choice'] + stats['second_choice']
                        if valid_choices > 0:
                            writer.writerow(["Accuracy (R1 / Valid Choices)", f"{stats['first_choice']/valid_choices*100:.2f}%"])
                            writer.writerow(["Accuracy (R2 / Valid Choices)", f"{stats['second_choice']/valid_choices*100:.2f}%"])
                        else:
                            writer.writerow(["Accuracy (R1 / Valid Choices)", "N/A"])
                            writer.writerow(["Accuracy (R2 / Valid Choices)", "N/A"])
                        writer.writerow(["Proportion R1 (overall paths)", f"{stats['first_choice']/stats['total_paths']*100:.2f}%"])
                        writer.writerow(["Proportion R2 (overall paths)", f"{stats['second_choice']/stats['total_paths']*100:.2f}%"])
                        writer.writerow(["Proportion Errors (overall paths)", f"{stats['processing_errors']/stats['total_paths']*100:.2f}%"])
                    writer.writerow([])

                    writer.writerow(["Subtask", "Total Paths", "Response 1 Chosen", "Response 2 Chosen",
                                    "Errors/No Choice", "Accuracy R1 (of valid)", "Accuracy R2 (of valid)",
                                    "Prop. R1 (of total)", "Prop. R2 (of total)", "Prop. Errors (of total)"])

                    for subtask, s_stats in stats["by_subtask"].items(): 
                        row_data = [
                            subtask, s_stats['total'], s_stats['first'], s_stats['second'], s_stats['errors'] 
                        ]
                        if s_stats['total'] > 0: 
                            sub_valid = s_stats['first'] + s_stats['second'] 
                            row_data.extend([
                                f"{s_stats['first']/sub_valid*100:.2f}%" if sub_valid > 0 else "N/A", 
                                f"{s_stats['second']/sub_valid*100:.2f}%" if sub_valid > 0 else "N/A", 
                                f"{s_stats['first']/s_stats['total']*100:.2f}%", 
                                f"{s_stats['second']/s_stats['total']*100:.2f}%",
                                f"{s_stats['errors']/s_stats['total']*100:.2f}%"
                            ])
                        else:
                            row_data.extend(["N/A"] * 5)
                        writer.writerow(row_data)
                print(f"\nEvaluation statistics saved to: {csv_file_path}")
            except IOError:
                print(f"Error: Could not write CSV file to {csv_file_path}. Check permissions or path.")
            except Exception as e:
                print(f"An unexpected error occurred while saving CSV: {e}")

    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()