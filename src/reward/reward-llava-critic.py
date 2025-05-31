from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from PIL import Image
import copy
import torch
import json
import sys
import warnings
import os
import gc
from collections import defaultdict
import csv 

warnings.filterwarnings("ignore")
torch.cuda.set_device(1)

def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()

pretrained = "./lmms-lab/llava-critic-7b"
model_name = "llava_qwen"
device_map = "cuda:1"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)

model.eval()

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

json_data = load_json_data("MMRB_data_reward.json")

stats = {
    "total_samples": 0,
    "total_paths": 0,
    "first_choice": 0,
    "second_choice": 0,
    "equally_good": 0,
    "no_match_found": 0,
    "by_subtask": defaultdict(lambda: {"total": 0, "first": 0, "second": 0, "equal": 0, "no_match": 0})
}

batch_size = 1  
image_cache = {}  
initial_max_new_tokens = 4096 

for sample_idx, sample in enumerate(json_data):
    try:
        stats["total_samples"] += 1
        
        available_path_indices = set()
        for path_data in sample["modified_reasoning_steps"]:
            available_path_indices.add(path_data["path_index"])

        image_tensors = []
        image_sizes = []

        all_images = sample["image_paths"]

        for img_path in all_images:
            cache_key = f"{img_path}"
            if cache_key in image_cache:
                image_tensor, image_size = image_cache[cache_key]
            else:
                image = Image.open(img_path)
                image_size = image.size
                processed_images = process_images([image], image_processor, model.config)
                image_tensor = processed_images[0].to(device="cuda:1", dtype=torch.float16)
                image_cache[cache_key] = (image_tensor, image_size)
                del image
                del processed_images
            image_tensors.append(image_tensor)
            image_sizes.append(image_size)

        conv_template = "qwen_1_5"
        
        for path_idx, path_index in enumerate(sorted(available_path_indices)):
            stats["total_paths"] += 1
            stats["by_subtask"][sample["subtask"]]["total"] += 1

            if path_idx > 0 and path_idx % 5 == 0:
                clear_gpu_memory()

            modified_steps = get_reasoning_steps_text(sample["modified_reasoning_steps"], path_index, True)
            raw_steps = get_reasoning_steps_text(sample["raw_reasoning_steps"], path_index, False)

            image_tokens = ""
            for i in range(len(all_images)):
                image_tokens += f'"image#{i}":{DEFAULT_IMAGE_TOKEN}\n'

            critic_prompt = f"""{{"from": "human", "value": "{image_tokens}
You are provided with some images and a question for these images. Please review the corresponding responses based on the following 5 factors:

1. Accuracy in Object Description: Evaluate the accuracy of the descriptions concerning the objects mentioned in the ground truth answer. Responses should minimize the mention of objects not present in the ground truth answer, and inaccuracies in the description of existing objects.

2. Accuracy in Depicting Relationships: Consider how accurately the relationships between objects are described compared to the ground truth answer. Rank higher the responses that least misrepresent these relationships.

3. Accuracy in Describing Attributes: Assess the accuracy in the depiction of objects' attributes compared to the ground truth answer. Responses should avoid inaccuracies in describing the characteristics of the objects present.

4. Helpfulness: Consider whether the generated text provides valuable insights, additional context, or relevant information that contributes positively to the user's comprehension of the image. Assess whether the language model accurately follows any specific instructions or guidelines provided in the prompt. Evaluate the overall contribution of the response to the user experience.

IMPORTANT INSTRUCTION: You MUST choose either Response 1 or Response 2 as better, even if the difference is extremely subtle. "Equally good" is NOT a valid answer. If you perceive the responses as very similar in quality, you must still identify and focus on even the smallest advantages one has over the other to make your decision.

You need to choose which response is better for the given question and provide a detailed reason.

Your task is provided as follows:

Question: {sample['question']}

Answer:
{sample['answer']}
Response 1:
{modified_steps}

Response 2:
{raw_steps}

ASSISTANT:
"}}"""
            
            print(f"\nProcessing sample {sample_idx + 1}/{len(json_data)}, path_index: {path_index} (Subtask: {sample['subtask']})")
            
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], critic_prompt)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to("cuda:1")

            current_max_tokens = initial_max_new_tokens
            found_target_phrase = False
            attempt_count = 0
            output_text = ""

            while attempt_count < 3 and not found_target_phrase:
                if attempt_count > 0:
                    current_max_tokens *= 2
                    print(f"Attempt {attempt_count + 1}/3, increasing tokens to: {current_max_tokens}")
                
                print(f"Model generating (attempt {attempt_count + 1}/3, max_new_tokens: {current_max_tokens})...")
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    cont = model.generate(
                        input_ids,
                        images=image_tensors,
                        image_sizes=image_sizes,
                        do_sample=False,
                        temperature=0,
                        max_new_tokens=current_max_tokens,
                    )
                
                text_outputs_list = tokenizer.batch_decode(cont, skip_special_tokens=True)
                output_text = text_outputs_list[0].strip() if text_outputs_list else ""

                if "The better response: [1]." in output_text:
                    stats["first_choice"] += 1
                    stats["by_subtask"][sample["subtask"]]["first"] += 1
                    found_target_phrase = True
                    print("Model choice: Response 1")
                elif "The better response: [2]." in output_text:
                    stats["second_choice"] += 1
                    stats["by_subtask"][sample["subtask"]]["second"] += 1
                    found_target_phrase = True
                    print("Model choice: Response 2")
                elif "Two responses are equally good." in output_text:
                    stats["equally_good"] += 1
                    stats["by_subtask"][sample["subtask"]]["equal"] += 1
                    found_target_phrase = True
                    print("Model choice: Equally good")
                
                if found_target_phrase:
                    break 
                
                attempt_count += 1
                del cont
                del text_outputs_list
                clear_gpu_memory()

            if not found_target_phrase:
                stats["no_match_found"] += 1
                stats["by_subtask"][sample["subtask"]]["no_match"] += 1
                print(f"Warning: After 3 attempts, target phrase not found in output for sample {sample_idx + 1}, path_index {path_index}.")
                print(f"Output from last attempt: {output_text[:500]}...")
            else:
                print(f"Final response (partial): {output_text[:300]}...")

            del input_ids
            if 'cont' in locals() and cont is not None: del cont
            if 'text_outputs_list' in locals() and text_outputs_list is not None: del text_outputs_list
            clear_gpu_memory()

        if len(image_cache) > 10:
            image_cache.clear()
        clear_gpu_memory()
        
    except Exception as e:
        print(f"Error processing sample {sample_idx + 1} (ID: {sample.get('id', 'N/A')}): {e}")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(f"Error type: {exc_type}, file: {fname}, line: {exc_tb.tb_lineno}")
        clear_gpu_memory()
        continue

print("\nFinal Statistics:")
print(f"Total samples processed: {stats['total_samples']}")
print(f"Total paths evaluated: {stats['total_paths']}")
print(f"Times Response 1 chosen: {stats['first_choice']}")
print(f"Times Response 2 chosen: {stats['second_choice']}")
print(f"Times responses equally good: {stats['equally_good']}")
print(f"Times no match found: {stats['no_match_found']}")

if stats['total_paths'] > 0:
    print(f"Percentage Response 1 chosen: {stats['first_choice']/stats['total_paths']*100:.2f}%")
    print(f"Percentage Response 2 chosen: {stats['second_choice']/stats['total_paths']*100:.2f}%")
    print(f"Percentage equally good: {stats['equally_good']/stats['total_paths']*100:.2f}%")
    print(f"Percentage no match found: {stats['no_match_found']/stats['total_paths']*100:.2f}%")

csv_file_path = "./reward_model_stats_0515.csv"
try:
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(["Overall Statistics", "Value"])
        writer.writerow(["Total Samples Processed", stats['total_samples']])
        writer.writerow(["Total Paths Evaluated", stats['total_paths']])
        writer.writerow(["Times Response 1 Chosen", stats['first_choice']])
        writer.writerow(["Times Response 2 Chosen", stats['second_choice']])
        writer.writerow(["Times Responses Equally Good", stats['equally_good']])
        writer.writerow(["Times No Match Found", stats['no_match_found']])
        if stats['total_paths'] > 0:
            writer.writerow(["Percentage Response 1 Chosen", f"{stats['first_choice']/stats['total_paths']*100:.2f}%"])
            writer.writerow(["Percentage Response 2 Chosen", f"{stats['second_choice']/stats['total_paths']*100:.2f}%"])
            writer.writerow(["Percentage Equally Good", f"{stats['equally_good']/stats['total_paths']*100:.2f}%"])
            writer.writerow(["Percentage No Match Found", f"{stats['no_match_found']/stats['total_paths']*100:.2f}%"])
        writer.writerow([])

        writer.writerow(["Subtask", "Total Paths", "Response 1 Chosen", "Response 2 Chosen", "Equally Good", "No Match Found",
                          "Response 1 %", "Response 2 %", "Equally Good %", "No Match %"])

        for subtask, s_stats in stats["by_subtask"].items():
            row = [
                subtask,
                s_stats['total'],
                s_stats['first'],
                s_stats['second'],
                s_stats['equal'],
                s_stats['no_match']
            ]
            if s_stats['total'] > 0:
                row.extend([
                    f"{s_stats['first']/s_stats['total']*100:.2f}%",
                    f"{s_stats['second']/s_stats['total']*100:.2f}%",
                    f"{s_stats['equal']/s_stats['total']*100:.2f}%",
                    f"{s_stats['no_match']/s_stats['total']*100:.2f}%"
                ])
            else:
                row.extend(["N/A", "N/A", "N/A", "N/A"])
            writer.writerow(row)

    print(f"\nStatistics saved to: {csv_file_path}")

except IOError:
    print(f"Error: Unable to write CSV file to path {csv_file_path}. Please check permissions or path.")
except Exception as e:
    print(f"Unknown error occurred while saving CSV: {e}")