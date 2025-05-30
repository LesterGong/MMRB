from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch
import json
import sys
import warnings
import os
import gc
from collections import defaultdict
import csv  # 导入csv模块

warnings.filterwarnings("ignore")

# 设置CUDA设备为cuda:1
torch.cuda.set_device(1)

# 显存管理函数
def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()

pretrained = "/data/user/zc/ts2/.cache/modelscope/hub/models/lmms-lab/llava-critic-7b"
model_name = "llava_qwen"
device_map = "cuda:1"  # 移动到CUDA:0
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)

model.eval()

# 读取JSON文件
def load_json_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 获取推理步骤文本
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

# 加载数据
json_data = load_json_data("MMRB_data_compare.json")

# 统计结果
stats = {
    "total_samples": 0,
    "total_paths": 0,
    "first_choice": 0,
    "second_choice": 0,
    "equally_good": 0,
    "no_match_found": 0,
    "by_subtask": defaultdict(lambda: {"total": 0, "first": 0, "second": 0, "equal": 0, "no_match": 0})
}

# 设置批处理大小和图像缓存
batch_size = 1  # 每次处理一个样本
image_cache = {}  # 缓存已处理的图像
initial_max_new_tokens = 4096  # 初始的max_new_tokens值

# 处理每个样本
for sample_idx, sample in enumerate(json_data):
    try:
        stats["total_samples"] += 1
        
        available_path_indices = set()
        for path_data in sample["modified_reasoning_steps"]:
            available_path_indices.add(path_data["path_index"])

        # 处理图片
        image_tensors = []
        image_sizes = []

        # 新增部分：根据图片数量进行等距采样
        num_images = len(sample["image_paths"])
        if num_images > 20:
            # 超过20张图片时，等距采样为20张
            sampled_images = [sample["image_paths"][i] for i in range(0, num_images, max(1, num_images // 20))]
            if len(sampled_images) > 20:
                sampled_images = sampled_images[:20]  # 如果结果大于20，则只取前20张
        else:
            # 图片数量小于等于20张时不做处理
            sampled_images = sample["image_paths"]

        # 去除缩放部分，直接使用原始尺寸
        for img_path in sampled_images:
            cache_key = f"{img_path}"
            if cache_key in image_cache:
                image_tensor, image_size = image_cache[cache_key]
            else:
                image = Image.open(img_path)
                image_size = image.size
                processed_images = process_images([image], image_processor, model.config)
                image_tensor = processed_images[0].to(device="cuda:1", dtype=torch.float16)  # 迁移到cuda:0
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
            for i in range(len(sampled_images)):
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
            
            print(f"\n处理样本 {sample_idx + 1}/{len(json_data)}, path_index: {path_index} (子任务: {sample['subtask']})")
            
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
                    print(f"尝试次数 {attempt_count + 1}/3，增加 token 数至: {current_max_tokens}")
                
                print(f"模型生成中 (尝试 {attempt_count + 1}/3, max_new_tokens: {current_max_tokens})...")
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
                    print("模型选择: Response 1")
                elif "The better response: [2]." in output_text:
                    stats["second_choice"] += 1
                    stats["by_subtask"][sample["subtask"]]["second"] += 1
                    found_target_phrase = True
                    print("模型选择: Response 2")
                elif "Two responses are equally good." in output_text:
                    stats["equally_good"] += 1
                    stats["by_subtask"][sample["subtask"]]["equal"] += 1
                    found_target_phrase = True
                    print("模型选择: 两者一样好")
                
                if found_target_phrase:
                    break 
                
                attempt_count += 1
                del cont
                del text_outputs_list
                clear_gpu_memory()

            if not found_target_phrase:
                stats["no_match_found"] += 1
                stats["by_subtask"][sample["subtask"]]["no_match"] += 1
                print(f"警告: 尝试3次后，在样本 {sample_idx + 1}, path_index {path_index} 的输出中未找到指定短语。")
                print(f"最后一次尝试的输出: {output_text[:500]}...")
            else:
                 print(f"最终回答 (部分): {output_text[:300]}...")

            del input_ids
            if 'cont' in locals() and cont is not None: del cont
            if 'text_outputs_list' in locals() and text_outputs_list is not None: del text_outputs_list
            clear_gpu_memory()

        if len(image_cache) > 10:
            image_cache.clear()
        clear_gpu_memory()
        
    except Exception as e:
        print(f"处理样本 {sample_idx + 1} (ID: {sample.get('id', 'N/A')}) 时出错: {e}")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(f"错误类型: {exc_type}, 文件: {fname}, 行号: {exc_tb.tb_lineno}")
        clear_gpu_memory()
        continue

# 打印并保存统计结果
print("\n最终统计结果:")
print(f"总样本数: {stats['total_samples']}")
print(f"总评估路径数: {stats['total_paths']}")
print(f"选择Response 1的次数: {stats['first_choice']}")
print(f"选择Response 2的次数: {stats['second_choice']}")
print(f"认为两者一样好的次数: {stats['equally_good']}")
print(f"未匹配到指定回复的次数: {stats['no_match_found']}")

if stats['total_paths'] > 0:
    print(f"选择Response 1的比例: {stats['first_choice']/stats['total_paths']*100:.2f}%")
    print(f"选择Response 2的比例: {stats['second_choice']/stats['total_paths']*100:.2f}%")
    print(f"认为两者一样好的比例: {stats['equally_good']/stats['total_paths']*100:.2f}%")
    print(f"未匹配到指定回复的比例: {stats['no_match_found']/stats['total_paths']*100:.2f}%")

# 保存统计结果到CSV文件
csv_file_path = "./reward_model_stats_0515.csv"
try:
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # 写入总体统计
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

        # 写入按子任务统计的表头
        writer.writerow(["Subtask", "Total Paths", "Response 1 Chosen", "Response 2 Chosen", "Equally Good", "No Match Found",
                         "Response 1 %", "Response 2 %", "Equally Good %", "No Match %"])

        # 写入每个子任务的统计
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

    print(f"\n统计结果已保存到: {csv_file_path}")

except IOError:
    print(f"错误: 无法写入CSV文件到路径 {csv_file_path}。请检查权限或路径。")
except Exception as e:
    print(f"保存CSV时发生未知错误: {e}")

# from llava.model.builder import load_pretrained_model
# from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
# from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
# from llava.conversation import conv_templates, SeparatorStyle

# from PIL import Image
# import requests
# import copy
# import torch
# import json
# import sys
# import warnings
# import os
# import gc
# from collections import defaultdict
# import csv
# import numpy as np

# os.environ['http_proxy'] = "http://127.0.0.1:7890"
# os.environ['https_proxy'] = "http://127.0.0.1:7890"
# warnings.filterwarnings("ignore")

# # 显存管理函数
# def clear_gpu_memory():
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()

# # --- MODIFICATION POINT 1: Model Loading and Device ---
# # Ensure device consistency. model.device will hold the correct device after loading.
# pretrained = "/data/user/zc/ts2/.cache/modelscope/hub/models/lmms-lab/llava-critic-7b"
# model_name = "llava_qwen"
# device_map = "cuda:1" # Or "auto" or specific device
# tokenizer, model, image_processor, max_length = load_pretrained_model(
#     pretrained,
#     None,
#     model_name,
#     device_map=device_map,
#     attn_implementation="eager" # Using "eager" as in original, consider "flash_attention_2" if supported and beneficial
# )

# model.eval()
# # Use model.device for subsequent tensor placements to ensure consistency
# current_device = model.device
# print(f"Model loaded on device: {current_device}")

# # 读取JSON文件
# def load_json_data(json_path):
#     with open(json_path, 'r', encoding='utf-8') as f:
#         return json.load(f)

# # 获取推理步骤文本
# def get_reasoning_steps_text(steps, path_index, is_modified=False):
#     for path_data in steps:
#         if path_data["path_index"] == path_index:
#             steps_text = []
#             for step in path_data["path"]:
#                 if is_modified:
#                     steps_text.append(f"步骤 {step['reasoning step']} ({step['reasoning type']}): {step['rationale_fix']}")
#                 else:
#                     steps_text.append(f"步骤 {step['reasoning step']} ({step['reasoning type']}): {step['rationale']}")
#             return "\n".join(steps_text)
#     return ""

# # 加载数据
# json_data = load_json_data("MMRB_data_compare.json")

# # 统计结果
# stats = {
#     "total_samples": 0,
#     "total_paths": 0,
#     "first_choice": 0,
#     "second_choice": 0,
#     "equally_good": 0,
#     "no_match_found": 0,
#     "by_subtask": defaultdict(lambda: {"total": 0, "first": 0, "second": 0, "equal": 0, "no_match": 0})
# }

# # 设置批处理大小和图像缓存
# batch_size = 1  # 每次处理一个样本
# image_cache = {}  # 缓存已处理的图像
# initial_max_new_tokens = 8192 # 初始的max_new_tokens值

# # 处理每个样本
# for sample_idx, sample in enumerate(json_data):
#     try:
#         stats["total_samples"] += 1
        
#         available_path_indices = set()
#         for path_data in sample["modified_reasoning_steps"]:
#             available_path_indices.add(path_data["path_index"])
        
#         all_image_paths_original = sample["image_paths"]
#         num_all_images = len(all_image_paths_original)
        
#         if num_all_images > 20:
#             selected_indices_float = np.linspace(0, num_all_images - 1, 20)
#             processed_indices = sorted(list(set(map(int, selected_indices_float))))
#             processed_image_paths = [all_image_paths_original[i] for i in processed_indices]
#         else:
#             processed_image_paths = all_image_paths_original

#         image_tensors = []
#         image_original_sizes = [] # To store original image dimensions
        
#         for img_path in processed_image_paths:
#             cache_key = img_path
#             if cache_key in image_cache:
#                 image_tensor, original_size = image_cache[cache_key]
#             else:
#                 image = Image.open(img_path).convert('RGB')
#                 original_size = image.size  # Get original image size

#                 image_tensors_list = process_images([image], image_processor, model.config)
                
#                 if image_tensors_list is not None and len(image_tensors_list) > 0:
#                     image_tensor = image_tensors_list[0].to(device="cuda:1", dtype=torch.float16)
#                     image_cache[cache_key] = (image_tensor, original_size)
#                 else:
#                     print(f"Warning: process_images returned None or empty list for {img_path}")
#                     continue  # Skip this image or handle error appropriately

#                 del image  # Free PIL image memory
#                 if 'image_tensors_list' in locals(): del image_tensors_list # Free tensor list memory

#             image_tensors.append(image_tensor)
#             image_original_sizes.append(original_size)  # Store the original size

#         conv_template = "qwen_1_5"
        
#         for path_idx, path_index in enumerate(sorted(available_path_indices)):
#             stats["total_paths"] += 1
#             stats["by_subtask"][sample["subtask"]]["total"] += 1

#             if path_idx > 0 and path_idx % 5 == 0:
#                 clear_gpu_memory()

#             modified_steps = get_reasoning_steps_text(sample["modified_reasoning_steps"], path_index, True)
#             raw_steps = get_reasoning_steps_text(sample["raw_reasoning_steps"], path_index, False)
            
#             image_tokens = ""
#             num_effective_images = len(processed_image_paths)
#             for i in range(num_effective_images):
#                 image_tokens += f'"image#{i}":{DEFAULT_IMAGE_TOKEN}\n'

#             critic_prompt = f"""{{"from": "human", "value": "{image_tokens}
# You are provided with an image and a question for this image. Please review the corresponding responses based on the following 5 factors:

# 1. Accuracy in Object Description: Evaluate the accuracy of the descriptions concerning the objects mentioned in the ground truth answer. Responses should minimize the mention of objects not present in the ground truth answer, and inaccuracies in the description of existing objects.

# 2. Accuracy in Depicting Relationships: Consider how accurately the relationships between objects are described compared to the ground truth answer. Rank higher the responses that least misrepresent these relationships.

# 3. Accuracy in Describing Attributes: Assess the accuracy in the depiction of objects' attributes compared to the ground truth answer. Responses should avoid inaccuracies in describing the characteristics of the objects present.

# 4. Helpfulness: Consider whether the generated text provides valuable insights, additional context, or relevant information that contributes positively to the user's comprehension of the image. Assess whether the language model accurately follows any specific instructions or guidelines provided in the prompt. Evaluate the overall contribution of the response to the user experience.

# IMPORTANT INSTRUCTION: You MUST choose either Response 1 or Response 2 as better, even if the difference is extremely subtle. "Equally good" is NOT a valid answer. If you perceive the responses as very similar in quality, you must still identify and focus on even the smallest advantages one has over the other to make your decision.

# You need to choose which response is better for the given question and provide a detailed reason.

# Your task is provided as follows:

# Question: {sample['question']}

# Options:
# {chr(10).join(sample['options'])}
# Answer:
# {sample['answer']}

# Response 1:
# {modified_steps}

# Response 2:
# {raw_steps}

# ASSISTANT:
# "}}
# ,{{"from": "gpt","value": ""}}"""
#             print(f"\n处理样本 {sample_idx + 1}/{len(json_data)}, path_index: {path_index} (子任务: {sample['subtask']})")

#             conv = copy.deepcopy(conv_templates[conv_template])
#             conv.append_message(conv.roles[0], critic_prompt)
#             conv.append_message(conv.roles[1], None)
#             prompt_question = conv.get_prompt()

#             input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to("cuda:1")

#             current_max_tokens = initial_max_new_tokens
#             found_target_phrase = False
#             attempt_count = 0
#             output_text = ""

#             while attempt_count < 3 and not found_target_phrase:
#                 if attempt_count > 0:
#                     current_max_tokens = min(current_max_tokens * 2, 32768)
#                     print(f"尝试次数 {attempt_count + 1}/3，增加 token 数至: {current_max_tokens}")
                
#                 print(f"模型生成中 (尝试 {attempt_count + 1}/3, max_new_tokens: {current_max_tokens})...")
#                 with torch.cuda.amp.autocast(dtype=torch.float16, enabled=torch.cuda.is_available()):
#                     cont = model.generate(
#                         input_ids,
#                         images=image_tensors if image_tensors else None,
#                         image_sizes=image_original_sizes if image_original_sizes else None,
#                         do_sample=False,
#                         temperature=0,
#                         max_new_tokens=current_max_tokens,
#                         use_cache=True  # Enable generation cache
#                     )

#                 text_outputs_list = tokenizer.batch_decode(cont, skip_special_tokens=True)
#                 output_text = text_outputs_list[0].strip() if text_outputs_list else ""

#                 if "The better response: [1]." in output_text or "The better response: [1]" in output_text:
#                     stats["first_choice"] += 1
#                     stats["by_subtask"][sample["subtask"]]["first"] += 1
#                     found_target_phrase = True
#                     print("模型选择: Response 1")
#                 elif "The better response: [2]." in output_text or "The better response: [2]" in output_text:
#                     stats["second_choice"] += 1
#                     stats["by_subtask"][sample["subtask"]]["second"] += 1
#                     found_target_phrase = True
#                     print("模型选择: Response 2")
#                 elif "equally good." in output_text.lower():
#                     stats["equally_good"] += 1
#                     stats["by_subtask"][sample["subtask"]]["equal"] += 1
#                     found_target_phrase = True
#                     print("模型选择: 两者一样好")
                
#                 if found_target_phrase:
#                     break 

#                 attempt_count += 1
#                 clear_gpu_memory()

#             if not found_target_phrase:
#                 stats["no_match_found"] += 1
#                 stats["by_subtask"][sample["subtask"]]["no_match"] += 1
#                 print(f"警告: 尝试3次后，在样本 {sample_idx + 1}, path_index {path_index} 的输出中未找到指定短语。")
#                 print(f"最后一次尝试的输出: {output_text[:500]}...")  # Log more of the output for debugging
#             else:
#                  print(f"最终回答 (部分): {output_text[:300]}...")

#             del input_ids
#             if 'cont' in locals() and cont is not None: del cont
#             if 'text_outputs_list' in locals() and text_outputs_list is not None: del text_outputs_list
#             clear_gpu_memory()

#         del image_tensors
#         del image_original_sizes
#         if len(image_cache) > 20:
#             image_cache.clear()
#         clear_gpu_memory()
        
#     except Exception as e:
#         print(f"处理样本 {sample_idx + 1} (ID: {sample.get('id', 'N/A')}) 时出错: {e}")
#         exc_type, exc_obj, exc_tb = sys.exc_info()
#         fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
#         print(f"错误类型: {exc_type}, 文件: {fname}, 行号: {exc_tb.tb_lineno}")
#         clear_gpu_memory()
#         continue

# # 打印并保存统计结果
# print("\n最终统计结果:")
# print(f"总样本数: {stats['total_samples']}")
# print(f"总评估路径数: {stats['total_paths']}")
# print(f"选择Response 1的次数: {stats['first_choice']}")
# print(f"选择Response 2的次数: {stats['second_choice']}")
# print(f"认为两者一样好的次数: {stats['equally_good']}")
# print(f"未匹配到指定回复的次数: {stats['no_match_found']}")

# if stats['total_paths'] > 0:
#     print(f"选择Response 1的比例: {stats['first_choice']/stats['total_paths']*100:.2f}%")
#     print(f"选择Response 2的比例: {stats['second_choice']/stats['total_paths']*100:.2f}%")
#     print(f"认为两者一样好的比例: {stats['equally_good']/stats['total_paths']*100:.2f}%")
#     print(f"未匹配到指定回复的比例: {stats['no_match_found']/stats['total_paths']*100:.2f}%")

# print("\n按子任务统计:")
# for subtask, subtask_stats in stats["by_subtask"].items():
#     print(f"\n子任务: {subtask}")
#     print(f"  总路径数: {subtask_stats['total']}")
#     print(f"  选择Response 1: {subtask_stats['first']}")
#     print(f"  选择Response 2: {subtask_stats['second']}")
#     print(f"  两者一样好: {subtask_stats['equal']}")
#     print(f"  未匹配: {subtask_stats['no_match']}")
#     if subtask_stats['total'] > 0:
#         print(f"  Response 1 比例: {subtask_stats['first']/subtask_stats['total']*100:.2f}%")
#         print(f"  Response 2 比例: {subtask_stats['second']/subtask_stats['total']*100:.2f}%")
#         print(f"  一样好比例: {subtask_stats['equal']/subtask_stats['total']*100:.2f}%")
#         print(f"  未匹配比例: {subtask_stats['no_match']/subtask_stats['total']*100:.2f}%")

# # 保存统计结果到CSV文件
# csv_file_path = "./reward_model_stats_0515_original_size.csv"
# os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

# try:
#     with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
#         writer = csv.writer(csvfile)
        
#         writer.writerow(["Overall Statistics", "Value"])
#         writer.writerow(["Total Samples Processed", stats['total_samples']])
#         writer.writerow(["Total Paths Evaluated", stats['total_paths']])
#         writer.writerow(["Times Response 1 Chosen", stats['first_choice']])
#         writer.writerow(["Times Response 2 Chosen", stats['second_choice']])
#         writer.writerow(["Times Responses Equally Good", stats['equally_good']])
#         writer.writerow(["Times No Match Found", stats['no_match_found']])
#         if stats['total_paths'] > 0:
#             writer.writerow(["Percentage Response 1 Chosen", f"{stats['first_choice']/stats['total_paths']*100:.2f}%"])
#             writer.writerow(["Percentage Response 2 Chosen", f"{stats['second_choice']/stats['total_paths']*100:.2f}%"])
#             writer.writerow(["Percentage Equally Good", f"{stats['equally_good']/stats['total_paths']*100:.2f}%"])
#             writer.writerow(["Percentage No Match Found", f"{stats['no_match_found']/stats['total_paths']*100:.2f}%"])
#         writer.writerow([])

#         writer.writerow(["Subtask", "Total Paths", "Response 1 Chosen", "Response 2 Chosen", "Equally Good", "No Match Found",
#                          "Response 1 %", "Response 2 %", "Equally Good %", "No Match %"])
        
#         for subtask, s_stats in stats["by_subtask"].items():
#             row = [
#                 subtask,
#                 s_stats['total'],
#                 s_stats['first'],
#                 s_stats['second'],
#                 s_stats['equal'],
#                 s_stats['no_match']
#             ]
#             if s_stats['total'] > 0:
#                 row.extend([
#                     f"{s_stats['first']/s_stats['total']*100:.2f}%",
#                     f"{s_stats['second']/s_stats['total']*100:.2f}%",
#                     f"{s_stats['equal']/s_stats['total']*100:.2f}%",
#                     f"{s_stats['no_match']/s_stats['total']*100:.2f}%"
#                 ])
#             else:
#                 row.extend(["N/A", "N/A", "N/A", "N/A"])
#             writer.writerow(row)
            
#     print(f"\n统计结果已保存到: {csv_file_path}")

# except IOError:
#     print(f"错误: 无法写入CSV文件到路径 {csv_file_path}。请检查权限或路径。")
# except Exception as e:
#     print(f"保存CSV时发生未知错误: {e}")

# print("\n脚本执行完毕。")
