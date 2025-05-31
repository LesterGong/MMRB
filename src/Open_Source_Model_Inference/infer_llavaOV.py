import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import json
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import sys
import os
from transformers import pipeline

answer_type = "cot"
model_path = f"llava-hf/llava-onevision-qwen2-7b-ov-hf"
data_path = "./annotation_MMRB/MMRB_data.json"
output_path = f"./answer_full_llavaOV_7B_{answer_type}_2.json"
max_new_tokens = 2048


pipe = pipeline("image-text-to-text", model=model_path)

existing_file_path = f"./answer_full_llavaOV_7B_{answer_type}.json"
with open(existing_file_path, "r", encoding='utf-8') as f:
    existing_data = json.load(f)
    print("len(existing_data)", len(existing_data))
existing_task_id = []
for i, data in enumerate(existing_data):
    subtask = data["subtask"]
    index = data["index"]
    task_id = (subtask, index)
    existing_task_id.append(task_id)
print("len(existing_task_id)", len(existing_task_id))

with open(data_path, "r", encoding='utf-8') as f:
    generated_data = json.load(f)
    print("len(generated_data)", len(generated_data))

generated_answer = []
for i, data in enumerate(tqdm(generated_data)):
    subtask = data["subtask"]
    index = data["index"]
    task_id = (subtask, index)
    if task_id in existing_task_id:
        continue

    try:
        image_paths = data["image_paths"]
        if len(image_paths) > 16:
            image_paths = image_paths[::2][:16]

        question = data["question"]
        options = " ".join([f"{option}" for i, option in enumerate(data["options"])])

        image_conent = []
        for path in image_paths:
            image_conent.append({"type": "image", "url": os.path.join("./annotation_MMRB", path)})

        if answer_type == "cot":
            question_type = data['question_type']
            if question_type == 'multi-choice':
                question = (
                    f"{data['question']}\n"
                    "Please think step by step.\n"
                    "Then write the option letter in the format: Answer[<letter>]."
                )
            else:  
                question = (
                    f"{data['question']}\n"
                    "Please think step by step.\n"
                    "Then write your final answer in the format: Answer[<your_answer_here>]."
                )

        if answer_type == "answer":
            question_type = data['question_type']
            if question_type == 'multi-choice':
                question = (
                    f"{data['question']}\n"
                    "Do not include any explanation. Only output your final answer in the exact format: Answer[<letter>].\n"
                )
            else:  
                question = (
                    f"{data['question']}\n"
                    "Do not include any explanation. Only output your final answer in the exact format: Answer[<your_answer_here>].\n"
                )
        if answer_type == "pure_cot":
            question = f"{data['question']}\n" + "Please think step by step."


        messages = [
            {
                "role": "user",
                "content": [
                    *image_conent,
                    {"type": "text", "text": question},
                ],
            }
        ]

        out = pipe(text=messages, max_new_tokens=max_new_tokens)
        output_text = out[0]["generated_text"][-1]['content']
        print(f'User: {question}\nAssistant: {output_text}')

        data["CoT_answer"] = output_text
        generated_answer.append(data)

    except Exception as e:
        print(i)
        print(e)

    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(generated_answer, file, ensure_ascii=False, indent=4)





