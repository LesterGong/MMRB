import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import json
from tqdm import tqdm
import os
import math
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig

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
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
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
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL3-1B': 24, 'InternVL3-2B': 24, 'InternVL3-8B': 32, 'InternVL3-9B': 48,
        'InternVL3-14B': 48, 'InternVL3-38B': 64, 'InternVL2_5-78B': 80}[model_name]
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

answer_type = "cot"
model_size = "9"
model_path = f"./models/OpenGVLab/InternVL3-{model_size}B"
data_path = "./annotation_MMRB/MMRB_data.json"
output_path = f"./answer_full_internVL3-{model_size}B_{answer_type}_2.json"


model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    ).eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
generation_config = dict(max_new_tokens=2048, do_sample=True)



existing_file_path = f"./answer_full_internVL3-{model_size}B_{answer_type}.json"
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
        question = data["question"]
        options = " ".join([f"{option}" for i, option in enumerate(data["options"])]),

        pixel_values = []
        num_patches_list = []
        for path in image_paths:
            pixel_value = load_image(os.path.join("./annotation_MMRB", path), max_num=6).to(torch.bfloat16).cuda()
            pixel_values.append(pixel_value)
            num_patches_list.append(pixel_value.size(0))
        pixel_values = torch.cat(pixel_values, dim=0)


        # CoT + Answer Prompt
        if answer_type == "cot":
            question_type = data['question_type']
            if question_type == 'multi-choice':
                question = (
                    f"{data['question']}\n"
                    "Please think step by step.\n"
                    "Then write the option letter in the format: Answer[<letter>]."
                    # "Then write your final answer (only option letter) in the format: Answer[<letter>]."
                )
            else: 
                question = (
                    f"{data['question']}\n"
                    "Please think step by step.\n"
                    "Then write your final answer in the format: Answer[<your_answer_here>]."
                )

        # Answer Prompt
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


        # CoT Prompt
        if answer_type == "pure_cot":
            question = f"{data['question']}\n" + "Please think step by step."


        question = "<image>"*len(image_paths) +  question
        response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                                    num_patches_list=num_patches_list,
                                    history=None, return_history=True)
        print(f'User: {question}\nAssistant: {response}')

        data["CoT_answer"] = response
        generated_answer.append(data)

    except Exception as e:
        print(i)
        print(e)
    
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(generated_answer, file, ensure_ascii=False, indent=4)





