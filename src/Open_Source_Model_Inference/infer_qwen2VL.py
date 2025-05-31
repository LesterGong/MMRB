import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import json
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import sys
import os

model_path = "./Qwen/Qwen2-VL-7B-Instruct"
data_path = "./annotation_MMRB/MMRB_data.json"
output_path = "./answer_full_qwen2VL-7B_answer.json"


model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_path)


with open(data_path, "r", encoding='utf-8') as f:
    generated_data = json.load(f)
    print("len(generated_data)", len(generated_data))

generated_answer = []
for i, data in enumerate(tqdm(generated_data)):
    try:
        image_paths = data["image_paths"]
        question = data["question"]
        options = " ".join([f"{option}" for i, option in enumerate(data["options"])])

        image_conent = []
        for path in image_paths:
            image_conent.append({"type": "image", "image": os.path.join("./annotation_MMRB", path)})

        # Answer Prompt
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

        messages = [
            {
                "role": "user",
                "content": [
                    *image_conent,
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        print(f'User: {question}\nAssistant: {output_text}')

        data["CoT_answer"] = output_text
        generated_answer.append(data)
    
    except Exception as e:
        print(i)
        print(e)
    
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(generated_answer, file, ensure_ascii=False, indent=4)





