import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import json
from tqdm import tqdm
import os


answer_type = "answer"
model_path = './openbmb/MiniCPM-V-2_6'
data_path = "./annotation_MMRB/MMRB_data.json"
output_path = f"./answer_full_minicpmV-8B_{answer_type}_2.json"

model = AutoModel.from_pretrained(model_path, 
                                  trust_remote_code=True,
                                  attn_implementation='sdpa',
                                  torch_dtype=torch.bfloat16,
                                  ) 
model = model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

existing_file_path = f"./answer_full_minicpmV-8B_{answer_type}.json"
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

        if len(image_paths) > 16:
            image_paths = image_paths[::2][:16]

        image_list = []
        for path in image_paths:
            image = Image.open(os.path.join("./annotation_MMRB", path)).convert('RGB')
            image_list.append(image)

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

        if answer_type == "pure_cot":
            question = f"{data['question']}\n" + "Please think step by step."

        msgs = [{'role': 'user', 'content': image_list + [question]}]
        response = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer
        )

        print(f'User: {question}\nAssistant: {response}')

        data["CoT_answer"] = response
        generated_answer.append(data)

    except Exception as e:
        print(i)
        print(e)
    
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(generated_answer, file, ensure_ascii=False, indent=4)
