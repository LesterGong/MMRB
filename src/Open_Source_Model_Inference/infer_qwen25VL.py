import torch
import json
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os

answer_type = "cot"
model_path = "./Qwen/Qwen2.5-VL-32B-Instruct"
data_path = "./annotation_MMRB/MMRB_data.json"
output_path = f"./answer_full_qwen25VL-32B_{answer_type}_2.json"


model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_path)


existing_file_path = f"./answer_full_qwen25VL-32B_{answer_type}.json"
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
        options = " ".join([f"{option}" for i, option in enumerate(data["options"])])

        image_conent = []
        for path in image_paths:
            image_conent.append({"type": "image", "image": os.path.join("./annotation_MMRB", path)})

        if answer_type == "cot":
            question_type = data['question_type']
            if question_type == 'multi-choice':
                question = (
                    f"{data['question']}\n"
                    "Please think step by step.\n"
                    "Then write the option letter in the format: Answer[<letter>]. Do not ask me anything."
                )
            else: 
                question = (
                    f"{data['question']}\n"
                    "Please think step by step.\n"
                    "Then write your final answer in the format: Answer[<your_answer_here>]. Do not ask me anything."
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





