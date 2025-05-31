import soundfile as sf
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import json
from tqdm import tqdm
import os


USE_AUDIO_IN_VIDEO = True
model_path = "./Qwen/Qwen2.5-Omni-3B"
data_path = "./annotation_MMRB/MMRB_data.json"
output_path = "./answer_full_qwen25O-3B_answer.json"

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)
model.disable_talker()
processor = Qwen2_5OmniProcessor.from_pretrained(model_path)


with open(data_path, "r", encoding='utf-8') as f:
    generated_data = json.load(f)
    print("len(generated_data)", len(generated_data))

generated_answer = []
for i, data in enumerate(tqdm(generated_data)):
    try:
        image_paths = data["image_paths"]
        question = data["question"]
        options = " ".join([f"{option}" for i, option in enumerate(data["options"])]),

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


        # CoT Prompt

        image_conent = []
        for path in image_paths:
            image_conent.append({"type": "image", "image": os.path.join("./annotation_MMRB", path)})
        
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ],
            },
            {
                "role": "user",
                "content": [
                    *image_conent, {"type": "text", "text": question},
                ],
            },
        ]

        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = inputs.to(model.device).to(model.dtype)

        text_ids = model.generate(**inputs, return_audio=False, use_audio_in_video=USE_AUDIO_IN_VIDEO)

        text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        response = text[0].split("assistant\n")[-1]

        print(f'User: {question}\nAssistant: {response}')

        data["CoT_answer"] = response
        generated_answer.append(data)

    except Exception as e:
        print(i)
        print(e)
    
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(generated_answer, file, ensure_ascii=False, indent=4)

