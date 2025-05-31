import transformers
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import argparse


PROMPT = """Generate an image description based on the question.
Then, provide a rationale to analyze the question.
Next, generate a step-by-step reasoning process to solve the problem. Ensure the steps are logical and concise.
Finally, provide a concise summary of the final answer in the following format: 'The final answer is: xxx'. If the question is multiple-choice, provide the options along with their content. If it is free-form, directly present the final result. Do not provide any explanation.

Format your response with the following sections, separated by ###:
### Image Description:
### Rationales:
### Let's think step by step.
### Step 1:
### Step 2:
...
### The final answer is: 

{question}"""

def mllama_infer(model_path, question, img_path, only_output_final_answer=False):
    model = MllamaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map='auto',
            )
    processor = AutoProcessor.from_pretrained(model_path)

    prompt = PROMPT.format(question=question)
    image = Image.open(img_path)
    messages = [
        {'role': 'user', 'content': [
            {'type': 'image'},
            {'type': 'text', 'text': prompt}
        ]}
    ]
    kwargs = dict(max_new_tokens=1024, do_sample=True, temperature=0.6, top_p=0.9)

    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(image, input_text, return_tensors='pt').to('cuda')
    answer = model.generate(**inputs, **kwargs)
    answer = processor.decode(answer[0][inputs['input_ids'].shape[1]:]).replace('<|eot_id|>', '')

    if only_output_final_answer:
        if len(answer.split('### The final answer is:')) == 2:
            answer = answer.split('### The final answer is:')[-1].strip()
            return answer
    else:
        return answer

def qwen_infer(model_path, question, img_path, only_output_final_answer=False):
    min_pixels = 256*28*28
    max_pixels = 1280*28*28
    processor = Qwen2VLProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map='auto', attn_implementation='flash_attention_2'
    )

    messages = [
        {
            'role': "system",
            "content": 'You are a helpful assistant.'
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img_path,
                },
                {"type": "text", "text": PROMPT.format(question=question)},
            ],
        },
    ]
    texts = [
        processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ]
    image = Image.open(img_path)

    inputs = processor(
        text=texts,
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to("cuda")


    generate_kwargs = dict(
        max_new_tokens=1024,
        top_p=0.001,
        top_k=1,
        temperature=1.0,
        repetition_penalty=1.0,
    )

    generated_ids = model.generate(
        **inputs,
        **generate_kwargs,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    out = processor.tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    answer = out[0]

    if only_output_final_answer:
        if len(answer.split('### The final answer is:')) == 2:
            answer = answer.split('### The final answer is:')[-1].strip()
            return answer
    else:
        return answer


def llava_infer(model_path, question, img_path, only_output_final_answer=False):

    def output_process(answer):
        if "<s>" in answer:
            answer = answer.replace("<s>", "").strip()
        if "[/INST]" in answer:
            answer = answer.split("[/INST]")[1].strip()
        elif "ASSISTANT:" in answer:
            answer = answer.split("ASSISTANT:")[1].strip()
        elif "assistant\n" in answer:
            answer = answer.split("assistant\n")[1].strip()
        elif "<|end_header_id|>\n\n" in answer:
            answer = answer.split("<|end_header_id|>\n\n")[2].strip()

        if "</s>" in answer:
            answer = answer.split("</s>")[0].strip()
        elif "<|im_end|>" in answer:
            answer = answer.split("<|im_end|>")[0].strip()
        elif "<|eot_id|>" in answer:
            answer = answer.split("<|eot_id|>")[0].strip()
        return answer


    processor = LlavaNextProcessor.from_pretrained(model_path)

    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_flash_attention_2=True,
    )

    model = model.eval().cuda()

    kwargs = dict(
        do_sample=False, temperature=0.9, max_new_tokens=1024, top_p=None, num_beams=1, repetition_penalty=1.0
    )

    images = [Image.open(img_path).convert("RGB")]
    
    content = [
        {"type": 'text', "text":PROMPT.format(question=question)},
        {"type": "image"}
        ]

    conversation = [
        {
            "role": "user",
            "content": content,
        }
    ]

    prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True
    )
    inputs = processor(prompt, images, return_tensors="pt").to(
        "cuda", torch.float16
    )

    output = model.generate(**inputs, **kwargs)
    answer = processor.decode(output[0], skip_special_token=True)
    answer = output_process(answer)

    if only_output_final_answer:
        if len(answer.split('### The final answer is:')) == 2:
            answer = answer.split('### The final answer is:')[-1].strip()
            return answer
    else:
        return answer

model_path = "./Mulberry_llava_8b"
question = "Describe the image"
img_path = "./annotation_MMRB/001_Comic_Panel_Identification/images/0.jpg"
answer = llava_infer(model_path, question, img_path)

print(answer)