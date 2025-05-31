from openai import OpenAI
import base64
import re
import json
from tqdm import tqdm
import time



def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

client = OpenAI(
    base_url='your_base_url', 
    api_key='your_api_key' 
)

def extract_json_from_string(s):

    pattern = r'```json(.*?)```'
    matches = re.findall(pattern, s, re.DOTALL)
    json_objects = []
    for match in matches:
        try:
            json_str = match.strip()
            json_obj = json.loads(json_str)
            json_objects.append(json_obj)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
    
    return json_objects

def build_image_messages(path_base, image_paths):
    return [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encode_image(path_base + image_path)}"},
        }
        for image_path in image_paths
    ]


user_text_template = """Please first think step by step, and then output the final answer.
Note that directly getting the answer from the image should not in reasoning steps.
The output must be in JSON format like this:
{{
    "reasoning steps": [<each step as a JSON object>],
    "Answer": "<final answer>"
}}

Question: {question}
"""


input_path = "./multi_image_out_domain.json"
with open(input_path, "r", encoding='utf-8') as f:
    data = json.load(f)

generated_data = []

for i, row in enumerate(tqdm(data)):
    sample_id = row["sample_id"]
    sub_task = row["sub_task"]
    question = row["conversations"][0]["value"]
    asnwer = row["conversations"][1]["value"]
    images = row["image"]

    if sub_task not in ["Forensic Detection", "Visual Similarity"]:
        continue

    image_paths = images
    print(f"Row {i}: {image_paths}")
    image_message = build_image_messages("./eval_images_fix/", image_paths)
    user_text = user_text_template.format(
        question=question,
    )

    MAX_RETRIES = 5
    RETRY_DELAY = 2
    retries = 0

    while retries < MAX_RETRIES:
        try:
            start_time = time.time()
            completion = client.chat.completions.create(
                model="gpt-4o-2024-11-20",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user",
                        "content": [
                            {"type": "text", "text": user_text},
                            *image_message
                        ]}
                ],
                temperature=1,
            )
            elapsed_time = time.time() - start_time
            print(f"Inference time: {elapsed_time:.2f} s")
            print(completion.usage)

            try:
                output_json = extract_json_from_string(completion.choices[0].message.content)[0]
            except:
                output_json = json.loads(completion.choices[0].message.content)

            break

        except Exception as e:
            print(f"Error during generation or parsing: {e}")
            retries += 1
            if retries < MAX_RETRIES:
                print(f"Retrying in {RETRY_DELAY} seconds... (Attempt {retries}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY)
            else:
                raise RuntimeError("Max retries reached. Failed to generate or parse output.")

    print(json.dumps(output_json, indent=4))
    row["evaluate_answer"] = output_json
    generated_data.append(row)

    output_path = "./CoT_evaluate_test.json"
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(generated_data, file, ensure_ascii=False, indent=4)
    





