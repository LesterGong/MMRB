from openai import OpenAI
import base64
import os
import re
import json
from tqdm import tqdm
import time
import argparse


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
            "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"},
        }
        for image_path in image_paths
    ]

def is_valid_output(output_json):
    """检查 output_json 是否满足结构要求：3个路径，每个路径包含 rationale 字段"""
    if not isinstance(output_json, list) or len(output_json) != 3:
        return False
    for path in output_json:
        if not isinstance(path, list):
            return False
        for step in path:
            if not isinstance(step, dict) or "rationale" not in step:
                return False
    return True


user_text_template = """You are an expert in multimodal reasoning. Given a multi-image reasoning task, generate three distinct reasoning paths in JSON format, where each path follows a step-by-step Chain of Thought (CoT). 
Each reasoning step must include:
- reasoning step: The step index.
- reasoning type: Categorized as one of the following:
Task Understanding / Information Grounding / Commonsense Seeking / Logical Reasoning / Arithmetic Calculating / Draw Conclusion
- rationale: A detailed explanation of the reasoning process at each step.
The output must be a JSON list of three reasoning paths, where each path is a list of step-by-step reasoning objects like this:
[
    {{
        "reasoning step": int,
        "reasoning type": str,
        "rationale": str
    }},
    ...
]
Question: {question}
Options: [{options}]
Answer: {answer}
"""

user_text_template_2 = """Question: {question}
Options: [{options}]
Answer: {answer}
The images are uploaded in sequence corresponding to <image-N>.
Can you help me generate the intermediate reasoning steps and rationales.
And finally output a sum up of the reasoning steps and rationales like: <In order to answer the question, I need to ...>.
You need to describe the images according to the question asks.
Please output in json form like this:
{{
    "reasoning step": int,
    "reasoning type": Task Understanding / Information Grounding / Commonsense Seeking/ Logical Reasoning / Arithmetic Calculating / Draw Conclusion / Summary
    "rationale": str
}}
You should think from novel perspectives that different from the follow ones:
<Different from this-1> 
<Different from this-2> """



def process_file(input_path, output_path):
    with open(input_path, "r", encoding='utf-8') as f:
        data = json.load(f)

    if os.path.exists(output_path):
        with open(output_path, "r", encoding='utf-8') as f:
            generated_data = json.load(f)
        print("Loaded generated data:", len(generated_data))
    else:
        generated_data = []

    generated_dict = {row["index"]: row for row in generated_data}

    for i, row in enumerate(tqdm(data)):
        row_id = row["index"]

        needs_generation = False

        if row_id in generated_dict:
            existing_steps = generated_dict[row_id].get("reasoning_steps", [])
            if not is_valid_output(existing_steps):
                print(f"Row {row_id} - Invalid output found, regenerating...")
                needs_generation = True
            else:
                continue 
        else:
            needs_generation = True

        if needs_generation:
            image_paths = row["image_paths"]
            print(f"Row {i}: {image_paths}")
            image_message = build_image_messages("/", image_paths)

            user_text = user_text_template.format(
                question=row["question"],
                options=" ".join([f"{option}" for i, option in enumerate(row["options"])]),
                answer=row["answer"]
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

                    if is_valid_output(output_json):
                        break
                    else:
                        raise ValueError("Generated output is not valid")

                except Exception as e:
                    print(f"Error during generation or parsing: {e}")
                    retries += 1
                    if retries < MAX_RETRIES:
                        print(f"Retrying in {RETRY_DELAY} seconds... (Attempt {retries}/{MAX_RETRIES})")
                        time.sleep(RETRY_DELAY)
                    else:
                        raise RuntimeError("Max retries reached. Failed to generate valid output.")

            print(json.dumps(output_json, indent=4))
            row["reasoning_steps"] = output_json
            generated_dict[row_id] = row
            with open(output_path, 'w', encoding='utf-8') as file:
                json.dump(list(generated_dict.values()), file, ensure_ascii=False, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help="Input JSON file path")
    parser.add_argument('--output', type=str, help="Output JSON file path")
    args = parser.parse_args()

    process_file(args.input, args.output)


if __name__ == "__main__":
    main()



