import json
import os

#uniform choice format
input_paths = [
    r"./046_Visual_Grounding/046_reasoning_steps_muirbench_visualGrounding_0_50_human.json",
    r"./077_Visual_Referring/077_reasoning_steps_miBench_visualReferring_0_50_human.json",
    r"./078_Text-Rich_Images/078_reasoning_steps_miBench_textRichImages_0_50_human.json",
    r"./079_Vision-linked_Textual_Knowledge/079_reasoning_steps_miBench_visionLinkedTextualKnowledge_0_50_human.json",
    ]

for input_path in input_paths:

    base, ext = os.path.splitext(input_path)
    output_path = base + ext

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        question = item.get("question", "")
        image_paths = item.get("image_paths", [])
        options = item.get("options", [])
        
        options_str = " ".join(options) if isinstance(options, list) else ""

        item["question"] = question + " " + options_str

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ done: {output_path}")


#uniform image token
for input_path in input_paths:
    base, ext = os.path.splitext(input_path)
    output_path = base + ext

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)


    for item in data:
        question = item.get("question", "")
        image_paths = item.get("image_paths", [])
        
        img_count = question.count("<image>")
        replacements = [f"{{image#{i+1}}}" for i in range(img_count)]
        
        for i, replacement in enumerate(replacements):
            question = question.replace("<image>", replacement, 1)
        
        item["question"] = question

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"all done, save to: {output_path}")
