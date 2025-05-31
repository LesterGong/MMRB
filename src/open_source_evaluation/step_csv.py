import json
import os
import pandas as pd
from tqdm import tqdm
from collections import defaultdict


reasoning_category_map = {
    "Normal Reasoning": [
        "Fashion QA", "Image Retrieval", "Art Style", "Attribute Similarity",
        "Visual Analogy", "Image Jigsaw", "Face Retrieval",
        "Handwritten Text Retrieval"
    ],
    "Semantic Reasoning": [
        "Text-Rich Images", "Vision-linked Textual Knowledge",
        "Closed-ended VQA", "Open_ended_VQA", "Hallucination",
        "Demo-based Task Learning", "Visual Cloze", "Textual Cloze",
        "Visual Coherence", "Visual Ordering", "GUI Next Action Prediction",
        "Image-to-Image Retrieval", "Sketch-to-Image Retrieval",
        "Text-to-Image Retrieval", "Spot the Similarity", "Webpage QA",
        "Textbook QA", "Complex Multimodal QA", "Slide QA", "OCR QA",
        "Industrial Inspection", "Property Coherence",
        "State Transformation Coherence", "Multi-Image Visual Entailment",
        "Meme Comprehension", "Interleaved Image-Text Analysis",
        "Long Text with Images QA", "Visual Correspondence",
        "Semantic Correspondence", "Functional Correspondence",
        "Mantis-Eval", "Difference Spotting PPT",
        "Image-Text Matching PubMed", "Visual Grounding", "Code Understanding",
        "Sightseeing Locations", "Food Comparisons", "ArXiv Citation Look Up",
        "Attribute Matching", "Emoji Algebra", "Clocks", "Schedule",
        "Code Edit", "Isomorphism", "IQ", "Raven", "MathVerse",
        "SciVerse", "Emotion Recognition"
    ],
    "Temporal Reasoning": [
        "Counterfactual Inference", "Next Image Prediction",
        "Single Object Tracking", "Multi-View Action Recognition",
        "Egocentric Video Question Answering", "Person Re-Identification",
        "Vehicle Re-Identification"
    ],
    "Spatial Reasoning": [
        "Geographic Understanding", "Func Read", "Geom Shape", "Geom Cost",
        "Collisions", "Maps", "Visual Referring"
    ],
    "Spatial & Semantic Reasoning": [
        "Image-Set QA", "Multi-view Reasoning", "Visual Retrieval",
        "3D Scene Counting", "Plot Code Understanding"
    ],
    "Temporal & Semantic Reasoning": [
        "Comic Panel Identification", "Difference Spotting",
        "Global Video Understanding", "Action Recognition",
        "Action Localization", "Action Prediction", "Scene Transition",
        "Ordering", "Meme Video Understanding", "Temporal Localization",
        "Temporal Ordering", "Casuality Reasoning"
    ],
    "All-Round Reasoning": [
        "Moving Attribute", "Object Shuffle", "Egocentric Navigation",
        "Moving Direction"
    ]
}

subtask_to_category = {
    subtask: category
    for category, subtasks in reasoning_category_map.items()
    for subtask in subtasks
}

category_order = list(reasoning_category_map.keys())

VLM_answer_paths_list = [
    "./step_results_mini10_gpt-4o-2024-11-20.json",
    "./step_results_mini10_gpt-4o-mini-2024-07-18.json",
    "./step_results_o1-2024-12-17.json",
    "./step_results_qwen25VL_3B.json",
    "./step_results_qwen25VL_7B.json",
    "./step_results_qwen25VL_32B.json",
    "./step_results_internVL2_5-8B.json",
    "./step_results_internVL2_5-26B.json",
    "./step_results_internVL3-1B.json",
    "./step_results_internVL3-9B.json",
    "./step_results_llavaOV_05B.json",
    "./step_results_llavaOV_7B.json"
]

def compute_category_averages(json_path: str):
    category_scores = defaultdict(list)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for row in data:
        all_step_correctness = row.get("all_step_correctness", [])
        subtask = row.get("subtask", "unknown")
        category = subtask_to_category.get(subtask, "Uncategorized")

        max_score = 0.0
        for reasoning_path in all_step_correctness:
            if not (isinstance(reasoning_path, list) and len(reasoning_path) > 2):
                continue
            total_steps = len(reasoning_path) - 2  # 去掉首尾
            correct_steps = sum(
                1 for step in reasoning_path[1:-1]
                if isinstance(step, dict) and step.get("correctness") is True
            )
            score = correct_steps / total_steps if total_steps > 0 else 0.0
            max_score = max(max_score, score)

        category_scores[category].append(max_score)

    return {
        cat: (sum(category_scores[cat]) / len(category_scores[cat]) * 100
              if category_scores[cat] else 0.0)
        for cat in category_order
    }

rows = []
for path in tqdm(VLM_answer_paths_list, desc="Processing"):
    if not os.path.exists(path):
        print(f"[Warning] File not found, skip: {path}")
        averages = {cat: 0.0 for cat in category_order}
    else:
        averages = compute_category_averages(path)

    row = {"Model_File": os.path.basename(path)}
    row.update(averages)
    rows.append(row)

df = pd.DataFrame(rows)
df = df[["Model_File"] + category_order]   
csv_out = "category_scores.csv"
df.to_csv(csv_out, index=False)
print(f"\n✅ Results saved to {csv_out}")
