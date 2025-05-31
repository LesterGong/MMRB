import json
from tqdm import tqdm
from collections import defaultdict


reasoning_category_map = {
    "Normal Reasoning": [
        "Fashion QA",
        "Image Retrieval",
        "Art Style",
        "Attribute Similarity",
        "Visual Analogy",
        "Image Jigsaw",
        "Face Retrieval",
        "Handwritten Text Retrieval"
    ],
    "Semantic Reasoning": [
        "Text-Rich Images",
        "Vision-linked Textual Knowledge",
        "Closed-ended VQA",
        "Open_ended_VQA",
        "Hallucination",
        "Demo-based Task Learning",
        "Visual Cloze",
        "Textual Cloze",
        "Visual Coherence",
        "Visual Ordering",
        "GUI Next Action Prediction",
        "Image-to-Image Retrieval",
        "Sketch-to-Image Retrieval",
        "Text-to-Image Retrieval",
        "Spot the Similarity",
        "Webpage QA",
        "Textbook QA",
        "Complex Multimodal QA",
        "Slide QA",
        "OCR QA",
        "Industrial Inspection",
        "Property Coherence",
        "State Transformation Coherence",
        "Multi-Image Visual Entailment",
        "Meme Comprehension",
        "Interleaved Image-Text Analysis",
        "Long Text with Images QA",
        "Visual Correspondence",
        "Semantic Correspondence",
        "Functional Correspondence",
        "Mantis-Eval",
        "Difference Spotting PPT",
        "Image-Text Matching PubMed",
        "Visual Grounding",
        "Code Understanding",
        "Sightseeing Locations",
        "Food Comparisons",
        "ArXiv Citation Look Up",
        "Attribute Matching",
        "Emoji Algebra",
        "Clocks",
        "Schedule",
        "Code Edit",
        "Isomorphism",
        "IQ",
        "Raven",
        "MathVerse",
        "SciVerse",
        "Emotion Recognition"
    ],
    "Temporal Reasoning": [
        "Counterfactual Inference",
        "Next Image Prediction",
        "Single Object Tracking",
        "Multi-View Action Recognition",
        "Egocentric Video Question Answering",
        "Person Re-Identification",
        "Vehicle Re-Identification"
    ],
    "Spatial Reasoning": [
        "Geographic Understanding",
        "Func Read",
        "Geom Shape",
        "Geom Cost",
        "Collisions",
        "Maps",
        "Visual Referring"
    ],
    "Spatial & Semantic Reasoning": [
        "Image-Set QA",
        "Multi-view Reasoning",
        "Visual Retrieval",
        "3D Scene Counting",
        "Plot Code Understanding"
    ],
    "Temporal & Semantic Reasoning": [
        "Comic Panel Identification",
        "Difference Spotting",
        "Global Video Understanding",
        "Action Recognition",
        "Action Localization",
        "Action Prediction",
        "Scene Transition",
        "Ordering",
        "Meme Video Understanding",
        "Temporal Localization",
        "Temporal Ordering",
        "Casuality Reasoning"
    ],
    "All-Round Reasoning": [
        "Moving Attribute",
        "Object Shuffle",
        "Egocentric Navigation",
        "Moving Direction"
    ]
}



subtask_to_category = {
    subtask: category
    for category, subtasks in reasoning_category_map.items()
    for subtask in subtasks
}


import os
import glob
import json


folder_path = "./MMRB"
json_files = glob.glob(os.path.join(folder_path, "step*.json"))

for file_path in json_files:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        overall_scores = []
        category_scores = defaultdict(list)
        for row in tqdm(data):
            all_step_correctness = row.get("all_step_correctness", [])
            subtask = row.get("subtask", "unknown")
            category = subtask_to_category.get(subtask, "Uncategorized")
            
            max_score = 0.0
            for reasoning_path in all_step_correctness:
                if not isinstance(reasoning_path, list) or len(reasoning_path) <= 2:
                    continue

                total_steps = len(reasoning_path) - 2
                correct_steps = sum(1 for step in reasoning_path[1:-1] if isinstance(step, dict) and step.get("correctness") == True)
                
                score = correct_steps / total_steps if total_steps > 0 else 0
                max_score = max(max_score, score)
            
            overall_scores.append(max_score)
            category_scores[category].append(max_score)

        average_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0
        print(f"\n[Overall] Average Max Correctness Score: {average_score * 100:.2f}%")

        print("\n[Per Category] Average Max Correctness Scores:")
        for category, scores in category_scores.items():
            avg = sum(scores) / len(scores) if scores else 0
            print(f"- {category}: {avg * 100:.2f}%")




    except Exception as e:
        print(f"read file  {file_path} raise error: {e}")



