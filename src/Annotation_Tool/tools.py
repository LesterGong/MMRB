import os
import shutil
import json
import tqdm
import glob
import pandas as pd
import numpy as np
from pathlib import Path 

def copy_human_json_files(src_root, dst_folder):
    # Copies all JSON files ending with "_human.json" from src_root to dst_folder.
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    copied_files = []

    for dirpath, dirnames, filenames in os.walk(src_root):
        for filename in filenames:
            if filename.endswith("_human.json"):
                src_file_path = os.path.join(dirpath, filename)
                dst_file_path = os.path.join(dst_folder, filename)

                shutil.copy2(src_file_path, dst_file_path)
                copied_files.append(dst_file_path)

    print(f"total copy{len(copied_files)} _human.json to {dst_folder}")

def copy_images_from_json(json_file_path, prefix_paths):
    # Copies images referenced in a JSON file to a new images subdirectory.
    json_path = Path(json_file_path)
    base_dir = json_path.parent
    output_dir = base_dir / 'images'
    output_dir.mkdir(exist_ok=True)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        image_paths = item.get('image_paths', [])
        for rel_path in image_paths:
            found = False
            for prefix in prefix_paths:
                full_path = Path(prefix) / rel_path
                if full_path.exists():
                    shutil.copy(full_path, output_dir)
                    found = True
                    break
            if not found:
                print(f"[{json_path.name}] Image not found: {rel_path}")

    print(f"[{json_path.name}] Images copied to: {output_dir}")

def process_all_annotations(annotation_root, prefix_paths):
    # Processes all annotation JSON files.
    annotation_dir = Path(annotation_root)

    for subdir in annotation_dir.iterdir():
        if subdir.is_dir():
            json_files = list(subdir.glob("*.json"))
            if len(json_files) == 1:
                print(f"Processing subtask: {subdir.name}")
                copy_images_from_json(json_files[0], prefix_paths)
            elif len(json_files) == 0:
                print(f"[Skipping] No JSON files in {subdir.name}")
            else:
                print(f"[Warning] Multiple JSON files in {subdir.name}, cannot determine which one to use")

def calculate_average_annotation_time(input_path: str) -> float:
    # Calculates the average annotation time.

    try:
        with open(input_path, "r", encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{input_path}' was not found.")
        return 0.0
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_path}'. Please ensure it's a valid JSON file.")
        return 0.0

    total_time = 0
    count = 0

    for row in tqdm(data, desc=f"Processing {input_path}"):
        annotation_time = row.get("annotation_time", 0)
        total_time += annotation_time
        count += 1

    average_time = total_time / count if count > 0 else 0
    print(f"Average time cost for '{input_path}': {average_time:.2f} seconds")
    return average_time

def save_human_json_paths(root_folder, output_filename="reasoning_path_human.txt"):
    # Saves the relative paths of all "_human.json" files to a text file.
    human_files = []

    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith("_human.json"):
                relative_path = "./annotation_MMRB/" + os.path.relpath(os.path.join(dirpath, filename), root_folder).replace("\\", "/")
                human_files.append(relative_path)

    output_path = os.path.join(os.getcwd(), output_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        for path in human_files:
            f.write(path + "\n")

    print(f"{len(human_files)}files have been saved to {output_path}")


def validate_json_annotation_files(file_paths: list[str]) -> dict:
    # Validates the structure and content of a list of JSON annotation files.
    validation_results = {
        "total_files_checked": len(file_paths),
        "valid_files": [],
        "invalid_files": [],
        "issues_log": []
    }

    print(f"Total number of annotation files to validate: {len(file_paths)}")

    for filename in file_paths:
        file_has_issues = False
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data_list = json.load(f)

            
            if len(data_list) == 50:
                print(f"  {len(data_list)} entries: {filename} (Matches expected 50)")
            else:
                print(f"  {len(data_list)} entries: {filename} (DOES NOT match expected 50)")
                validation_results["issues_log"].append(f"File '{filename}' has {len(data_list)} entries; expected 50.")
                file_has_issues = True

            for i, data_item in enumerate(data_list):
                item_index = data_item.get("index", f"unknown_index_at_position_{i}")
                if "reasoning_steps" in data_item:
                    reasoning_steps = data_item["reasoning_steps"]
                    if len(reasoning_steps) != 3:
                        print(f"    Item {item_index}: Invalid 'reasoning_steps' length {len(reasoning_steps)} (Expected 3) in {filename}")
                        validation_results["issues_log"].append(
                            f"Item {item_index} in '{filename}': 'reasoning_steps' has length {len(reasoning_steps)}; expected 3."
                        )
                        file_has_issues = True
                else:
                    print(f"    Item {item_index}: Key 'reasoning_steps' not found in {filename}")
                    validation_results["issues_log"].append(
                        f"Item {item_index} in '{filename}': 'reasoning_steps' key not found."
                    )
                    file_has_issues = True

            if file_has_issues:
                validation_results["invalid_files"].append(filename)
            else:
                validation_results["valid_files"].append(filename)

        except FileNotFoundError:
            print(f"Error: File not found - {filename}")
            validation_results["issues_log"].append(f"Error: File not found - '{filename}'.")
            validation_results["invalid_files"].append(filename)
            file_has_issues = True
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format - {filename}")
            validation_results["issues_log"].append(f"Error: Invalid JSON format - '{filename}'.")
            validation_results["invalid_files"].append(filename)
            file_has_issues = True
        except Exception as e:
            print(f"An unexpected error occurred with {filename}: {e}")
            validation_results["issues_log"].append(f"An unexpected error occurred with '{filename}': {e}.")
            validation_results["invalid_files"].append(filename)
            file_has_issues = True
            
    return validation_results

def update_image_paths_in_json(json_file_path):
    # Updates image paths 
    json_path = Path(json_file_path)

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}. Please check the file's content.")
        return

    modified = False

    for item in data:
        if 'image_paths' in item:
            new_paths = []
            for path in item['image_paths']:
                filename = Path(path).name
                new_paths.append(f"./images/{filename}")
            item['image_paths'] = new_paths
            modified = True

    if modified:
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Updated: {json_path}")
        except IOError as e:
            print(f"Error writing to file {json_path}: {e}")
    else:
        print(f"Not updated (no 'image_paths' field found): {json_path}")

def process_all_annotations(annotation_root):
    # Processes all JSON annotation files in subdirectories, updating image paths.
    annotation_dir = Path(annotation_root)
    if not annotation_dir.is_dir():
        print(f"Error: Annotation root '{annotation_root}' is not a valid directory.")
        return
    for subdir in annotation_dir.iterdir():
        if subdir.is_dir():
            json_files = list(subdir.glob("*.json"))
            if len(json_files) == 1:
                update_image_paths_in_json(json_files[0])
            elif len(json_files) == 0:
                print(f"[Skipping] No JSON files found in {subdir.name}")
            else:
                print(f"[Warning] Multiple JSON files found in {subdir.name}, cannot determine which one to use.")