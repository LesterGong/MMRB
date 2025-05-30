import base64
import requests
import json
from pathlib import Path
from typing import List
from openai import OpenAI
from tqdm import tqdm
import mimetypes


DATA_PATH = Path("./annotation_MMRB/MMRB_data_mini10.json")
OUTPUT_PATH = DATA_PATH.with_name("MMRB_data_mini10_cot_output.json")
MODEL_NAME = "./OpenGVLab/InternVL3-1B"
IMAGE_ROOT = "./annotation_MMRB"
MAX_TOKENS = 512

client = OpenAI(
    api_key="api_key",
    base_url="http://localhost:8000/v1",
)

def encode_base64_content_from_path(img_path: str | Path) -> str:
    img_path = Path(img_path)
    if not img_path.is_file():
        raise FileNotFoundError(img_path)

    mime, _ = mimetypes.guess_type(img_path)
    mime = mime or "image/jpeg"
    data_b64 = base64.b64encode(img_path.read_bytes()).decode()
    return f"data:{mime};base64,{data_b64}"


def build_messages(question: str, image_paths: List[str]):
    content = [{"type": "text", "text": question}]

    for rel in image_paths:
        full_path = "file://" + IMAGE_ROOT + rel[1:]
        content.append({
            "type": "image_url",
            "image_url": {"url": full_path},
        })

    return [{"role": "user", "content": content}]


def main() -> None:
    with DATA_PATH.open("r", encoding="utf-8") as f:
        samples = json.load(f)

    results = []

    for idx, sample in enumerate(tqdm(samples, desc="Inference")):
        try:
            messages = build_messages(
                question=sample["question"],
                image_paths=sample["image_paths"],
            )

            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
            )

            sample["CoT_answer"] = resp.choices[0].message.content
        except Exception as exc:
            print(f"[Sample {idx}] Error: {exc}")
            pass

        finally:
            results.append(sample)

        with OUTPUT_PATH.open("w", encoding="utf-8") as fout:
            json.dump(results, fout, ensure_ascii=False, indent=2)

    print(f"\nAll done! Answers written to {OUTPUT_PATH}\n")


if __name__ == "__main__":
    main()