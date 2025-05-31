from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
from typing import List
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_data():
    with open("reasoning_steps_mantis_multi_0_217_chinese.json", "r", encoding="utf-8") as f:
        return json.load(f)

data = load_data()

class ReasoningStep(BaseModel):
    reasoning_step: int
    reasoning_type: str
    rationale: str
    rationale_chinese: str

class Annotation(BaseModel):
    index: int
    reasoning_steps: List[List[ReasoningStep]] 

@app.get("/data")
def get_data():
    return data

@app.post("/annotate")
def save_annotation(annotation: Annotation):
    if annotation.index >= len(data):
        raise HTTPException(status_code=400, detail="Index out of range")
    
    item = data[annotation.index]
    
    if "reasoning_steps" not in item or not isinstance(item["reasoning_steps"], list):
        item["reasoning_steps"] = [[]]

    while len(item["reasoning_steps"]) < len(annotation.reasoning_steps):
        item["reasoning_steps"].append([])

    for i in range(len(annotation.reasoning_steps)):
        while len(item["reasoning_steps"][i]) < len(annotation.reasoning_steps[i]):
            item["reasoning_steps"][i].append({
                "reasoning step": len(item["reasoning_steps"][i]) + 1,
                "reasoning type": "",
                "rationale": "",
                "rationale_chinese": ""
            })
    for i in range(len(annotation.reasoning_steps)):
        for j in range(len(annotation.reasoning_steps[i])):
            item["reasoning_steps"][i][j] = annotation.reasoning_steps[i][j].dict()

    with open("reasoning_steps_mantis_multi_0_217_human.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    return {"message": "Annotation saved successfully"}




# uvicorn annotation_backend:app --host 0.0.0.0 --port 9000 --reload
# streamlit run annotation_frontend_0405.py