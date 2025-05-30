import streamlit as st
import json
import time
import os

INPUT_FILE = "DATA FILE PATH"
OUTPUT_FILE = "OUTPUT FILE PATH"
PROGRESS_FILE = "annotation_progress.json" 

st.set_page_config(
    page_title="Annotation Tool",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    .stButton button {width: 100%; padding: 0.5rem;}
    .status-badge {
        border-radius: 0.5rem;
        padding: 0.3rem 0.5rem;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .completed {background-color: #d1f0d1; color: #0a6e0a;}
    .pending {background-color: #ffecb3; color: #805b00;}
    .timer {color: #ff6b6b; font-weight: bold;}
    .question-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .options-container {
        background-color: #e9ecef;
        padding: 0.7rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .answer-container {
        background-color: #d8e2dc;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .progress-bar {
        height: 0.5rem;
        background-color: #e9ecef;
        border-radius: 0.25rem;
        margin-bottom: 0.5rem;
    }
    .progress-bar-inner {
        height: 100%;
        background-color: #4CAF50;
        border-radius: 0.25rem;
    }
    .nav-buttons {margin-top: 1rem;}
    .path-header {
        background-color: #f1f8ff;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-bottom: 0.5rem;
    }
    .step-container {
        border-left: 3px solid #90e0ef;
        padding-left: 0.5rem;
        margin-bottom: 0.8rem;
    }
    .similarity-section {
        background-color: #f8f5ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .jump-container {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def load_data():
    if "data" not in st.session_state:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            original_data = json.load(f)
        if os.path.exists(OUTPUT_FILE):
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                edited_data = json.load(f)
            st.session_state.data = edited_data  
        else:
            st.session_state.data = original_data 
            
    return st.session_state.data

def load_progress():
    if "progress" not in st.session_state:
        st.session_state.progress = {}
        if os.path.exists(PROGRESS_FILE):
            try:
                with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                    st.session_state.progress = json.load(f)
            except:
                pass
    return st.session_state.progress

data = load_data()
progress = load_progress()

if "index" not in st.session_state:
    st.session_state.index = 0
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()
if "elapsed_time" not in st.session_state:
    st.session_state.elapsed_time = 0
if "jump_to" not in st.session_state:
    st.session_state.jump_to = ""

item = data[st.session_state.index]

def update_timer():
    st.session_state.elapsed_time = time.time() - st.session_state.start_time

st.markdown("<h1 style='text-align: center; margin-bottom: 1rem;'>annotation tool</h1>", unsafe_allow_html=True)

completed_count = sum(1 for k, v in st.session_state.progress.items() if v)
progress_percentage = completed_count / len(data) * 100

st.markdown(f"""
<div style='display: flex; justify-content: space-between; margin-bottom: 0.2rem;'>
    <span>total progress</span>
    <span>{completed_count} / {len(data)} ({progress_percentage:.1f}%)</span>
</div>
<div class='progress-bar'>
    <div class='progress-bar-inner' style='width: {progress_percentage}%;'></div>
</div>
""", unsafe_allow_html=True)


update_timer()
current_completed = str(st.session_state.index) in st.session_state.progress and st.session_state.progress[str(st.session_state.index)]
status_badge = "completed" if current_completed else "pending"
status_text = "completed" if current_completed else "not annotated"

st.markdown(f"""
<div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;'>
    <div class='status-badge {status_badge}'>{status_text}</div>
    <div class='timer'>time: {st.session_state.elapsed_time:.1f} s</div>
    <div>current question: {st.session_state.index + 1} / {len(data)}</div>
</div>
""", unsafe_allow_html=True)


col_prev, col_jump, col_next = st.columns([1, 3, 1])

with col_prev:
    def go_previous():
        if st.session_state.index > 0:
            st.session_state.index -= 1
            st.session_state.start_time = time.time() 
    st.button("⬅️ previous question", on_click=go_previous)

with col_jump:
    jump_col1, jump_col2 = st.columns([4, 1])
    with jump_col1:
        jump_to = st.number_input("jump to question number:", min_value=1, max_value=len(data),
                                value=st.session_state.index+1, step=1, format="%d",
                                label_visibility="collapsed")
    
    def jump_to_question():
        if jump_to and 1 <= jump_to <= len(data):
            st.session_state.index = jump_to - 1  
            st.session_state.start_time = time.time() 
    
    with jump_col2:
        st.button("jump", on_click=jump_to_question)

with col_next:
    def go_next():
        if st.session_state.index < len(data) - 1:
            st.session_state.index += 1
            st.session_state.start_time = time.time()  
    st.button("next question ➡️", on_click=go_next)


col1, col2 = st.columns([1.2, 1.8]) 


with col1:
    st.markdown("<h3>question and options</h3>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='question-container'>
        <h4>question:</h4>
        <p>{item["question"]}</p>
    </div>
    """, unsafe_allow_html=True)

    options_str = " | ".join(item["options"])
    st.markdown(f"""
    <div class='options-container'>
        <h4>options:</h4>
        <p>{options_str}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class='answer-container'>
        <h4>answer:</h4>
        <p>{item['answer']}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h4>images:</h4>", unsafe_allow_html=True)
    for path in item["image_paths"]:
        st.image(path, use_container_width=True)

with col2:
    st.markdown("<h3>reasoning path annotation</h3>", unsafe_allow_html=True)
    if "reasoning_steps" not in item or not isinstance(item["reasoning_steps"], list):
        item["reasoning_steps"] = [[]]
    while len(item["reasoning_steps"]) < 3:
        item["reasoning_steps"].append([])
    for i in range(len(item["reasoning_steps"])):
        if not isinstance(item["reasoning_steps"][i], list):
            item["reasoning_steps"][i] = []
        if len(item["reasoning_steps"][i]) < 1:
            item["reasoning_steps"][i].append({
                "reasoning step": 1,
                "reasoning type": "",
                "rationale": "",
                "rationale_chinese": "",
                "edited": False
            })

    updated_rationales = []


    for i, path in enumerate(item["reasoning_steps"]):
        with st.expander(f"reasoning path {i+1}", expanded=(i==0)): 
            st.markdown(f"<div class='path-header'>🛤️ path {i+1}</div>", unsafe_allow_html=True)
            path_rationales = []

            for j, step in enumerate(path):
                step_type = step['reasoning type'] if step['reasoning type'] else "unknown"
                
                st.markdown(f"""
                <div class='step-container'>
                    <p><strong>step {step['reasoning step']}:</strong> {step_type}</p>
                </div>
                """, unsafe_allow_html=True)
                original_text = step["rationale_chinese"]

                rationale_chinese = st.text_area(
                    f"chinese step (path {i+1}, step {j+1})",
                    original_text,
                    height=100,
                    key=f"step_{i}_{j}"
                )
                path_rationales.append(rationale_chinese)

            updated_rationales.append(path_rationales)

    st.markdown("""
    <div class='similarity-section'>
        <h4>path similarity annotation</h4>
    </div>
    """, unsafe_allow_html=True)

    path1_equals_path2 = st.checkbox("path 1 equals path 2", value=item.get("path1_equals_path2", False))
    path1_equals_path3 = st.checkbox("path 1 equals path 3", value=item.get("path1_equals_path3", False))
    path2_equals_path3 = st.checkbox("path 2 equals path 3", value=item.get("path2_equals_path3", False))

    def save_annotation():
        annotation_time = time.time() - st.session_state.start_time  
        current_data = st.session_state.data[st.session_state.index] 

        for i, path in enumerate(current_data["reasoning_steps"]):
            for j, step in enumerate(path):
                if i < len(updated_rationales) and j < len(updated_rationales[i]):
                    original_text = step["rationale_chinese"]
                    new_text = updated_rationales[i][j]
                    current_data["reasoning_steps"][i][j]["rationale_chinese"] = new_text
                    current_data["reasoning_steps"][i][j]["edited"] = (original_text != new_text)

        current_data["path1_equals_path2"] = path1_equals_path2
        current_data["path1_equals_path3"] = path1_equals_path3
        current_data["path2_equals_path3"] = path2_equals_path3
        current_data["annotation_time"] = round(annotation_time, 2)

        st.session_state.progress[str(st.session_state.index)] = True

        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state.progress, f, indent=4, ensure_ascii=False)

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state.data, f, indent=4, ensure_ascii=False)

        st.success(f"✅ Annotation saved! (Time taken: {annotation_time:.2f} seconds)")

    st.button("💾 Save Annotation", on_click=save_annotation, use_container_width=True)