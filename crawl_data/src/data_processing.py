# process_dichvu_data.py

import os
import glob
import json
import pandas as pd
import re
import yaml


def clean_answer(text):
    """Xử lý làm sạch câu trả lời."""
    ad_keywords = [
        "ĐẶT LỊCH TƯ VẤN",
        "Xem chi tiết",
        "090 195 8868"
    ]
    for keyword in ad_keywords:
        text = text.replace(keyword, '')

    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def clean_question(question):
    # Bỏ số thứ tự đầu dòng (VD: "1. ", "2) ", "3 -")
    question = re.sub(r"^\s*\d+[\.\)\-]\s*", "", question)

    # Loại bỏ khoảng trắng dư thừa
    question = re.sub(r"\s+", " ", question).strip()

    # Viết hoa chữ cái đầu
    question = question[0].upper() + question[1:] if question else question

    return question

# Danh sách từ khóa cho từng intent
with open("config/intent_keywords.yaml", "r", encoding="utf-8") as f:
    intent_keywords = yaml.safe_load(f)


# Hàm gán intent
def assign_intent(question):
    question = question.lower()
    for intent, keywords in intent_keywords.items():
        for keyword in keywords:
            if keyword in question:
                return intent
    return "khac"

def process_raw_data(input_folder="data/raw/", output_file="data/clean/dichvu_qa.json"):
    all_rows = []

    for file_path in glob.glob(os.path.join(input_folder, "*.json")):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                qas = json.load(f)
                for qa in qas:
                    question = clean_question(qa.get("question", "")).strip()
                    answer = clean_answer(qa.get("answer", "").strip())
                    intent = assign_intent(question)
                    source_url = qa.get("source_url", "").strip()
                    all_rows.append({
                        "category": os.path.splitext(os.path.basename(file_path))[0],
                        "intent": intent,
                        "question": question,
                        "answer": answer,
                        "source_url": source_url
                    })
            except json.JSONDecodeError:
                print(f"Lỗi đọc file: {file_path}")

    df = pd.DataFrame(all_rows)

    df = df.dropna(subset=["question", "answer"])
    df = df.drop_duplicates(subset=["question", "answer"])

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_json(output_file, orient="records", force_ascii=False, indent=2)

    print(f"Đã xử lý và lưu {len(df)} Q&A có intent tại {output_file}")

if __name__ == "__main__":
    process_raw_data()