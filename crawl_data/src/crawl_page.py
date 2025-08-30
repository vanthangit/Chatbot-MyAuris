import requests
from bs4 import BeautifulSoup
import json
import os
import yaml
import re

with open("config/config_myauris.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

service_urls = config["services"]


headers = {
    'User-Agent': 'Mozilla/5.0'
}

output_dir = "data/raw"
os.makedirs(output_dir, exist_ok=True)

def extract_qa_from_html(html, config):
    soup = BeautifulSoup(html, 'html.parser')
    qa_pairs = []

    # Extract từ config
    main_selector = config.get("selectors", {}).get("main_content", "main")
    question_tags = config.get("selectors", {}).get("question_tags", ["h2", "h3"])
    question_pattern = re.compile(config.get("selectors", {}).get("is_question_regex", ".*"))
    exclude_keywords = config.get("selectors", {}).get("exclude_keywords", [])

    def should_exclude(text):
        return any(keyword.lower() in text.lower() for keyword in exclude_keywords)

    def add_qa(question, answer):
        if question and answer and not should_exclude(question):
            qa_pairs.append({"question": question.strip(), "answer": answer.strip()})

    # --- Phần main content ---
    main = soup.select_one(main_selector)
    if main:
        current_q, current_a = None, []
        for tag in main.find_all(question_tags + ['p', 'table', 'li']):
            tag_text = tag.get_text(strip=True)
            if tag.name in question_tags and question_pattern.match(tag_text):
                if current_q and current_a:
                    add_qa(current_q, "\n".join(current_a))
                current_q = tag_text
                current_a = []
            elif current_q:
                current_a.append(tag_text)
        if current_q and current_a:
            add_qa(current_q, "\n".join(current_a))

    # --- Phần FAQ section ---
    faq_cfg = config.get("faq_section", {})
    for section in soup.find_all(faq_cfg.get("section_tags", ["section", "div"])):
        h2 = section.find('h2')
        if h2 and faq_cfg.get("title_keyword", "").lower() in h2.get_text(strip=True).lower():
            for item in section.select(".accordion-item"):
                q_el = item.select_one(faq_cfg.get("question_selector", ""))
                a_el = item.select_one(faq_cfg.get("answer_selector", ""))
                if q_el and a_el:
                    add_qa(q_el.get_text(), a_el.get_text())

    return qa_pairs

def crawl_page():
    with open("config/config_myauris.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    urls = config["services"]

    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                print(f"[WARN] Failed to fetch {url}")
                continue

            qa_data = extract_qa_from_html(response.text, config)
            for qa in qa_data:
                qa["source_url"] = url

            filename = url.rstrip("/").split("/")[-1].replace("-", "_") + ".json"
            output_path = os.path.join(output_dir, filename)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(qa_data, f, ensure_ascii=False, indent=2)

            print(f"[OK] Saved: {output_path}")
        except Exception as e:
            print(f"[ERROR] {url}: {e}")


if __name__ == "__main__":
    crawl_page()