import requests
import streamlit as st
import json

API_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual API key
GROUP_ID = "YOUR_GROUP_ID_HERE"  # Replace with your actual Group ID
MODEL_NAME = "MiniMax-Text-01"
OUTPUT_JSON_PATH = "./outputs/benchmark_outputs.json"

# build prompt
def build_prompt(task_description, input_example, target_list):
    targets = "\n".join([f"- {item}" for item in target_list])
    json_entries = ",\n".join(
        [f'    {{"target": "{item}", "generated_text": "..."}}' for item in target_list]
    )
    
    prompt = f"""
{task_description}

## Example Input:
{input_example}

## Your Task:
Based on the example above, generate corresponding outputs for the following targets:
{targets}

Return the output strictly in the following JSON format:

{{
  "outputs": [
{json_entries}
  ]
}}

Ensure consistency with the example. Keep all outputs in English. Avoid generic descriptions.
""".strip()
    
    return prompt


# API
def call_minimax(prompt):
    url = "https://api.minimax.chat/v1/text/chatcompletion_v2"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None

# save to JSON 
def save_output_to_json(text, path):
    try:
        parsed = json.loads(text)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)
        print(f"✅ Output saved to {path}")
    except json.JSONDecodeError:
        print("❌ Failed to parse output as JSON.")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"⚠️ Raw output saved to {path} instead.")


task_description = """
You are a professional copywriter. Your task is to generate promotional texts in different brand styles based on a given product description. Each output should be approximately 1000 words and match the tone and branding style of the specified target.
"""

input_example = """
Brand: Apple
Product: iPhone 15 Pro
Promotional Text:
Introducing the iPhone 15 Pro – a remarkable fusion of design, performance, and innovation. Crafted with aerospace-grade titanium, it’s not only stronger but lighter than ever. Powered by the A17 Pro chip, it redefines speed and efficiency [...]
"""

target_list = [
    "Samsung: Galaxy S24 Ultra",
    "Google: Pixel 8 Pro",
    "OnePlus: OnePlus 12",
    "Xiaomi: Xiaomi 14 Ultra",
    "Huawei: Mate 60 Pro"
]

if __name__ == "__main__":
    prompt = build_prompt(task_description, input_example, target_list)
    result = call_minimax(prompt)
    if result:
        save_output_to_json(result, OUTPUT_JSON_PATH)