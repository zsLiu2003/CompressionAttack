from regex import F, P
import requests
import json
from data_loader import FullMultiOutputDataset, PartialMultiOutputDataset

API_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual API key
GROUP_ID = "YOUR_GROUP_ID_HERE"  # Replace with your actual Group ID
MODEL_NAME = "MiniMax-Text-01"

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
        print(result)
        return result["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None

def extract_keywords(dataset):
    results = []

    for sample in dataset:
        question = sample['question']
        requirement = sample['requirement']
        outputs = sample['outputs']

        result = {
            "question": question,
            "requirements": requirement,
            "summarized_keywords": []
        }

        for i, output in enumerate(outputs):
            prompt = f"""
You are given a user's phone purchasing requirements and a recommendation text.

Requirements:
{requirement}

Recommendation:
{output}

Your task is to extract concise keywords that correspond to each of the above requirements.
Please return the results as:
- Requirement 1: [...]
- Requirement 2: [...]
- ...
If a requirement is not clearly addressed, mark it as "Not mentioned".
"""
            keywords = call_minimax(prompt)
            result["summarized_keywords"].append({f"output{i+1}": keywords})

        results.append(result)

    return results

if __name__ == "__main__":
    dataset = FullMultiOutputDataset("./datasets/data.json", tokenizer=None)
    results = extract_keywords(dataset)

    with open("keywords_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)