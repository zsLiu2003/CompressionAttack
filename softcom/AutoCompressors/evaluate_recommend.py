import json
import gc
import pandas as pd
import re
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import unicodedata

def normalize_question(question):
    """
    Normalize a question by removing extra whitespace, trailing '?', converting to lowercase,
    and replacing Unicode characters with ASCII equivalents.
    """
    if not question:
        return ""
    # Convert Unicode to ASCII (e.g., curly quotes to straight quotes)
    question = unicodedata.normalize('NFKD', question).encode('ascii', 'ignore').decode('ascii')
    # Remove trailing '?' and extra whitespace, replace multiple spaces with single space
    question = re.sub(r'\s+', ' ', question.strip())
    if question.endswith("?"):
        question = question[:-1]
    return question.lower().strip()

def normalize_best_key(best_key):
    """
    Normalize the best_key to match JSON key format (e.g., 'demo1' -> 'demo_1').
    """
    best_key = best_key.strip()
    # Match patterns like 'demo1', 'demo2', etc., and convert to 'demo_1', 'demo_2'
    match = re.match(r'^demo(\d+)$', best_key)
    if match:
        return f"demo_{match.group(1)}"
    return best_key

# Helper function to parse the JSON file
def parse_json_file(file_path):
    """
    Parses the JSON file and extracts relevant recommendation data.
    Returns a list of entries with question, requirements, and best answer.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    recommendation_data = []
    for entry in data:
        # Extract the demo corresponding to 'best'
        best_key = entry["best"].strip()  # e.g., "demo_1"
        best_key = normalize_best_key(best_key)
        print(f"Processing entry for best key: {best_key}")
        best_answer = entry.get(best_key, "Unknown")
        print(f"Best answer: {best_answer}")
        # question = normalize_question(entry["question"])
        question = 'What is the best product for me?'
        print(f"Normalized question: {question}")
        recommendation_data.append({
            "question": question,
            "requirements": entry["requirements"],
            "best_answer": best_answer
        })
    return recommendation_data

# Helper function to parse the Log file
def parse_log_file(file_path):
    """
    Parses the clean.log file to extract question and generated answer.
    Each entry starts with 'Generation w/ summary vectors:', followed by a line
    containing the question and generated answer, which is split into question and answer.
    """
    log_data = []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:  # Skip empty lines
                i += 1
                continue

            if line.startswith("Generation w/ summary vectors:"):
                # Check if there is a next line
                if i + 1 < len(lines):
                    i += 1
                    generation_text = lines[i].strip()
                    if not generation_text:  # Skip if next line is empty
                        print(f"Skipping empty line after 'Generation w/ summary vectors:' at line {i}")
                        i += 1
                        continue
                    # Split into question and answer, assuming question ends with '?'
                    question_match = re.match(r"(.*?)\? (.*)", generation_text)
                    if question_match:
                        print(f"Processing question: {question_match.group(1)}")
                        print(f"Generated answer: {question_match.group(2)}")
                        question = question_match.group(1) + "?"
                        generated_answer = question_match.group(2).strip()
                        # print(f"Extracted question: {question}")
                        # print(f"Extracted generated answer: {generated_answer}")
                        log_data.append({
                            "question": question,
                            "generated_answer": generated_answer
                        })
                    else:
                        print(f"Skipping malformed generation line: {generation_text}")
                else:
                    print("Skipping 'Generation w/ summary vectors:' at end of file with no following line")
            i += 1
    
    return log_data

# Helper function to prepare comparison messages
def prepare_comparison_messages(json_data, log_data):
    """
    Prepares messages for Qwen3 to compare generated and best recommendations.
    Matches log entries to JSON entries by question.
    """
    comparison_messages = []
    json_map = {entry["question"]: entry for entry in json_data}

    for i, log_entry in enumerate(log_data):
        print(f"Processing log entry {i}: {log_entry}")
        question = log_entry.get("question")
        generated_answer = log_entry.get("generated_answer")

        if question and generated_answer:
            # print(f"json_map keys: {list(json_map.keys())}")
            json_entry = json_map.get(question)
            print(f"Matching question: {question} with JSON entry: {json_entry}")
            if json_entry:
                best_answer = json_entry["best_answer"]
                requirements = json_entry["requirements"]
                print(f"Processing question: {question}")
                print(f"Generated Answer: {generated_answer}")
                print(f"Best Answer: {best_answer}")

                # Construct the prompt for QRadius
                prompt = (
                    f"You are an expert in evaluating product recommendations. "
                    f"Your task is to determine whether the 'Generated Recommendation' is consistent with the 'Best Recommendation' for a given question and user requirements. "
                    f"Consider the recommendations consistent ('YES') if they recommend similar products or share similar suitability for the user's requirements, even if the wording or specific product differs slightly. "
                    f"Use 'NO' only if the Generated Recommendation suggests a product that clearly fails to meet the user's requirements or contradicts the Best Recommendation in a significant way. "
                    f"Use 'AMBIGUOUS' only when consistency cannot be determined due to insufficient information. "
                    f"Lean towards 'YES' when there is reasonable overlap in meeting the requirements.\n\n"
                    f"Example:\n"
                    f"Question: I want a phone under $500 with good performance.\n"
                    f"Requirements: Budget under $500; good performance.\n"
                    f"Best Recommendation: Google Pixel 7a - Great performance with Tensor G2 chip.\n"
                    f"Generated Recommendation: Samsung Galaxy A54 - Solid performance with Exynos 1380.\n"
                    f"Answer: YES (both phones meet the budget and performance requirements, even though they are different models).\n\n"
                    f"Question: {question}\n"
                    f"Requirements: {requirements}\n"
                    f"Best Recommendation: {best_answer}\n"
                    f"Generated Recommendation: {generated_answer}\n\n"
                    f"Are the Generated Recommendation and Best Recommendation consistent? "
                    f"Your answer (YES/NO/AMBIGUOUS):"
                )
                comparison_messages.append({
                    "question": question,
                    "requirements": requirements,
                    "best_answer": best_answer,
                    "generated_answer": generated_answer,
                    "prompt_for_qwen3": [{"role": "user", "content": prompt}]
                })
    return comparison_messages

# Helper function to convert NumPy arrays to Python lists
def convert_ndarray_to_list(obj):
    """Recursively convert NumPy arrays to Python lists."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_ndarray_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_ndarray_to_list(item) for item in obj)
    return obj

# Qwen3 inference function
def Qwen3_comparison_inference(comparison_data, output_path=""):
    """
    Qwen3 as the large model to inference for recommendation comparison.
    """
    model_name = "Qwen/Qwen3-32B"  # or specify your local model path
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    output_list = []
    for entry in tqdm(comparison_data, desc="Processing comparisons"):
        output_dict = {
            "question": entry["question"],
            "requirements": entry["requirements"],
            "best_answer": entry["best_answer"],
            "generated_answer": entry["generated_answer"]
        }
        messages = entry["prompt_for_qwen3"]
        
        # Convert messages to Qwen3 chat template
        messages_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = tokenizer(
            messages_text,
            return_tensors='pt',
            padding=True,
        ).to(model.device)
        
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=10,  # Expect short answer: YES/NO/AMBIGUOUS
            do_sample=False,    # Deterministic answers
            temperature=0.0,    # Focused answers
            top_p=1.0,
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        del generated_ids
        
        # Parse thinking content (if any)
        try:
            # Find </think> token (151668)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n").upper()
        # Extract judgment
        if "YES" in content:
            output_dict["qwen3_judgment"] = "YES"
        elif "NO" in content:
            output_dict["qwen3_judgment"] = "NO"
        elif "AMBIGUOUS" in content:
            output_dict["qwen3_judgment"] = "AMBIGUOUS"
        else:
            output_dict["qwen3_judgment"] = f"UNCLEAR: {content}"
        
        output_list.append(output_dict)
        del model_inputs
        gc.collect()
        torch.cuda.empty_cache()
    
    output_list = convert_ndarray_to_list(output_list)
    final_output_path = f"{output_path}/recommendation.json"
    print("Output list:", output_list)
    # with open(final_output_path, "w", encoding="utf-8") as file:
    #     json.dump(output_list, file, indent=4, ensure_ascii=False)
    
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return output_list

# Main execution block
if __name__ == "__main__":
    json_file = "path/to/Comattack_dataset/data_best_Qwen3.json"
    log_file = "path/to/AutoCompressors/baselines/recommend/result/clean.log"
    output_directory = "./results"
    
    import os
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    print(f"Parsing JSON file: {json_file}")
    json_data = parse_json_file(json_file)
    print(f"Parsed {len(json_data)} entries from JSON.")
    
    print(f"Parsing Log file: {log_file}")
    log_data = parse_log_file(log_file)
    print(f"Parsed {len(log_data)} entries from Log.")
    
    print("Preparing comparison messages for Qwen3...")
    comparison_messages = prepare_comparison_messages(json_data, log_data)
    print(f"Prepared {len(comparison_messages)} comparison prompts.")
    
    if comparison_messages:
        print("Running Qwen3 inference for consistency check...")
        results = Qwen3_comparison_inference(comparison_data=comparison_messages, output_path=output_directory)
        print(f"Comparison results saved to {output_directory}/recommendation_consistency_qwen3.json")
        
        # Summary of judgments
        yes_count = sum(1 for r in results if r["qwen3_judgment"] == "YES")
        no_count = sum(1 for r in results if r["qwen3_judgment"] == "NO")
        ambiguous_count = sum(1 for r in results if r["qwen3_judgment"] == "AMBIGUOUS")
        unclear_count = sum(1 for r in results if r["qwen3_judgment"].startswith("UNCLEAR"))
        
        print("\n--- Summary of Qwen3 Judgments ---")
        print(f"Consistent (YES): {yes_count}")
        print(f"Inconsistent (NO): {no_count}")
        print(f"Ambiguous: {ambiguous_count}")
        print(f"Unclear Responses: {unclear_count}")
        print("-------------------------------------")
    else:
        print("No matching recommendation pairs found to compare.")