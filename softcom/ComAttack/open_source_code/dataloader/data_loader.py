import json
import re
import sys
from sympy import N
from torch.utils.data import Dataset
from tqdm import tqdm

class BaseMultiOutputDataset(Dataset):
    def __init__(self, json_path, output_keys, tokenizer=None, max_length=512):
        """
        :param json_path: Path to JSON file
        :param output_keys: List of keys to extract outputs (e.g., ['output1', 'output2'])
        :param tokenizer: HuggingFace tokenizer (optional)
        :param max_length: Max token length if using tokenizer
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.output_keys = output_keys
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        question = entry["question"]
        requirement = entry["requirements"]
        outputs = [entry[key] for key in self.output_keys]

        if self.tokenizer:
            encoded = self.tokenizer(
                [question] * len(outputs),  # paired with each output
                outputs,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            return {
                'input_ids': encoded['input_ids'],  # shape: (num_choices, seq_len)
                'attention_mask': encoded['attention_mask'],
                'question': question,
                'requirement': requirement,
                'outputs': outputs
            }
        else:
            return {
                'question': question,
                'requirement': requirement,
                'outputs': outputs
            }

class FullMultiOutputDataset(BaseMultiOutputDataset):
    def __init__(self, json_path, tokenizer=None, max_length=512):
        output_keys = ['output1', 'output2', 'output3', 'output4', 'output5']
        super().__init__(json_path, output_keys, tokenizer, max_length)

class PartialMultiOutputDataset(BaseMultiOutputDataset):
    def __init__(self, json_path, tokenizer=None, max_length=512):
        output_keys = ['output1', 'output2']
        super().__init__(json_path, output_keys, tokenizer, max_length)


class FullMultiOutputDatasetWithTarget(Dataset):
    def __init__(self, json_path, tokenizer=None):
        self.tokenizer = tokenizer
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.samples = []
        for entry in self.data:
            question = entry.get("question", "")

            # all requirement fields
            requirements = [
                value for key, value in entry.items()
                if key.startswith("requirement")
            ]
            
            # all demo_X 
            demos = [entry[f"demo_{i}"] for i in range(1, 6) if f"demo_{i}" in entry]
            
            best_key = entry.get("best")
            target_key = entry.get("target")
            # print(f"best_key: {best_key}, target_key: {target_key}")

            best_idx = int(re.search(r'(\d+)$', best_key).group(1)) - 1 if best_key else None
            target_idx = int(target_key.split("_")[1]) - 1 if target_key else None

            self.samples.append({
                "question": question,
                "requirements": requirements,
                "demos": demos,
                "best": best_idx if best_idx is not None else None,
                "target": target_idx if target_idx is not None else None
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class FiveDemoDataset(Dataset):
    def __init__(self, json_path, tokenizer=None):
        self.tokenizer = tokenizer
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.samples = []
        for entry in self.data:
            demos = [entry[f"demo_{i}"] for i in range(1, 6) if f"demo_{i}" in entry]

            self.samples.append({
                "demos": demos
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
class KeywordDataset(Dataset):
    def __init__(self, json_path, tokenizer=None):
        self.tokenizer = tokenizer
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.samples = []
        for entry in self.data:
            demo_keywords = [entry[f"demo{i}"] for i in range(1, 6) if f"demo{i}" in entry]

            self.samples.append({
                "demo_keywords": demo_keywords
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

if __name__ == "__main__":
    # Example usage
    # target_key = 0
    # keyword_dataset = KeywordDataset("./datasets/data_keywords_with_Qwen3.json")
    # for i, sample in tqdm(enumerate(keyword_dataset), desc="Running ICAE Edit Recommendation Attacker"):
    #     print(sample["demo_keywords"])
    #     print(sample["demo_keywords"][target_key])
    dataset = FullMultiOutputDatasetWithTarget("./datasets/data_best_Qwen3.json")
    print(f"Loaded dataset with {len(dataset)} samples.")
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    for i, sample in tqdm(enumerate(dataset), desc="Running ICAE Edit Recommendation Attacker"):
        demos = sample["demos"][0]
        best = sample["best"]
        target = sample["target"]
        question = sample["question"][0]
        requirement = sample["requirements"]
        # print(sample.keys())  
        # print(sample['question'])
        print(sample['requirements'])
        # print(sample['demos'][0])
        # print(sample['best'])
        # print(sample['target'])
        i+=1
        if i > 2:
            sys.exit(0)
