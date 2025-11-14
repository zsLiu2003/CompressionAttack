
from torch.utils.data import Dataset
import json

class SquadDataset(Dataset):
    def __init__(self, file_path=None, data=None, split="train"):
        """
        Initialize the raw SQuAD dataset loader.
        
        Args:
            file_path: Path to JSON file (optional)
            data: Pre-loaded dataset (optional)
            split: "train" or "validation"
        
        Note:
            Must provide either file_path or data parameter
        """
        if file_path:
            # Load data from local JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                squad_data = json.load(f)
            self.data = squad_data['data']
        elif data:
            # Use pre-loaded data from Hugging Face
            self.data = data[split]
        else:
            raise ValueError("Must provide either file_path or data parameter")
        
        # Flatten the nested SQuAD structure into list of QA pairs
        self.examples = []
        for article in self.data:
            # print(article.keys()) # (['id', 'title', 'context', 'question', 'answers'])           
            self.examples.append(
                {
                    'id': article['id'],
                    'context': article['context'],
                    'question': article['question'],
                    'title': article['title'],
                    'answers': {
                        'answer_text': article['answers']['text'],
                        'answer_start': article['answers']['answer_start']
                    }
                }
            )

    def __len__(self):
        """Return the total number of examples in dataset"""
        return len(self.examples)
    
    def __getitem__(self, idx):
        """
        Get a single QA example by index
        
        Returns:
            Dictionary containing:
                - id: QA pair ID
                - context: Reference text passage
                - question: Question text
                - answers: List of answer dictionaries
        """
        example = self.examples[idx]
        return {
            'id': example['id'],
            'context': example['context'],
            'question': example['question'],
            'title': example['title'],
            'answers': {
                'answer_text': example['answers']['answer_text'],
                'answer_start': example['answers']['answer_start']
            }
        }
    
from datasets import load_dataset
from torch.utils.data import DataLoader

if __name__ == "__main__":

    # Load raw dataset from Hugging Face
    raw_dataset = load_dataset("rajpurkar/squad")

    # Initialize dataset classes
    train_dataset = SquadDataset(data=raw_dataset, split="train")
    val_dataset = SquadDataset(data=raw_dataset, split="validation")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)

    # Inspect a sample batch
    sample = next(iter(train_loader))
    print(f"ID: {sample['id'][0]}")
    print(f"Context: {sample['context'][0][:100]}...")  # First 100 chars
    print(f"Question: {sample['question'][0]}")
    print(f"Title: {sample['title'][0]}")
    print(f"Answers: {sample['answers']['answer_text'][0]} at {sample['answers']['answer_start'][0]}")