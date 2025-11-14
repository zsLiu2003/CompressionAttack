# calculate_clean_ppl.py
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from dataloader.dataloader_squad import SquadDataset
import numpy as np

# Define the get_ppl function (removing 'self' since it's not in a class)
def get_ppl(text: str, model, tokenizer) -> float:
    """
    Calculates the overall perplexity of a given text.
    """
    device = model.device
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    
    max_length = model.config.n_positions
    if input_ids.size(1) > max_length:
        input_ids = input_ids[:, :max_length]

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    ppl = torch.exp(loss)
    return ppl.item()

if __name__ == "__main__":
    # Load model and tokenizer from the first script
    model_name = "path/to/gpt2-large"
    model = GPT2LMHeadModel.from_pretrained(model_name).eval()
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load SQuAD dataset
    raw_dataset = load_dataset("parquet", data_files={
        "train": "path/to/squad/train-00000-of-00001.parquet",
        "validation": "path/to/squad/validation-00000-of-00001.parquet"
    })
    val_dataset = SquadDataset(data=raw_dataset, split="validation")
    val_loader = DataLoader(val_dataset, batch_size=1)

    # Calculate perplexity for each context
    contexts = []
    for i, sample in enumerate(val_loader):
        context = sample["context"][0]
        ppl = get_ppl(context, model, tokenizer)
        print(f"Context {i}: {context[:50]}... Perplexity: {ppl:.2f}")
        contexts.append({
            'context': context,
            'ppl': ppl
        })
    ppl_values = [item['ppl'] for item in contexts]  # Extract all ppl values
    if ppl_values:  # Ensure the list is not empty
        ppl_threshold = np.percentile(ppl_values, 99)
        print(f"Perplexity threshold (99th percentile): {ppl_threshold:.2f}")

    