from calendar import c
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from dataloader.dataloader_squad import SquadDataset
import numpy as np

# Define the get_ppl function 
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
    model_name = "gpt2-large"  # or specify your local model path
    model = GPT2LMHeadModel.from_pretrained(model_name).eval()
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")  # Use CPU for this script
    model.to(device)

    # Load qa.log
    # file_path = "path/to/Comattack/open_source_code/QA_val_1213.log"
    file_path = "path/to/Comattack/open_source_code/output_degrade.log"
    threshold = 100 # 50.63
    count = 0 
    context = ""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        # for line in f:
        #     line = line.strip()
        for i in range(len(lines) - 1):  
            if lines[i].startswith("[Final Edited Text]"):
                print(lines[i])
                context = lines[i + 1]
                if not context:
                    continue
                print(f"Context: {context[:50]}...")
                ppl = get_ppl(context, model, tokenizer)
                print(f"Context: {context[:50]}... Perplexity: {ppl:.2f}")
                if ppl > threshold:
                    count += 1
    print(f"Number of contexts with perplexity above threshold: {count}")

    # Load SQuAD dataset
    # raw_dataset = load_dataset("parquet", data_files={
    #     "train": "path/to/Comattack_dataset/squad/train-00000-of-00001.parquet",
    #     "validation": "path/to/Comattack_dataset/squad/validation-00000-of-00001.parquet"
    # })
    # val_dataset = SquadDataset(data=raw_dataset, split="validation")
    # val_loader = DataLoader(val_dataset, batch_size=1)

    # # Calculate perplexity for each context
    # contexts = []
    # for i, sample in enumerate(val_loader):
    #     context = sample["context"][0]
    #     ppl = get_ppl(context, model, tokenizer)
    #     print(f"Context {i}: {context[:50]}... Perplexity: {ppl:.2f}")
    #     contexts.append({
    #         'context': context,
    #         'ppl': ppl
    #     })
    # ppl_values = [item['ppl'] for item in contexts]  # Extract all ppl values
    # if ppl_values:  # Ensure the list is not empty
    #     ppl_threshold = np.percentile(ppl_values, 99)
    #     print(f"Perplexity threshold (99th percentile): {ppl_threshold:.2f}")

    