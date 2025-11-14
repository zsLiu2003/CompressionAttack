import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from transformers import HfArgumentParser
from modeling_icae_multi_span import ICAE, ModelArguments, DataArguments, TrainingArguments
from peft import LoraConfig
from safetensors.torch import load_file
from utils import generate_output_from_attacked_suffix
from icae_attack import MinimalEditICAEAttacker, TargetedICAEAttacker, NonTargetedICAEAttacker
from dataloader.data_loader import FullMultiOutputDataset, PartialMultiOutputDataset, FullMultiOutputDatasetWithTarget

device = "cuda"

# Load model and arguments
parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

lora_config = LoraConfig(
    r=512,
    lora_alpha=32,
    lora_dropout=model_args.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM"
)

model = ICAE(model_args, training_args, lora_config)
state_dict = load_file(training_args.output_dir)
model.load_state_dict(state_dict, strict=False)
model = model.to(device)

dataset = FullMultiOutputDatasetWithTarget("./datasets/data_with_target.json", tokenizer=None)
print(f"Loaded dataset with {len(dataset)} samples.")
for idx in tqdm(range(len(dataset)), desc="Running attacks"):
    sample = dataset[idx]

    base_text = sample['demos']  
    target_text = sample['requirements']  # requirement
    prompt_text = sample['question']
    best_text = sample['best']  # best
    target_text = sample['target']  # target
    

    #  Option 1: Targeted attack
    # attacker = TargetedICAEAttacker(model, base_text, target_text, suffix_len=5, num_steps=50, device=device)

    #  Option 2: Non-targeted attack
    # attacker = NonTargetedICAEAttacker(model, base_text, suffix_len=5, num_steps=50, device=device)

    #  Option 3: Minimal edit attack
    attacker = MinimalEditICAEAttacker(model, base_text, target_text=target_text, num_steps=50, device=device, targeted=True)


    edited_text, token_ids = attacker.run()

    generated_answer = generate_output_from_attacked_suffix(
        model,
        base_text=base_text,
        suffix_token_ids=token_ids,
        prompt_text=prompt_text,
        max_out_length=128,
        device="cuda"
    )

    print("\n[Final Answer]:", generated_answer)