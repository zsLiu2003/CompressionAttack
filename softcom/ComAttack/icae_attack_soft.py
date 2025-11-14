import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from transformers import HfArgumentParser
from modeling_icae_multi_span import ICAE, ModelArguments, DataArguments, TrainingArguments
from peft import LoraConfig
from safetensors.torch import load_file

device = "cuda"
num_steps = 1000
suffix_len = 5
save_path = "icae_soft_attack.pth"

# Load arguments and ICAE model
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
model.eval()

# Get target embedding
target_text = "This is the target text that we want the suffix to mimic."
with torch.no_grad():
    target_input_ids = model.tokenizer(target_text, truncation=True, max_length=5120, padding=False, return_attention_mask=False)['input_ids']
    target_input_ids = torch.LongTensor([target_input_ids]).to(device)
    target_embedding = model._compress(target_input_ids).detach()

# Tokenize base input
base_text = "I don't have a favorite condiment..."
base_input_ids = model.tokenizer(base_text, truncation=True, max_length=5120, padding=False, return_attention_mask=False)['input_ids']
base_input_ids = torch.LongTensor([base_input_ids]).to(device)

# Get base embeddings
with torch.no_grad():
    base_embeds = model.tokens_to_embeddings(base_input_ids)  # [1, L, D]

# Soft suffix: trainable token logits
vocab_size = model.get_input_embeddings().weight.shape[0]
embed_dim = model.get_input_embeddings().weight.shape[1]
token_logits = torch.randn((suffix_len, vocab_size), requires_grad=True, device=device)

optimizer = Adam([token_logits], lr=1e-2)

for step in tqdm(range(num_steps)):
    optimizer.zero_grad()

    # Convert to soft embeddings
    token_probs = F.softmax(token_logits, dim=-1).to(model.icae.dtype)                   # [suffix_len, vocab_size]
    vocab_embeds = model.get_input_embeddings().weight              # [vocab_size, D]
    soft_suffix = token_probs @ vocab_embeds                        # [suffix_len, D]

    # Combine with base
    combined = torch.cat([base_embeds, soft_suffix.unsqueeze(0)], dim=1)  # [1, L+suffix, D]
    memory_slot = model._compress(inputs_embeds=combined)

    loss = -F.cosine_similarity(memory_slot, target_embedding.unsqueeze(0), dim=-1).mean()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"Step {step}: cosine_loss={-loss.item():.4f}")

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, save_path)

with torch.no_grad():
    token_probs = F.softmax(token_logits, dim=-1)  # [suffix_len, vocab_size]
    token_ids = torch.argmax(token_probs, dim=-1).tolist()
    decoded_suffix = model.tokenizer.decode(token_ids)
    print("\n Soft Suffix Decoded:", decoded_suffix)

# Append to base input
final_input_ids = torch.cat([base_input_ids[0], torch.LongTensor(token_ids).to(device)], dim=0).unsqueeze(0)

with torch.no_grad():
    memory_final = model._compress(final_input_ids)
    final_sim = F.cosine_similarity(memory_final, target_embedding, dim=-1).mean().item()

print(f"[Direct] Cosine similarity with target: {final_sim:.4f}")

# Also eval soft suffix directly
with torch.no_grad():
    soft_suffix = (F.softmax(token_logits, dim=-1).to(model.icae.dtype) @ model.get_input_embeddings().weight).unsqueeze(0)
    soft_input = torch.cat([base_embeds, soft_suffix], dim=1)
    soft_memory = model._compress(inputs_embeds=soft_input)
    soft_sim = F.cosine_similarity(soft_memory, target_embedding.unsqueeze(0), dim=-1).mean().item()

print(f"[Soft] Cosine similarity with target (soft suffix): {soft_sim:.4f}")