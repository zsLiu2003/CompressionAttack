import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from transformers import HfArgumentParser
from modeling_icae_multi_span import ICAE, ModelArguments, DataArguments, TrainingArguments
from peft import LoraConfig
from safetensors.torch import load_file

device = "cuda"
num_steps = 3000
save_path = "icae_attack.pth"

# Load arguments and initialize ICAE model
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

# ------------------- Step 1: Get Target Embedding -------------------

target_text = "This is the target text that we want the suffix to mimic."

# Tokenize and get memory slot embedding for target
with torch.no_grad():
    target_input_ids = model.tokenizer(target_text, truncation=True, max_length=5120, padding=False, return_attention_mask=False)['input_ids']
    target_input_ids = torch.LongTensor([target_input_ids]).to(device)
    target_embedding = model._compress(target_input_ids)  # shape: [slot_dim]
    target_embedding = target_embedding.detach()

# ------------------- Step 2: Optimize Suffix Embedding -------------------

base_text = "I don't have a favorite condiment..."

base_input_ids = model.tokenizer(base_text, truncation=True, max_length=5120, padding=False, return_attention_mask=False)['input_ids']
base_input_ids = torch.LongTensor([base_input_ids]).to(device)

# Get embedding for base input (no grad)
with torch.no_grad():
    base_embeddings = model.tokens_to_embeddings(base_input_ids)  # [1, L, D]

# Initialize trainable suffix
suffix_len = 5
embed_dim = model.tokens_to_embeddings(torch.LongTensor([[1]]).to(device)).shape[-1]
trainable_suffix = torch.randn((1, suffix_len, embed_dim), requires_grad=True, device=device)

optimizer = Adam([trainable_suffix], lr=1e-2)

for step in tqdm(range(num_steps)):
    optimizer.zero_grad()

    # Convert base input to embeddings
    base_embeds = model.tokens_to_embeddings(base_input_ids.unsqueeze(0))  # [1, L, D]
    
    # Combine original input embeddings and suffix embeddings
    combined_embedding = torch.cat([base_embeddings, trainable_suffix], dim=1)  # [1, L+suffix, D]

    # print(model.icae.dtype)
    # encoder
    memory_slot = model._compress(inputs_embeds=combined_embedding.to(dtype=model.icae.dtype))

    # Cosine loss w.r.t. target embedding
    loss = -F.cosine_similarity(memory_slot, target_embedding.unsqueeze(0), dim=-1).mean()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"Step {step}: cosine_loss={loss.item():.4f}")

# torch.save({
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
# }, save_path)
# trainable_suffix_decoded = model.tokenizer.decode(trainable_suffix[0].argmax(dim=-1).cpu().numpy())
# print(f"Optimized Suffix: {trainable_suffix_decoded}")
combined_embedding = torch.cat([base_embeddings, trainable_suffix], dim=1)
final_memory = model._compress(inputs_embeds=combined_embedding.to(dtype=model.icae.dtype))
final_sim = F.cosine_similarity(final_memory, target_embedding, dim=-1).mean().item()
print(f"[Direct] Cosine similarity with target: {final_sim:.4f}")

# # ------------------- Step 3&4 : Soft Decode & Evaluate -------------------
# with torch.no_grad():
#     suffix_embeds = trainable_suffix.detach()[0]  # [suffix_len, D]
#     vocab_embeds = model.get_input_embeddings().weight  # [vocab_size, D]

#     soft_suffix_embeds = []
#     for i in range(suffix_len):
#         # Compute cosine similarity with vocab
#         sim = F.cosine_similarity(suffix_embeds[i].unsqueeze(0), vocab_embeds, dim=-1)  # [vocab_size]
#         weights = F.softmax(sim / 0.1, dim=-1).to(model.icae.dtype)  # optional temperature τ=0.1
#         soft_token_embed = torch.matmul(weights, vocab_embeds)  # [D]
#         soft_suffix_embeds.append(soft_token_embed)

#     soft_suffix_embeds = torch.stack(soft_suffix_embeds, dim=0).unsqueeze(0)  # [1, suffix_len, D]
#     soft_suffix_embeds_decoded = model.tokenizer.decode(soft_suffix_embeds[0].argmax(dim=-1).cpu().numpy())  # Decode soft suffix
#     print(f"Soft Suffix Decoded: {soft_suffix_embeds_decoded}")

#     # Combine base and soft suffix
#     combined_embedding = torch.cat([base_embeddings, soft_suffix_embeds], dim=1)  # [1, L+suffix, D]

#     # Encode and compare
#     final_memory = model._compress(inputs_embeds=combined_embedding.to(dtype=model.icae.dtype))
#     final_sim = F.cosine_similarity(final_memory, target_embedding, dim=-1).mean().item()
#     print(f" Cosine similarity with target (soft suffix): {final_sim:.4f}")

# ------------------- Step 3: Decode Suffix -------------------

with torch.no_grad():
    suffix_embeds = trainable_suffix.detach()[0]  # shape: [suffix_len, D]
    vocab_embeds = model.get_input_embeddings().weight # shape: [vocab_size, D]

    token_ids = []
    for i in range(suffix_len):
        sim = F.cosine_similarity(suffix_embeds[i].unsqueeze(0), vocab_embeds, dim=-1)
        token_id = torch.argmax(sim).item()
        token_ids.append(token_id)

    decoded_suffix = model.tokenizer.decode(token_ids)
    print("\n Optimized Suffix:", decoded_suffix)

# ------------------- Step 4: Final Similarity -------------------

# Combine input ids and optimized suffix
final_input_ids = torch.cat([base_input_ids[0], torch.LongTensor(token_ids).to(device)], dim=0).unsqueeze(0)
# print("base_input_ids ",base_input_ids)
# print("base_input_ids shape: ",base_input_ids.shape)
# base_input_decoded = model.tokenizer.decode(base_input_ids[0])
# token_decoded = model.tokenizer.decode(token_ids)
# print("token_ids shape: ",len(token_ids))
# print("base_input_decoded: ",base_input_decoded)
# print("token_decoded: ",token_decoded)
with torch.no_grad():
    final_memory = model._compress(final_input_ids)
    # print(final_memory.shape)
    # print(target_embedding.shape)
    final_sim = F.cosine_similarity(final_memory, target_embedding, dim=-1)
    print(final_sim.shape)
    final_sim = final_sim.mean().item()

print(f" Cosine similarity with target: {final_sim:.4f}")
