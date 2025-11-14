import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer
from auto_compressor import LlamaAutoCompressorModel

# ------------------- Hyperparameter Configuration -------------------
MODEL_PATH = "path/to/models/AutoCompressor-Llama-2-7b-6k"
MODEL_NAME = "princeton-nlp/AutoCompressor-Llama-2-7b-6k"
suffix_length = 3
lr = 5e-1
n_steps = 100
torch_dtype = torch.bfloat16
prompt_text = 'The first name of the current US president is "'

# Ablation candidates
lambda_sets = [
    (0.0, 0.0, 0.0),
    (0.001, 0.0, 0.0),
    (0.001, 1e-5, 0.0),
    (0.001, 1e-5, 0.01),
    (0.002, 1e-5, 0.01),
]

# --- Device Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Load Model and Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("Loading model...")
model = LlamaAutoCompressorModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch_dtype
).eval().to(device)
model.gradient_checkpointing_enable()
print(f"Model loaded to device: {model.device}")

# --- Prepare Prompt and Context ---
prompt_tokens = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

context_text = """Joe Biden, born in Scranton, Pennsylvania, on November 20, 1942, had a modest upbringing in a middle-class family..."""  # shortened here for readability
context_tokens = tokenizer(context_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
context_length = context_tokens.size(1)
embed_dim = model.get_input_embeddings().weight.size(1)

# --- Precompute Original Softprompt ---
with torch.no_grad():
    context_embed = model.get_input_embeddings()(context_tokens)
    dummy_suffix = torch.zeros((1, suffix_length, embed_dim), device=device, dtype=torch_dtype)
    dummy_input = torch.cat([context_embed, dummy_suffix], dim=1)
    dummy_seg_lengths = [context_length, suffix_length]
    softprompt = model(context_tokens, output_softprompt=True).softprompt
    original_softprompt = model(inputs_embeds=dummy_input, output_softprompt=True, segment_lengths=dummy_seg_lengths).softprompt

# --- Attack Runner ---
def run_attack(lambda1, lambda2, lambda3):
    suffix_embeds = torch.randn((1, suffix_length, embed_dim), requires_grad=True, device=device, dtype=torch_dtype)
    optimizer = torch.optim.Adam([suffix_embeds], lr=lr)

    for step in range(n_steps):
        context_embed = model.get_input_embeddings()(context_tokens)
        attacked_input_embed = torch.cat([context_embed, suffix_embeds], dim=1)
        segment_lengths = [context_length, suffix_length]
        summary_vec = model(inputs_embeds=attacked_input_embed, segment_lengths=segment_lengths, output_softprompt=True).softprompt

        cos_sim = F.cosine_similarity(original_softprompt.flatten(), summary_vec.flatten(), dim=0)
        l2 = torch.norm(suffix_embeds, p=2)
        mse = F.mse_loss(summary_vec, original_softprompt)
        orig_soft = F.softmax(original_softprompt, dim=-1)
        attk_soft = F.log_softmax(summary_vec, dim=-1)
        kl = F.kl_div(attk_soft, orig_soft, reduction="batchmean")

        loss = cos_sim + lambda1 * l2 + lambda2 * kl + lambda3 * mse
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Final metrics
    with torch.no_grad():
        attacked_embed = torch.cat([model.get_input_embeddings()(context_tokens), suffix_embeds], dim=1)
        attacked_summary = model(inputs_embeds=attacked_embed, segment_lengths=[context_length, suffix_length], output_softprompt=True).softprompt
        final_cos_sim = F.cosine_similarity(original_softprompt.flatten(), attacked_summary.flatten(), dim=0).item()
        final_mse = F.mse_loss(attacked_summary, original_softprompt).item()
        final_l2 = torch.norm(suffix_embeds, p=2).item()
        final_kl = F.kl_div(F.log_softmax(attacked_summary, dim=-1), F.softmax(original_softprompt, dim=-1), reduction="batchmean").item()
        gen = model.generate(
            prompt_tokens,
            softprompt=attacked_summary,
            max_new_tokens=4,
            do_sample=False
        )[0]
        return {
            "lambda": (lambda1, lambda2, lambda3),
            "cos_sim": final_cos_sim,
            "mse": final_mse,
            "l2": final_l2,
            "kl": final_kl,
            "generated_text": tokenizer.decode(gen)
        }

# --- Run all configurations ---
results = []
for λ1, λ2, λ3 in lambda_sets:
    print(f"\nRunning attack with λ1={λ1}, λ2={λ2}, λ3={λ3}")
    metrics = run_attack(λ1, λ2, λ3)
    print(metrics)
    results.append(metrics)

# --- Save and plot results ---
df = pd.DataFrame(results)
df.to_csv("ablation_results.csv", index=False)

labels = [str(x["lambda"]) for x in results]
plt.figure(figsize=(10, 6))
plt.plot(labels, [x["cos_sim"] for x in results], marker='o', label="Cosine Similarity ↓")
plt.plot(labels, [x["mse"] for x in results], marker='s', label="MSE Loss ↓")
plt.plot(labels, [x["l2"] for x in results], marker='^', label="L2 Norm ↓")
plt.plot(labels, [x["kl"] for x in results], marker='d', label="KL Divergence ↓")
plt.xticks(rotation=30)
plt.xlabel("$(λ_1, λ_2, λ_3)$")
plt.ylabel("Metric Value")
plt.title("Ablation Study on $\lambda$ Weights")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("lambda_ablation_results.png")
plt.show()

print("\n✅ Ablation finished. Results saved to ablation_results.csv and lambda_ablation_results.png.")