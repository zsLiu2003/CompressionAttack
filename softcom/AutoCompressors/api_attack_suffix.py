import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from auto_compressor import LlamaAutoCompressorModel
import os

# ------------------- Hyperparameter Configuration -------------------
MODEL_PATH = "path/to/models/AutoCompressor-Llama-2-7b-6k"
MODEL_NAME = "princeton-nlp/AutoCompressor-Llama-2-7b-6k"

suffix_length = 3           # Length of the perturbation to optimize
lr = 5e-1                      # Learning rate
n_steps = 100                  # Number of optimization steps
prompt_text = 'The first name of the current US president is "'
torch_dtype = torch.bfloat16
# --------------------------------------------------

# --- Device Setup ---
# Check for available CUDA device (GPU), otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Load Model and Tokenizer ---
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model
# We explicitly load the model to a single device to avoid multi-GPU conflicts.
# Do not use device_map="auto" for this kind of optimization script.
print("Loading model...")
model = LlamaAutoCompressorModel.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch_dtype
).eval().to(device)

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()
print(f"Model loaded to device: {model.device}")


# --- Prepare Tensors ---
# Prepare prompt and context, and move them to the model's device
prompt_tokens = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)

context_text = """Joe Biden, born in Scranton, Pennsylvania, on November 20, 1942, had a modest upbringing in a middle-class family. He attended the University of Delaware, where he double-majored in history and political science, graduating in 1965. Afterward, he earned his law degree from Syracuse University College of Law in 1968.\nBiden's early political career began in 1970 when he was elected to the New Castle County Council in Delaware. In 1972, tragedy struck when his wife Neilia and 1-year-old daughter Naomi were killed in a car accident, and his two sons, Beau and Hunter, were injured. Despite this devastating loss, Biden chose to honor his commitment and was sworn in as a senator by his sons' hospital bedsides.\nHe went on to serve as the United States Senator from Delaware for six terms, from 1973 to 2009. During his time in the Senate, Biden was involved in various committees and was particularly known for his expertise in foreign affairs, serving as the chairman of the Senate Foreign Relations Committee on multiple occasions.\nIn 2008, Joe Biden was selected as the running mate for Barack Obama, who went on to win the presidential election. As Vice President, Biden played an integral role in the Obama administration, helping to shape policies and handling issues such as economic recovery, foreign relations, and the implementation of the Affordable Care Act (ACA), commonly known as Obamacare.\nAfter completing two terms as Vice President, Joe Biden decided to run for the presidency in 2020. He secured the Democratic nomination and faced the incumbent President Donald Trump in the general election. Biden campaigned on a platform of unity, promising to heal the divisions in the country and tackle pressing issues, including the COVID-19 pandemic, climate change, racial justice, and economic inequality.\nIn the November 2020 election, Biden emerged victorious, and on January 20, 2021, he was inaugurated as the 46th President of the United States. At the age of 78, Biden became the oldest person to assume the presidency in American history.\nAs President, Joe Biden has worked to implement his agenda, focusing on various initiatives, such as infrastructure investment, climate action, immigration reform, and expanding access to healthcare. He has emphasized the importance of diplomacy in international relations and has sought to rebuild alliances with global partners.\nThroughout his long career in public service, Joe Biden has been recognized for his commitment to bipartisanship, empathy, and his dedication to working-class issues. He continues to navigate the challenges facing the nation, striving to bring the country together and create a positive change for all Americans."""
context_tokens = tokenizer(context_text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
context_length = context_tokens.size(1)

# Get the word embedding matrix
embedding_matrix = model.get_input_embeddings().weight
embed_dim = embedding_matrix.size(1)

# --- Get Original Softprompt (Target) ---
# Calculate the original softprompt without attack, which serves as our optimization target
with torch.no_grad():
    context_embed = model.get_input_embeddings()(context_tokens)
    dummy_suffix = torch.zeros((1, suffix_length, embed_dim), device=model.device, dtype=torch_dtype)
    dummy_input = torch.cat([context_embed, dummy_suffix], dim=1)
    dummy_seg_lengths = [context_length, suffix_length]
    softprompt = model(context_tokens, output_softprompt=True).softprompt
    original_softprompt = model(inputs_embeds=dummy_input, output_softprompt=True, segment_lengths=dummy_seg_lengths).softprompt

# --- Optimization Setup ---
# Initialize the trainable perturbation (suffix) and place it on the correct device
suffix_embeds = torch.randn((1, suffix_length, embed_dim), requires_grad=True, device=model.device, dtype=torch_dtype)

# Define the optimizer, which only optimizes the suffix_embeds we created
optimizer = torch.optim.Adam([suffix_embeds], lr=lr)

# --- Attack Process ---
print("\nStarting attack optimization...")
for step in range(n_steps):
    # Get the context embeddings
    context_embed = model.get_input_embeddings()(context_tokens)
    
    # Concatenate the context embeddings with the trainable suffix embeddings
    attacked_input_embed = torch.cat([context_embed, suffix_embeds], dim=1)
    
    # Define segment lengths
    segment_lengths = [context_length, suffix_length]
    
    # Forward pass to generate the attacked softprompt
    summary_vec = model(
        inputs_embeds=attacked_input_embed,
        segment_lengths=segment_lengths,
        output_softprompt=True
    ).softprompt
    
    # Calculate the loss function: we want the attacked softprompt to be as dissimilar to the original as possible.
    # Therefore, we use the cosine similarity as the loss.
    cos_sim = F.cosine_similarity(original_softprompt.flatten(), summary_vec.flatten(), dim=0)
    # === L2 Norm on suffix to restrict magnitude ===
    edit_loss_L2 = torch.norm(suffix_embeds, p=2)

    # === MSE Loss on softprompt ===
    edit_loss_mse = F.mse_loss(summary_vec, original_softprompt)

    # === KL Divergence between softprompt distributions ===
    # You can treat softprompt as a distribution over d-dim
    orig_soft = F.softmax(original_softprompt, dim=-1)
    attk_soft = F.log_softmax(summary_vec, dim=-1)
    edit_loss_kl = F.kl_div(attk_soft, orig_soft, reduction="batchmean")

    print(f"[Step {step}] Cosine similarity: {cos_sim.item():.6f}, L2 Loss: {edit_loss_L2.item():.6f}, MSE Loss: {edit_loss_mse.item():.6f}, KL Loss: {edit_loss_kl.item():.6f}")
    # === Final loss combination ===
    λ1, λ2, λ3 = 0.001, 1e-5, 0.01 # you can tune them
    loss = cos_sim + λ1 * edit_loss_L2 + λ2 * edit_loss_kl + λ3 * edit_loss_mse
    # loss = cos_sim
    
    # Standard optimization steps
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"[Step {step}] Cosine similarity: {cos_sim.item():.6f}, Loss: {loss.item():.6f}")

# --- Generate Final Outputs ---
print("\n--- Generating final outputs ---")
with torch.no_grad():
    # Use the optimized suffix_embeds to calculate the final attacked softprompt
    context_embed = model.get_input_embeddings()(context_tokens)
    attacked_input_embed = torch.cat([context_embed, suffix_embeds], dim=1)
    segment_lengths = [context_length, suffix_length]
    
    attacked_summary = model(
        inputs_embeds=attacked_input_embed,
        segment_lengths=segment_lengths,
        output_softprompt=True
    ).softprompt
    
    # Generate text using the attacked softprompt
    gen = model.generate(
        prompt_tokens,
        softprompt=attacked_summary,
        max_new_tokens=12,
        do_sample=False
    )[0]
    print("\n[Attacked Generation]")
    print(tokenizer.decode(gen))

# --- Original Generation for Comparison ---
# Generate the original, unattacked result for comparison
with torch.no_grad():
    orig_gen = model.generate(
        prompt_tokens,
        softprompt=softprompt,
        max_new_tokens=12,
        do_sample=False
    )[0]
    print("\n[Original Generation]")
    print(tokenizer.decode(orig_gen))

print("\nScript finished.")
