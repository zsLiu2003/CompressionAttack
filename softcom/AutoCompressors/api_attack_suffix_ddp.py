import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from auto_compressor import LlamaAutoCompressorModel
import torch.distributed as dist
from torch.cuda.amp import autocast
import os
import functools

# --- FSDP Imports ---
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import CPUOffload

# IMPORTANT: You might need to find the correct Decoder Layer class from your model's source code.
# It is often found in a file like `modeling_llama.py`.
# If the import below fails, please locate the correct class and update the path.
from modeling_flash_llama import LlamaDecoderLayer


def setup_ddp(backend="nccl"):
    """
    Initializes the distributed environment. FSDP uses the same setup.
    """
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    
    # FSDP will manage device placement, but setting the current device is good practice.
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        print(f"Initialized {world_size} processes for FSDP.")
    print(f"[Process {rank}] Using device: {device}")
    
    return rank, local_rank, world_size, device

# --- Distributed Setup ---
rank, local_rank, world_size, device = setup_ddp()

# ------------------- Hyperparameter Configuration -------------------
MODEL_PATH = "path/to/models/AutoCompressor-Llama-2-7b-6k"
MODEL_NAME = "princeton-nlp/AutoCompressor-Llama-2-7b-6k"

suffix_length = 3
lr = 5e-2
n_steps = 100
prompt_text = 'The first name of the current US president is "'
torch_dtype = torch.bfloat16
# --------------------------------------------------

# Load tokenizer (same for all processes)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# --- Model Loading for FSDP ---
# With FSDP, we load the model on the CPU. FSDP will handle moving shards to the GPU.
if rank == 0:
    print("Loading model onto CPU...")
# The model stays on the CPU. Do not call .to(device) or use device_map.
model = LlamaAutoCompressorModel.from_pretrained(MODEL_PATH, torch_dtype=torch_dtype, device_map="cpu").eval()

# Enable gradient checkpointing for memory savings.
# This must be done BEFORE wrapping with FSDP.
model.gradient_checkpointing_enable()

# --- FSDP Wrapping ---
# Define the wrapping policy. This tells FSDP how to break the model into shards.
# We provide the specific class for the transformer blocks.
transformer_layer_cls = { LlamaDecoderLayer }
auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls=transformer_layer_cls,
)

# Initialize FSDP
# cpu_offload=CPUOffload(offload_params=True) is key to solving OOM. It moves model params to CPU when not in use.
model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    cpu_offload=CPUOffload(offload_params=True),
    device_id=local_rank, # Use local_rank for the device_id
)
if rank == 0:
    print("Model wrapped with FSDP and CPU Offloading.")

# --- Prepare Tensors ---
# Other tensors are still moved to the GPU for computation.
prompt_tokens = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
context_text = """Joe Biden, born in Scranton, Pennsylvania, on November 20, 1942, had a modest upbringing in a middle-class family. He attended the University of Delaware, where he double-majored in history and political science, graduating in 1965. Afterward, he earned his law degree from Syracuse University College of Law in 1968.\nBiden's early political career began in 1970 when he was elected to the New Castle County Council in Delaware. In 1972, tragedy struck when his wife Neilia and 1-year-old daughter Naomi were killed in a car accident, and his two sons, Beau and Hunter, were injured. Despite this devastating loss, Biden chose to honor his commitment and was sworn in as a senator by his sons' hospital bedsides.\nHe went on to serve as the United States Senator from Delaware for six terms, from 1973 to 2009. During his time in the Senate, Biden was involved in various committees and was particularly known for his expertise in foreign affairs, serving as the chairman of the Senate Foreign Relations Committee on multiple occasions.\nIn 2008, Joe Biden was selected as the running mate for Barack Obama, who went on to win the presidential election. As Vice President, Biden played an integral role in the Obama administration, helping to shape policies and handling issues such as economic recovery, foreign relations, and the implementation of the Affordable Care Act (ACA), commonly known as Obamacare.\nAfter completing two terms as Vice President, Joe Biden decided to run for the presidency in 2020. He secured the Democratic nomination and faced the incumbent President Donald Trump in the general election. Biden campaigned on a platform of unity, promising to heal the divisions in the country and tackle pressing issues, including the COVID-19 pandemic, climate change, racial justice, and economic inequality.\nIn the November 2020 election, Biden emerged victorious, and on January 20, 2021, he was inaugurated as the 46th President of the United States. At the age of 78, Biden became the oldest person to assume the presidency in American history.\nAs President, Joe Biden has worked to implement his agenda, focusing on various initiatives, such as infrastructure investment, climate action, immigration reform, and expanding access to healthcare. He has emphasized the importance of diplomacy in international relations and has sought to rebuild alliances with global partners.\nThroughout his long career in public service, Joe Biden has been recognized for his commitment to bipartisanship, empathy, and his dedication to working-class issues. He continues to navigate the challenges facing the nation, striving to bring the country together and create a positive change for all Americans."""
context_tokens = tokenizer(context_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
context_length = context_tokens.size(1)

# Get embed_dim safely by calculating on rank 0 and broadcasting.
embed_dim_tensor = torch.zeros(1, dtype=torch.long, device=device)
if rank == 0:
    with FSDP.summon_full_params(model, writeback=False, rank0_only=True) as full_model:
        embed_dim_tensor[0] = full_model.get_input_embeddings().weight.size(1)
dist.broadcast(embed_dim_tensor, src=0)
embed_dim = embed_dim_tensor.item()

# Get original softprompt (target)
with torch.no_grad():
    with autocast(dtype=torch_dtype):
        # Create a placeholder tensor for the embeddings on all ranks.
        context_embed_shape = (context_tokens.shape[0], context_tokens.shape[1], embed_dim)
        context_embed = torch.empty(context_embed_shape, device=device, dtype=torch_dtype)
        
        # Calculate the actual embeddings only on rank 0.
        if rank == 0:
            with FSDP.summon_full_params(model, writeback=False, rank0_only=True) as full_model:
                 context_embed.copy_(full_model.get_input_embeddings()(context_tokens))
        
        # Broadcast the embeddings from rank 0 to all other ranks.
        dist.broadcast(context_embed, src=0)
        
        # Now all ranks have the correct context_embed.
        dummy_suffix = torch.zeros((1, suffix_length, embed_dim), device=device, dtype=torch_dtype)
        dummy_input = torch.cat([context_embed, dummy_suffix], dim=1)
        dummy_seg_lengths = [context_length, suffix_length]
        original_softprompt = model(inputs_embeds=dummy_input, output_softprompt=True, segment_lengths=dummy_seg_lengths).softprompt

# Initialize perturbation and optimizer
suffix_embeds = torch.randn((1, suffix_length, embed_dim), requires_grad=True, device=device, dtype=torch_dtype)
optimizer = torch.optim.Adam([suffix_embeds], lr=lr)

# --- Attack Process ---
if rank == 0:
    print("\nStarting attack optimization with FSDP (bfloat16)...")

for step in range(n_steps):
    with autocast(dtype=torch_dtype):
        # Use the same broadcast pattern inside the loop.
        context_embed_shape = (context_tokens.shape[0], context_tokens.shape[1], embed_dim)
        context_embed = torch.empty(context_embed_shape, device=device, dtype=torch_dtype)
        if rank == 0:
            with FSDP.summon_full_params(model, writeback=False, rank0_only=True) as full_model:
                context_embed.copy_(full_model.get_input_embeddings()(context_tokens))
        dist.broadcast(context_embed, src=0)

        attacked_input_embed = torch.cat([context_embed, suffix_embeds], dim=1)
        segment_lengths = [context_length, suffix_length]
        
        summary_vec = model(
            inputs_embeds=attacked_input_embed,
            segment_lengths=segment_lengths,
            output_softprompt=True
        ).softprompt
        
        cos_sim = F.cosine_similarity(original_softprompt.flatten(), summary_vec.flatten(), dim=0)
        loss = -cos_sim

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if rank == 0 and step % 10 == 0:
        print(f"[Step {step}] Cosine similarity: {cos_sim.item():.6f}, Loss: {loss.item():.6f}")

# --- Generation should only be done on the main process (rank 0) ---
if rank == 0:
    print("\n--- Generating final outputs on rank 0 ---")
    with torch.no_grad():
        with autocast(dtype=torch_dtype):
            # Use rank0_only=True since we are already inside an `if rank == 0` block.
            with FSDP.summon_full_params(model, writeback=False, rank0_only=True) as full_model:
                context_embed = full_model.get_input_embeddings()(context_tokens)
                attacked_input_embed = torch.cat([context_embed, suffix_embeds], dim=1)
                segment_lengths = [context_length, suffix_length]
                
                attacked_summary = full_model(
                    inputs_embeds=attacked_input_embed,
                    segment_lengths=segment_lengths,
                    output_softprompt=True
                ).softprompt
                
                gen = full_model.generate(
                    prompt_tokens,
                    softprompt=attacked_summary,
                    max_new_tokens=12,
                    do_sample=False
                )[0]
                print("\n[Attacked Generation]")
                print(tokenizer.decode(gen))

                orig_gen = full_model.generate(
                    prompt_tokens,
                    softprompt=original_softprompt,
                    max_new_tokens=12,
                    do_sample=False
                )[0]
                print("\n[Original Generation]")
                print(tokenizer.decode(orig_gen))

# Clean up the process group
dist.destroy_process_group()
