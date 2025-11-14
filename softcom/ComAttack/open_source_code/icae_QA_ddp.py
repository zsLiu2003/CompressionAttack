import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from modeling_icae_multi_span import ICAE, ModelArguments, DataArguments, TrainingArguments
from transformers import HfArgumentParser
from peft import LoraConfig
from safetensors.torch import load_file
from typing import List, Dict, Optional, final
from utils import generate_output_from_decoder_memory
from dataloader.dataloader_squad import SquadDataset
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset
import torch.distributed as dist
import os


class ICAEQAAttackerBase:
    def __init__(
        self,
        model,
        context: str,
        question: str,
        answer: Dict,
        keywords: List[str],
        attack_mode: str = "edit",  # or "suffix"
        num_steps: int = 300,
        suffix_len: int = 5,
        lr: float = 1e-2,
        edit_weight: float = 0.01,
        device: str = "cuda"
    ):
        assert attack_mode in ["edit", "suffix"], "attack_mode must be 'edit' or 'suffix'"

        self.model = model.to(device)
        self.tokenizer = model.module.tokenizer 
        self.device = device
        self.context = context
        self.question = question
        self.answer = answer
        self.keywords = keywords
        self.attack_mode = attack_mode
        self.num_steps = num_steps
        self.lr = lr
        self.edit_weight = edit_weight
        self.prefix_prompt = '''
        # ROLE
        You are a precise and meticulous text-analysis assistant.
        # RULES
        1.  Your ONLY source of information is the text provided below.
        2.  You MUST answer the question based ONLY on the provided text.
        3.  You are strictly FORBIDDEN from using any external knowledge or making any inferences that are not directly supported by the text.
        4.  If the answer is present in the text, provide it clearly and concisely.
        '''

        # Tokenize context
        self.original_ids = self.tokenizer(context, return_tensors='pt')['input_ids'][0].to(device)

        # Cache original compressed embedding
        with torch.no_grad():
            self.original_embedding = model.module._compress(self.original_ids.unsqueeze(0)).detach()

        if self.attack_mode == "edit":
            self.edit_positions = self._find_keyword_positions(self.original_ids)
            if not self.edit_positions: # 如果找不到关键词，则跳过
                print(f"Warning: Keywords {keywords} not found in context. Skipping attack.")
                self.valid = False
                return
            self.valid = True

            vocab_size = model.module.get_input_embeddings().weight.shape[0]
            self.token_logits = torch.zeros((len(self.edit_positions), vocab_size), requires_grad=True, device=device)
            self.optimizer = Adam([self.token_logits], lr=lr)

        elif self.attack_mode == "suffix":
            self.valid = True
            self.suffix_logits = torch.zeros((suffix_len, model.get_input_embeddings().weight.shape[1]), requires_grad=True, device=device)
            self.optimizer = Adam([self.suffix_logits], lr=lr)

    def _find_keyword_positions(self, input_ids):
        input_id_list = input_ids.tolist()
        edit_positions = []

        for kw in self.keywords:
            candidates = [kw, " " + kw]
            for variant in candidates:
                kw_ids = self.tokenizer(variant, add_special_tokens=False)['input_ids']
                if not kw_ids: continue
                for i in range(len(input_id_list) - len(kw_ids) + 1):
                    if input_id_list[i:i + len(kw_ids)] == kw_ids:
                        edit_positions.extend(list(range(i, i + len(kw_ids))))
                        # print(f"Keyword '{kw}' found at positions: {list(range(i, i + len(kw_ids)))}")
        return sorted(list(set(edit_positions)))

    def run(self):
        if not self.valid:
            return "Skipped (keywords not found)."
            
        # tqdm for rank 0 only to avoid messy output
        progress_bar = tqdm(range(self.num_steps), desc=f"GPU {self.device} Attacking", disable=(self.device != "cuda:0"))

        for step in progress_bar:
            self.optimizer.zero_grad()

            if self.attack_mode == "edit":
                input_ids = self.original_ids.clone()
                input_embeds = self.model.module.tokens_to_embeddings(input_ids.unsqueeze(0)).squeeze(0)

                token_probs = F.softmax(self.token_logits, dim=-1).to(self.model.module.icae.dtype)
                soft_embeds = token_probs @ self.model.module.get_input_embeddings().weight

                for j, pos in enumerate(self.edit_positions):
                    input_embeds[pos] = soft_embeds[j]

                memory = self.model.module._compress(inputs_embeds=input_embeds.unsqueeze(0))
                loss_main = -F.cosine_similarity(memory, self.original_embedding, dim=-1).mean()
                original_onehot = F.one_hot(self.original_ids[self.edit_positions], num_classes=token_probs.shape[-1]).float()
                edit_loss = F.kl_div(token_probs.log(), original_onehot, reduction='batchmean')
                loss = loss_main + self.edit_weight * edit_loss

            elif self.attack_mode == "suffix":
                suffix_embeds = self.suffix_logits.unsqueeze(0)
                original_embeds = self.model.module.tokens_to_embeddings(self.original_ids.unsqueeze(0))
                new_embeds = torch.cat([original_embeds, suffix_embeds], dim=1)
                memory = self.model.module._compress(inputs_embeds=new_embeds)
                loss = F.cosine_similarity(memory, self.original_embedding, dim=-1).mean()

            loss.backward()
            self.optimizer.step()

            if (step % 100 == 0 or step == self.num_steps - 1) and self.device == "cuda:0":
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Finalize attack and return answer
        edited_ids = self.finalize()
        memory_input = self.generate_decoder_input(edited_ids).to(self.device)

        adv_answer = generate_output_from_decoder_memory(
            model=self.model.module,
            decoder_memory_embeddings=memory_input,
            prompt_text=self.prefix_prompt + self.question,
            max_out_length=128,
            device=self.device
        )
        
        if self.device == "cuda:0":
             print(f"\n[GPU {self.device}] Question:", self.question)
             print(f"[GPU {self.device}] Adversarial Answer]:", adv_answer)
             print(f"[GPU {self.device}] Ground Truth Answer(s)]:", self.answer)
        return adv_answer

    def finalize(self):
        with torch.no_grad():
            if self.attack_mode == "edit":
                probs = F.softmax(self.token_logits, dim=-1)
                token_ids = torch.argmax(probs, dim=-1)
                edited_ids = self.original_ids.clone()
                for i, pos in enumerate(self.edit_positions):
                    edited_ids[pos] = token_ids[i]
                return edited_ids.tolist()

            elif self.attack_mode == "suffix":
                suffix_embeds = self.suffix_logits.unsqueeze(0)
                original_embeds = self.model.module.tokens_to_embeddings(self.original_ids.unsqueeze(0))
                new_embeds = torch.cat([original_embeds, suffix_embeds], dim=1)
                self._cached_suffix_embeds = suffix_embeds
                return self.original_ids.tolist()

    def generate_decoder_input(self, edited_token_ids: List[int]):
        if self.attack_mode == "edit":
            ids_tensor = torch.LongTensor([edited_token_ids]).to(self.device)
            memory = self.model.module._compress(ids_tensor)
        elif self.attack_mode == "suffix":
            original_embeds = self.model.module.tokens_to_embeddings(torch.LongTensor([edited_token_ids]).to(self.device))
            suffix_embeds = self._cached_suffix_embeds.to(self.device)
            full_embeds = torch.cat([original_embeds, suffix_embeds], dim=1)
            memory = self.model.module._compress(inputs_embeds=full_embeds)
        return memory.unsqueeze(0)


def setup():
    """Initialize distributed environment"""
    dist.init_process_group("nccl")
    #  Get current process rank and world_size
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank) 

def cleanup():
    """Clean up distributed environment"""
    dist.destroy_process_group()

def main():
    setup()

    # Get current process rank and world_size
    rank = dist.get_rank()
    
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    lora_config = LoraConfig(
        r=512, lora_alpha=32, lora_dropout=model_args.lora_dropout,
        bias="none", task_type="CAUSAL_LM"
    )

    # Each process loads its own model instance
    model = ICAE(model_args, training_args, lora_config)
    state_dict = load_file(training_args.output_dir)
    model.load_state_dict(state_dict, strict=False)

    # Move model to the GPU corresponding to the current process
    device = f"cuda:{rank}"
    model.to(device)

    # Wrap model with DDP
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # Load dataset
    raw_dataset = load_dataset("parquet", data_files={
        "train": "path/to/Comattack_dataset/squad/train-00000-of-00001.parquet",
        "validation": "path/to/Comattack_dataset/squad/validation-00000-of-00001.parquet"
    })
    val_dataset = SquadDataset(data=raw_dataset, split="validation")
    
    sampler = DistributedSampler(val_dataset, shuffle=False)
    
    val_loader = DataLoader(val_dataset, batch_size=1, sampler=sampler)

    for i, sample in enumerate(val_loader):
        # Limit to attack only the first few samples for demonstration
        if i >= 5: 
            break
            
        if rank == 0:
            print(f"\n===== Attacking sample {i+1} on GPU {rank} =====")
        
        # Attacker is created and run on each GPU
        # Note: Use model.module to access the original model before DDP wrapping
        attacker = ICAEQAAttackerBase(
            model=model,
            context=sample["context"][0],
            question=sample["question"][0],
            answer=sample["answers"]["answer_text"][0],
            attack_mode="edit",
            keywords=["Denver", "Broncos"], 
            device=device,
            num_steps=500
        )

        adv_answer = attacker.run()
        
    cleanup()


if __name__ == "__main__":
    # torchrun --nproc_per_node=2 your_script_ddp.py
    main()