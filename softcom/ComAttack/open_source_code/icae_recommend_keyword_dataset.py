import re
from aiohttp import content_disposition_filename
from numpy import mod
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from modeling_icae_multi_span import ICAE, ModelArguments, DataArguments, TrainingArguments
from transformers import HfArgumentParser
from peft import LoraConfig
from safetensors.torch import load_file
from typing import List, Dict, Optional, final
from utils import  generate_output_from_attacked_suffix, generate_output_from_decoder_memory
from dataloader.data_loader import FullMultiOutputDatasetWithTarget, FiveDemoDataset, KeywordDataset
import logging

class ICAEEditRecommendationAttacker:
    def __init__(
        self,
        model,
        demos: Dict[str, str],
        target_demo_key: str,
        keywords: List[str],
        best_demo_key: str = None,
        attack_mode: str = "improve_target",  # or "degrade_best"
        modified_demo: Optional[str] = None,
        num_steps: int = 500,
        lr: float = 1e-2,
        edit_weight: float = 0.01,
        device: str = "cuda"
    ):
        assert attack_mode in ["improve_target", "degrade_best"], "Invalid attack_mode"

        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = model.tokenizer
        self.device = device
        self.lr = lr
        self.num_steps = num_steps
        self.edit_weight = edit_weight
        self.attack_mode = attack_mode

        self.demos = demos
        self.target_demo_key = target_demo_key
        self.best_demo_key = best_demo_key
        self.keywords = keywords
        self.modified_demo = modified_demo
        self.modified_ids = self.tokenizer(self.modified_demo, return_tensors='pt')['input_ids'][0].to(device)
        self.modified_embedding = model._compress(self.modified_ids.unsqueeze(0)).detach() if modified_demo else None

        self.target_text = demos[target_demo_key]
        self.target_ids = self.tokenizer(self.target_text, return_tensors='pt')['input_ids'][0].to(device)

        if attack_mode == "improve_target":
            self.edit_positions = self._find_keyword_positions(self.target_ids)
        else:  # degrade_best
            self.edit_positions = self._find_keyword_positions(self.tokenizer(demos[best_demo_key], return_tensors='pt')['input_ids'][0])
        # assert self.edit_positions, f"No keywords {keywords} found in demo text."
        if not self.edit_positions:
            error_message = f"No keywords found in demo text for attack_mode: {attack_mode}. Keywords: {keywords if 'keywords' in locals() else 'N/A'}"
            raise ValueError(error_message)

        with torch.no_grad():
            self.original_embedding = model._compress(self.target_ids.unsqueeze(0)).detach()
            if attack_mode == "degrade_best":
                self.best_ids = self.tokenizer(demos[best_demo_key], return_tensors='pt')['input_ids'][0].to(device)
                self.best_ids_cpu = self.tokenizer(demos[best_demo_key], return_tensors='pt')['input_ids'][0] 
                self.best_embedding = model._compress(self.best_ids.unsqueeze(0)).detach()

        vocab_size = model.get_input_embeddings().weight.shape[0]
        self.token_logits = torch.zeros((len(self.edit_positions), vocab_size), requires_grad=True, device=device)
        self.optimizer = Adam([self.token_logits], lr=lr)

    def _find_keyword_positions(self, input_ids):
        input_id_list = input_ids.tolist()
        tokenized_input = self.tokenizer.convert_ids_to_tokens(input_id_list)
        # print(f"Tokenized input: {tokenized_input}")
        edit_positions = []

        for kw in self.keywords:
            # imitate tokenizer 
            candidates = [kw, " " + kw]

            found = False
            for variant in candidates:
                kw_ids = self.tokenizer(variant, add_special_tokens=False)['input_ids']
                for i in range(len(input_id_list) - len(kw_ids) + 1):
                    if input_id_list[i:i + len(kw_ids)] == kw_ids:
                        edit_positions.extend(list(range(i, i + len(kw_ids))))
                        found = True
                        print(f"Keyword '{kw}' (matched as '{variant}') found at positions: {list(range(i, i + len(kw_ids)))}")
            if not found:
                print(f"[Warning] Keyword '{kw}' not found in tokenized input.")
    
        return sorted(list(set(edit_positions)))    

    def run(self):
        for step in tqdm(range(self.num_steps)):
            self.optimizer.zero_grad()

            if self.attack_mode == "improve_target":
                input_ids = self.target_ids.clone()
            else:  # degrade_best
                input_ids = self.best_ids.clone()
            input_embeds = self.model.tokens_to_embeddings(input_ids.unsqueeze(0)).squeeze(0)
            # print("[Input embeds shape]:", input_embeds.shape)

            token_probs = F.softmax(self.token_logits, dim=-1).to(self.model.icae.dtype)
            # print(self.tokenizer.convert_ids_to_tokens(torch.argmax(token_probs, dim=-1).tolist()))
            soft_embeds = token_probs @ self.model.get_input_embeddings().weight  # [E, D]

            for j, pos in enumerate(self.edit_positions):
                input_embeds[pos] = soft_embeds[j]

            memory = self.model._compress(inputs_embeds=input_embeds.unsqueeze(0))
            
             
            # if self.attack_mode == "improve_target":
            #     loss_main = -F.cosine_similarity(memory, self.modified_embedding, dim=-1).mean()
            # else:
            #     loss_main = -F.cosine_similarity(memory, self.best_embedding, dim=-1).mean()
            loss_main = -F.cosine_similarity(memory, self.modified_embedding, dim=-1).mean()

            if self.attack_mode == "improve_target":
                original_onehot = F.one_hot(self.target_ids[self.edit_positions], num_classes=token_probs.shape[-1]).float()
            else:  # degrade_best
                original_onehot = F.one_hot(self.best_ids[self.edit_positions], num_classes=token_probs.shape[-1]).float()
            edit_loss = F.kl_div(token_probs.log(), original_onehot, reduction='batchmean')

            loss = loss_main + self.edit_weight * edit_loss
            loss.backward()
            self.optimizer.step()

            if step % 50 == 0 or step == self.num_steps - 1:
                print(f"[Step {step}] Loss: {loss.item():.4f}, CosSim: {-loss_main.item():.4f}")
            if step % 50 == 0:
                topk_probs, topk_ids = torch.topk(token_probs, 3, dim=-1)
                for i, pos in enumerate(self.edit_positions):
                    top_tokens = self.tokenizer.convert_ids_to_tokens(topk_ids[i].tolist())
                    # print(f"[Step {step}] Pos {pos} top-3: {list(zip(top_tokens, topk_probs[i].tolist()))}")

        return self.finalize()

    def finalize(self):
        with torch.no_grad():
            probs = F.softmax(self.token_logits, dim=-1)
            token_ids = torch.argmax(probs, dim=-1)
            if self.attack_mode == "improve_target":
                edited_input_ids = self.target_ids.clone()
            else:  # degrade_best
                edited_input_ids = self.best_ids.clone()
            for i, pos in enumerate(self.edit_positions):
                edited_input_ids[pos] = token_ids[i]

            if self.attack_mode == "improve_target":
                original_tokens = self.tokenizer.convert_ids_to_tokens(self.target_ids[self.edit_positions])
            else:  # degrade_best
                original_tokens = self.tokenizer.convert_ids_to_tokens(self.best_ids[self.edit_positions])
            edited_tokens = self.tokenizer.convert_ids_to_tokens(token_ids.tolist())

            print("\n[Keyword Edits]")
            for i, pos in enumerate(self.edit_positions):
                print(f" - Position {pos}: {original_tokens[i]} -> {edited_tokens[i]}")

            final_text = self.tokenizer.decode(edited_input_ids, skip_special_tokens=True)
            print("\n[Final Edited Text]")
            print(final_text)
            return final_text, edited_input_ids.tolist()
        
    def generate_decoder_input(self, edited_token_ids: List[int]) -> torch.Tensor:
        """
        Compress all demos (replace the edited demo with its modified version) 
        and return concatenated memory. For 'improve_target' mode, replace the target demo;
        for 'degrade_best' mode, replace the best demo.
        Also prints token length for each demo and total.
        """
        compressed_memories = []
        total_token_count = 0

        print("\n[Token count per demo]")
        for key, text in self.demos.items():
            should_replace = (
                (self.attack_mode == "improve_target" and key == self.target_demo_key) or
                (self.attack_mode == "degrade_best" and key == self.best_demo_key)
            )

            if should_replace:
                edited_ids = torch.LongTensor([edited_token_ids]).to(self.device)
                token_len = edited_ids.shape[1]
                memory = self.model._compress(edited_ids)
                print(f" - {key} (edited): {token_len} tokens")
            else:
                ids = self.tokenizer(text, truncation=True, return_tensors='pt')['input_ids'].to(self.device)
                token_len = ids.shape[1]
                memory = self.model._compress(ids)
                print(f" - {key}: {token_len} tokens")

            total_token_count += token_len
            compressed_memories.append(memory)

        print(f"[Total token count for all 3 demos]: {total_token_count} tokens\n")

        return torch.cat(compressed_memories, dim=0).unsqueeze(0)
        
if __name__ == "__main__":
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
    attack_mode = training_args.attack_mode  # 'improve_target' or 'degrade_best'
    print(f"[Attack Mode]: {attack_mode}")

    dataset = FullMultiOutputDatasetWithTarget("path/to/Comattack_dataset/data_best_Qwen3.json")
    decrease_dataset = FiveDemoDataset("path/to/Comattack_dataset/keywords_decrease_filtered.json")
    increase_dataset = FiveDemoDataset("path/to/Comattack_dataset/keywords_increase_filtered.json")
    keyword_dataset = KeywordDataset("path/to/Comattack_dataset/data_keywords_with_Qwen3.json")
    for sample, decrease_sample, increase_sample, keyword_sample in tqdm(zip(dataset, decrease_dataset, increase_dataset, keyword_dataset), desc="Running ICAE Edit Recommendation Attacker"):
        try:
            # demos = sample["demos"]
            demos = {
            "demo_1": sample["demos"][0],
            "demo_2": sample["demos"][1],
            "demo_3": sample["demos"][2],
            "demo_4": sample["demos"][3],
            "demo_5": sample["demos"][4]
            }
            best_key = sample["best"]
            target_key = sample["target"]
            demo_best_key = "demo_" + str(sample["best"]+1)  # best is 0-indexed, demo keys are 1-indexed
            demo_target_key = "demo_" + str(sample["target"]+1)  # target is 0-indexed, demo keys are 1-indexed
            print(f" - Best Demo: {demos[demo_best_key]}")
            print(f" - Target Demo: {demos[demo_target_key]}")
            question = sample["question"]
            requirement = str(sample["requirements"])

            names = ""
            pattern = r'^([^\u2013:]+)'
            for key, value in demos.items():
                match = re.search(pattern, value)
                name = match.group(1).strip()
                names += name + ", "
            
            # print("target key: ", target_key)
            # print("increase sample",increase_sample["demos"])
            try:
                if attack_mode == 'improve_target':
                    modified_demo = increase_sample["demos"][target_key]
                    keywords = keyword_sample["demo_keywords"][target_key]
                else: # 'degrade_best'
                    modified_demo = decrease_sample["demos"][best_key]
                    keywords = keyword_sample["demo_keywords"][best_key]
            except IndexError as e:
                print('key not found! Ignored')
                continue
            # print(f"\n[Modified Demo]: {modified_demo}")

            attacker = ICAEEditRecommendationAttacker(
                model=model,
                attack_mode=attack_mode,
                best_demo_key=demo_best_key,
                target_demo_key=demo_target_key,
                keywords=keywords,
                modified_demo=modified_demo,
                num_steps=100,
                lr=1e-2,
                edit_weight=0.0001,
                device="cuda",
                demos=demos,
            )

            edited_text, edited_token_ids = attacker.run()
            with torch.no_grad():
                final_memory_input = attacker.generate_decoder_input(edited_token_ids)

            print("\n[Final memory input shape]:", final_memory_input.shape)

            final_memory_input = final_memory_input.to("cuda")
            generated_answer = generate_output_from_decoder_memory(
            model=model,
            decoder_memory_embeddings=final_memory_input,
            prompt_text=question + "Which product would you recommend?' ",
            # prompt_text= question + "Products: " + names + "\n\n" + "Requirements: " + requirement,
            max_out_length=12800,
            device="cuda"
            )

            print("\n[Final Answer]:", generated_answer)
            
        except (IndexError, ValueError) as e:
            print(
                f"Skipping sample due to an error: {e}. "
                f"Problematic sample info - target_key: {target_key if 'target_key' in locals() else 'N/A'}, "
                f"best_key: {best_key if 'best_key' in locals() else 'N/A'}, "
                f"attack_mode: {attack_mode}."
            )
            continue

        # TODO: calculate Attack Success Rate (ASR) 
        # if attack_mode == "improve_target":
        #     if generated_answer == re.search(pattern, sample["demos"][target_key]).group(1).strip():
        #         print("[ASR] Success")
        #     else:
        #         print("[ASR] Failed")