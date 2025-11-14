import re
from numpy import mod
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from random import choice
from modeling_icae_multi_span import ICAE, ModelArguments, DataArguments, TrainingArguments
from transformers import HfArgumentParser
from peft import LoraConfig
from safetensors.torch import load_file
from typing import List, Dict, Optional, final
from utils import  generate_output_from_attacked_suffix, generate_output_from_decoder_memory
from dataloader.data_loader import FullMultiOutputDatasetWithTarget, FiveDemoDataset, KeywordDataset
from icae_recommend_keyword_dataset import ICAEEditRecommendationAttacker

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
    # keyword_dataset = KeywordDataset("path/to/Comattack_dataset/data_keywords_with_Qwen3.json")
    keyword_dataset = KeywordDataset("path/to/Comattack_dataset/revised_keywords_with_Qwen3.json")
    for idx, (sample, decrease_sample, increase_sample, keyword_sample) in tqdm(
        enumerate(zip(dataset, decrease_dataset, increase_dataset, keyword_dataset)),
        desc="Running ICAE Edit Recommendation Attacker"
    ):
        if idx == 28 or idx ==20:
            continue
        all_indices = list(range(5))
        best_idx = sample["best"]
        target_idx = sample["target"]

        # Collect the indices of best and target
        selected_indices = {best_idx, target_idx}

        # Select one more index that's not best or target
        remaining_indices = list(set(all_indices) - selected_indices)
        third_idx = choice(remaining_indices)
        print("\n[Third Index Selected]:", third_idx)

        selected_indices = [best_idx, target_idx, third_idx]  # final 3
        selected_indices.sort()  # To ensure consistent demo naming (demo_1, demo_2, demo_3)

        # Build new demo dict and remap index
        demos = {}
        index_map = {}
        for new_i, old_i in enumerate(selected_indices):
            demos[f"demo_{new_i + 1}"] = sample["demos"][old_i]
            index_map[old_i] = new_i

        # Remap best and target index to new range (0/1/2)
        new_best_index = index_map[best_idx]
        new_target_index = index_map[target_idx]
        demo_best_key = "demo_" + str(new_best_index + 1)
        demo_target_key = "demo_" + str(new_target_index + 1)

        question = sample["question"]
        requirement = str(sample["requirements"])

        names = ""
        pattern = r'^([^\u2013:]+)'
        for key, value in demos.items():
            match = re.search(pattern, value)
            name = match.group(1).strip()
            names += name + ", "
       
        if attack_mode == 'improve_target':
            modified_demo = increase_sample["demos"][target_idx]
            keywords = keyword_sample["demo_keywords"][target_idx]
        else:  # 'degrade_best'
            modified_demo = decrease_sample["demos"][best_idx]
            keywords = keyword_sample["demo_keywords"][best_idx]
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
            edit_weight=0.000001,
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
        prompt_text= question + "Please recommend for me.",
        max_out_length=12800,
        device="cuda"
        )

        print("\n[Final Answer]:", generated_answer)

