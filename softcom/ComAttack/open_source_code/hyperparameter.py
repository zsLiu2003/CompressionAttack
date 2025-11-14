from icae_recommend_keyword_dataset import ICAEEditRecommendationAttacker
from transformers import HfArgumentParser, TrainingArguments
from peft import LoraConfig
from modeling_icae_multi_span import ICAE, ModelArguments, DataArguments
from safetensors.torch import load_file
import torch
from utils import generate_output_from_decoder_memory

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