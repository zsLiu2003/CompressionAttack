import torch
import torch.nn.functional as F
from torch.optim import Adam
from transformers import AutoTokenizer
from typing import Dict, List, Optional
from api_attack_edit_tokenlogits import EditTokenLogitAttacker 
from transformers import AutoTokenizer
from auto_compressor import LlamaAutoCompressorModel
import re
from tqdm import tqdm
import sys
sys.path.append("")
from Comattack.open_source_code.dataloader.data_loader import FullMultiOutputDatasetWithTarget, FiveDemoDataset, KeywordDataset


class AutoCompressorRecommendationLogitAttacker(EditTokenLogitAttacker):
    def __init__(
        self,
        model,
        tokenizer,
        demos: Dict[str, str],
        prompt_text: str,
        target_demo_key: str,
        keywords: List[str],
        best_demo_key: Optional[str] = None,
        modified_demo: Optional[str] = None,
        attack_mode: str = "improve_target",
        num_steps: int = 100,
        lr: float = 1e-2,
        edit_weight: float = 0.01,
        device: str = "cuda"
    ):
        demo_text = demos[target_demo_key] if attack_mode == "improve_target" else demos[best_demo_key]
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            context_text=demo_text,
            prompt_text=prompt_text,
            keywords=keywords,
            num_steps=num_steps,
            lr=lr,
            edit_weight=edit_weight,
            device=device
        )
        self.demos = demos
        self.target_demo_key = target_demo_key
        self.best_demo_key = best_demo_key
        self.attack_mode = attack_mode
        self.modified_demo = modified_demo

        self.target_ids = tokenizer(demos[target_demo_key], return_tensors='pt')['input_ids'][0].to(device)
        self.best_ids = tokenizer(demos[best_demo_key], return_tensors='pt')['input_ids'][0].to(device)

        if attack_mode == "improve_target":
            self.original_ids = self.target_ids
        else:
            self.original_ids = self.best_ids

        with torch.no_grad():
            if modified_demo:
                modified_ids = tokenizer(modified_demo, return_tensors='pt')['input_ids'][0].to(device)

    def run(self):
        print("\n=== Starting Recommendation Attack (Logit-Based Edit) ===")
        super().run()
        print("\n=== Generating Answer ===")
        with torch.no_grad():
            probs = F.softmax(self.token_logits, dim=-1)
            token_ids = torch.argmax(probs, dim=-1)
            edited_ids = self.context_ids.clone()
            for i, pos in enumerate(self.edit_positions):
                edited_ids[pos] = token_ids[i]

            edited_embeds = self.model.get_input_embeddings()(edited_ids).unsqueeze(0)
            softprompt = self.model(
                inputs_embeds=edited_embeds,
                segment_lengths=[edited_ids.shape[0]],
                output_softprompt=True
            ).softprompt

            output_ids = self.model.generate(
                self.prompt_ids,
                softprompt=softprompt,
                max_new_tokens=64,
                do_sample=False
            )[0]

            answer_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            print("[Question]:", self.prompt_text)
            print("[Adversarial Answer]:", answer_text)
            # print("[Ground Truth]:", self.answer)
        return self.finalize()

    def finalize(self):
        with torch.no_grad():
            probs = F.softmax(self.token_logits, dim=-1)
            token_ids = torch.argmax(probs, dim=-1)
            edited_ids = self.context_ids.clone()
            for i, pos in enumerate(self.edit_positions):
                edited_ids[pos] = token_ids[i]

            final_text = self.tokenizer.decode(edited_ids, skip_special_tokens=True)
            print("\n[Edited Demo Text]")
            print(final_text)
            return final_text, edited_ids.tolist()

if __name__ == "__main__":
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/AutoCompressor-Llama-2-7b-6k")
    model = LlamaAutoCompressorModel.from_pretrained("path/to/models/AutoCompressor-Llama-2-7b-6k", torch_dtype=torch.bfloat16).eval().cuda()
    attack_mode = 'improve_target'  # or 'degrade_best'
    
    dataset = FullMultiOutputDatasetWithTarget("path/to/Comattack_dataset/data_best_Qwen3.json")
    decrease_dataset = FiveDemoDataset("path/to/Comattack_dataset/keywords_decrease.json")
    increase_dataset = FiveDemoDataset("path/to/Comattack_dataset/keywords_increase.json")
    keyword_dataset = KeywordDataset("path/to/Comattack_dataset/data_keywords_with_Qwen3.json")
    i=0
    for sample, decrease_sample, increase_sample, keyword_sample in tqdm(zip(dataset, decrease_dataset, increase_dataset, keyword_dataset), desc="Running ICAE Edit Recommendation Attacker"):
        # demos = sample["demos"]
        print(f"Sample {i}:")
        i += 1
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
        question = sample["question"]
        requirement = str(sample["requirements"])

        names = ""
        pattern = r'^([^\u2013:]+)'
        for key, value in demos.items():
            match = re.search(pattern, value)
            name = match.group(1).strip()
            names += name + ", "
       
        if attack_mode == 'improve_target':
            modified_demo = increase_sample["demos"][target_key]
            keywords = keyword_sample["demo_keywords"][target_key]
        else: # 'degrade_best'
            modified_demo = decrease_sample["demos"][best_key]
            keywords = keyword_sample["demo_keywords"][best_key]
    
        attacker = AutoCompressorRecommendationLogitAttacker(
            model=model,
            tokenizer=tokenizer,
            demos=demos,
            prompt_text=question + "Which product would you recommend?' ",
            target_demo_key=demo_target_key,
            best_demo_key=demo_best_key,
            keywords=keywords,
            modified_demo=modified_demo,
            attack_mode=attack_mode,
            num_steps=100,
            lr=1e-2,
            edit_weight=0.01,
            device="cuda"
        )

        edited_text, edited_token_ids = attacker.run()