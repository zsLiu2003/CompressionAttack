import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from transformers import AutoTokenizer
from auto_compressor import LlamaAutoCompressorModel
import sys
sys.path.append("")
from Comattack.open_source_code.dataloader.dataloader_squad import SquadDataset
from torch.utils.data import DataLoader
from datasets import load_dataset
from typing import Dict, List
from api_attack_edit_tokenlogits import EditTokenLogitAttacker
from api_attack_edit_perturb import EditTokenAttacker

class AutoCompressorQALogitAttacker(EditTokenLogitAttacker):
    def __init__(
        self,
        model,
        tokenizer,
        context: str,
        question: str,
        answer: Dict,
        keywords: List[str],
        num_steps: int = 100,
        lr: float = 5e-1,
        edit_weight: float = 0.01,
        device: str = "cuda"
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            context_text=context,
            prompt_text=question,
            keywords=keywords,
            num_steps=num_steps,
            lr=lr,
            edit_weight=edit_weight,
            device=device
        )
        self.answer = answer 
        self.question = question
        self.question_ids = tokenizer(question, return_tensors='pt').input_ids.to(device)

    def run(self):
        print("\n=== Starting QA Attack (Logit-Based Edit) ===")
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
                self.question_ids,
                softprompt=softprompt,
                max_new_tokens=64,
                do_sample=False
            )[0]

            answer_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            print("[Question]:", self.question)
            print("[Adversarial Answer]:", answer_text)
            print("[Ground Truth]:", self.answer)

        return answer_text

class AutoCompressorQAPerturbAttacker(EditTokenAttacker):
    def __init__(
        self,
        model,
        tokenizer,
        context: str,
        question: str,
        answer: Dict,
        keywords: List[str],
        num_steps: int = 100,
        lr: float = 5e-1,
        edit_weight: tuple = (0.001, 1e-5),  # (l2, kl)
        device: str = "cuda"
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            context_text=context,
            prompt_text=question,
            keywords=keywords,
            num_steps=num_steps,
            lr=lr,
            edit_weight=edit_weight,
            device=device
        )
        self.answer = answer 
        self.question = question
        self.question_ids = tokenizer(question, return_tensors='pt').input_ids.to(device)

    def run(self):
        print("\n=== Starting QA Attack (Perturbation-Based Edit) ===")
        super().run()  

        print("\n=== Generating Answer ===")
        with torch.no_grad():
            # Get input embeddings for context tokens
            edited_embeds = self.model.get_input_embeddings()(self.context_tokens).detach().clone()
            # Apply perturbations to specified positions
            for i, pos in enumerate(self.edit_positions):
                edited_embeds[pos] += self.perturb[i]

            # Generate softprompt with perturbed embeddings
            softprompt = self.model(
                inputs_embeds=edited_embeds.unsqueeze(0),
                segment_lengths=[self.context_length],
                output_softprompt=True
            ).softprompt

            output_ids = self.model.generate(
                self.question_ids,
                softprompt=softprompt,
                max_new_tokens=64,
                do_sample=False
            )[0]

            answer_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            print("[Question]:", self.question)
            print("[Adversarial Answer]:", answer_text)
            print("[Ground Truth]:", self.answer)

        return answer_text
    
if __name__ == "__main__":
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/AutoCompressor-Llama-2-7b-6k")
    model = LlamaAutoCompressorModel.from_pretrained("path/to/models/AutoCompressor-Llama-2-7b-6k", torch_dtype=torch.bfloat16).eval().cuda()

    raw_dataset = load_dataset("parquet",data_files={
        "train": "path/to/Comattack_dataset/squad/train-00000-of-00001.parquet",
        "validation": "path/to/Comattack_dataset/squad/validation-00000-of-00001.parquet"
    })
    val_dataset = SquadDataset(data=raw_dataset, split="validation")
    val_loader = DataLoader(val_dataset, batch_size=1)
    # sample = next(iter(val_loader))
    
    for i,sample in enumerate(val_loader):
        # if i<114:
        #     continue

        context=sample["context"][0]
        question = sample["question"][0]
        answer_texts = sample["answers"]["answer_text"]
        answer_starts = sample["answers"]["answer_start"]

        # Extract keywords for each answer
        keywords = []
        for text, start in zip(answer_texts, answer_starts):
            end = start + len(str(text[0]))
            keywords.extend(context[start:end].split())
        keywords = list(dict.fromkeys(keywords))
        print("Extracted Keywords:", keywords)

        # attacker = AutoCompressorQALogitAttacker(model, tokenizer, context, question, answer_texts, keywords)
        # edit_losses =  [
        #     (0.0, 0.0),
        #     (1e-3, 0.1),
        #     (1e-3, 0.01),
        #     (1e-5, 0.1),
        #     (1e-5, 0.001),
        #     (1e-4, 0.001),
        #     (1e-4, 0.01),
        #     (1e-4, 0.05),
        # ]
        # for edit_loss in edit_losses:
        attacker = AutoCompressorQAPerturbAttacker(model, tokenizer, context, question, answer_texts, keywords)
        attacker.run()