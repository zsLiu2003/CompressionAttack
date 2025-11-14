import torch
import torch.nn.functional as F
from typing import List, Union
from tqdm import tqdm
from transformers import AutoTokenizer
from auto_compressor import LlamaAutoCompressorModel
from transformers import XLMRobertaTokenizer, XLMRobertaForMaskedLM, BertTokenizer, BertForCausalLM
import math
import re

class EditTokenAttacker:
    def __init__(
        self,
        model,
        tokenizer,
        context_text: str,
        prompt_text: str,
        keywords: List[str],
        num_steps: int = 100,
        lr: float = 5e-1,
        edit_weight: tuple = (1e-4, 0.001),  # (l2, kl)
        device: str = "cuda"
    ):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.context_text = context_text
        self.prompt_text = prompt_text
        self.keywords = keywords
        self.num_steps = num_steps
        self.lr = lr
        self.mu = edit_weight[0]  # L2 regularization weight
        self.lambda_ = edit_weight[1]  # KL divergence weight

        # === Tokenize input ===
        self.context_tokens = tokenizer(
            context_text, return_tensors='pt', truncation=True, add_special_tokens=False
        ).input_ids[0].to(device)  # [T]
        self.prompt_tokens = tokenizer(
            prompt_text, return_tensors='pt', add_special_tokens=False
        ).input_ids.to(device)
        self.context_length = self.context_tokens.shape[0]

        # === Keyword positions ===
        self.edit_positions = self._find_keyword_positions(self.context_tokens, self.keywords)
        if not self.edit_positions:
            print("No keyword positions found.")
        # assert len(self.edit_positions) > 0, "No keyword positions found."

        # === Original softprompt ===
        with torch.no_grad():
            self.original_softprompt = model(
                self.context_tokens.unsqueeze(0),
                output_softprompt=True
            ).softprompt[0].detach()  # shape: [D]

        # === Perturbations ===
        embed_dim = model.get_input_embeddings().weight.shape[1]
        self.perturb = torch.zeros((len(self.edit_positions), embed_dim), requires_grad=True, device=device)
        self.optimizer = torch.optim.Adam([self.perturb], lr=lr)

    def _find_keyword_positions(self, input_ids: torch.Tensor, keywords: List[str]) -> List[int]:
        id_list = input_ids.tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(id_list)
        edit_positions = []

        for kw in keywords:
            variants = [kw, " " + kw]
            for var in variants:
                kw_ids = self.tokenizer(var, add_special_tokens=False).input_ids
                for i in range(len(id_list) - len(kw_ids) + 1):
                    if id_list[i:i + len(kw_ids)] == kw_ids:
                        found = list(range(i, i + len(kw_ids)))
                        edit_positions.extend(found)
                        # print(f"Found '{kw}' (as '{var}') at positions: {found}")
        return sorted(set(edit_positions))
    
    def calculate_perplexity(self, text, model_name='bert-base-multilingual-cased', invalid_ppl=1000):
        def has_invalid_characters(text, valid_pattern=r'^[\w\s.,!?]+$'):
                """
                Check if the text contains invalid characters.
                Args:
                    text (str): Input text to check.
                    valid_pattern (str): Regex pattern for valid characters (alphanumeric, spaces, common punctuation).
                Returns:
                    bool: True if text contains invalid characters, False otherwise.
                """
                return not bool(re.match(valid_pattern, text))
        # if has_invalid_characters(text):
        #     return invalid_ppl
        # Load pre-trained model and tokenizer
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForCausalLM.from_pretrained(model_name)
        model.eval()
        
        # Encode the input text
        encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        input_ids = encodings['input_ids']
        
        # Calculate loss
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
        
        # Calculate perplexity
        perplexity = math.exp(loss.item())
        return perplexity

    def run(self):
        print("\n=== Starting EditTokenAttacker ===")
        for step in tqdm(range(self.num_steps)):
            self.optimizer.zero_grad()
            
            embed = self.model.get_input_embeddings()(self.context_tokens).detach().clone()  # [T, D]
            for i, pos in enumerate(self.edit_positions):
                embed[pos] += self.perturb[i]  # inject perturbation
            
            softprompt = self.model(
                inputs_embeds=embed.unsqueeze(0),
                segment_lengths=[self.context_length],
                output_softprompt=True
            ).softprompt[0]

            cos_sim = F.cosine_similarity(softprompt.flatten(), self.original_softprompt.flatten(), dim=0)
            l2_loss = torch.norm(self.perturb, p=2)
            orig_soft = F.softmax(self.original_softprompt, dim=-1)
            attk_soft = F.log_softmax(softprompt, dim=-1)
            kl = F.kl_div(attk_soft, orig_soft, reduction="batchmean")

            loss = cos_sim + self.mu * l2_loss + self.lambda_ * kl
            loss.backward()
            self.optimizer.step()

            # if step % 10 == 0 or step == self.num_steps - 1:
            #     print(f"[Step {step}] CosSim: {cos_sim.item():.6f}, L2: {l2_loss.item():.6f}, KL: {kl.item():.6f}, Loss: {loss.item():.6f}")

        return self.finalize()

    def finalize(self):
        print("\n=== Final Generation ===")
        with torch.no_grad():
            embed = self.model.get_input_embeddings()(self.context_tokens).detach().clone()
            for i, pos in enumerate(self.edit_positions):
                embed[pos] += self.perturb[i]

            softprompt = self.model(
                inputs_embeds=embed.unsqueeze(0),
                segment_lengths=[self.context_length],
                output_softprompt=True
            ).softprompt

            output_ids = self.model.generate(
                self.prompt_tokens,
                softprompt=softprompt,
                max_new_tokens=12,
                do_sample=False
            )[0]

            result = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            print(f"→ Generated: {result}")
            print(f"→ Generated Perplexity: {self.calculate_perplexity(result)}")
        # Compute final metrics
        final_cos_sim = F.cosine_similarity(softprompt.flatten(), self.original_softprompt.flatten(), dim=0).item()
        final_l2 = torch.norm(self.perturb, p=2).item()
        orig_soft = F.softmax(self.original_softprompt, dim=-1)
        attk_soft = F.log_softmax(softprompt, dim=-1)
        final_kl = F.kl_div(attk_soft, orig_soft, reduction="batchmean").item()
        print({
            "mu": self.mu,
            "lambda": self.lambda_,
            "cos_sim": final_cos_sim,
            "l2": final_l2,
            "kl": final_kl,
            "generated_text": result
        })
        return result, softprompt

if __name__ == "__main__":

    # context_texts = [
    #     "The quick brown fox jumps over the lazy dog.",
    #     "A journey of a thousand miles begins with a single step."
    # ]
    # prompt_text = "What is the meaning of life?"
    # keywords = ["fox", "journey"]
    context_text = """Joe Biden, born in Scranton, Pennsylvania, on November 20, 1942, had a modest upbringing in a middle-class family. He attended the University of Delaware, where he double-majored in history and political science, graduating in 1965. Afterward, he earned his law degree from Syracuse University College of Law in 1968.\nBiden's early political career began in 1970 when he was elected to the New Castle County Council in Delaware. In 1972, tragedy struck when his wife Neilia and 1-year-old daughter Naomi were killed in a car accident, and his two sons, Beau and Hunter, were injured. Despite this devastating loss, Biden chose to honor his commitment and was sworn in as a senator by his sons' hospital bedsides.\nHe went on to serve as the United States Senator from Delaware for six terms, from 1973 to 2009. During his time in the Senate, Biden was involved in various committees and was particularly known for his expertise in foreign affairs, serving as the chairman of the Senate Foreign Relations Committee on multiple occasions.\nIn 2008, Joe Biden was selected as the running mate for Barack Obama, who went on to win the presidential election. As Vice President, Biden played an integral role in the Obama administration, helping to shape policies and handling issues such as economic recovery, foreign relations, and the implementation of the Affordable Care Act (ACA), commonly known as Obamacare.\nAfter completing two terms as Vice President, Joe Biden decided to run for the presidency in 2020. He secured the Democratic nomination and faced the incumbent President Donald Trump in the general election. Biden campaigned on a platform of unity, promising to heal the divisions in the country and tackle pressing issues, including the COVID-19 pandemic, climate change, racial justice, and economic inequality.\nIn the November 2020 election, Biden emerged victorious, and on January 20, 2021, he was inaugurated as the 46th President of the United States. At the age of 78, Biden became the oldest person to assume the presidency in American history.\nAs President, Joe Biden has worked to implement his agenda, focusing on various initiatives, such as infrastructure investment, climate action, immigration reform, and expanding access to healthcare. He has emphasized the importance of diplomacy in international relations and has sought to rebuild alliances with global partners.\nThroughout his long career in public service, Joe Biden has been recognized for his commitment to bipartisanship, empathy, and his dedication to working-class issues. He continues to navigate the challenges facing the nation, striving to bring the country together and create a positive change for all Americans."""
    prompt_text = 'The first name of the current US president is "'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Load Model and Tokenizer ---
    MODEL_PATH = "path/to/models/AutoCompressor-Llama-2-7b-6k"
    MODEL_NAME = "princeton-nlp/AutoCompressor-Llama-2-7b-6k"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load model
    print("Loading model...")
    model = LlamaAutoCompressorModel.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch_dtype
    ).eval().to(device)

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    print(f"Model loaded to device: {model.device}")

    keywords=["Biden", "president"]

    lambda_sets = [
        (0.0, 0.0),
        (1e-3, 0.1),
        (1e-3, 0.01),
        (1e-5, 0.1),
        (1e-5, 0.001),
        (1e-4, 0.001),
        (1e-4, 0.01),
        (1e-4, 0.05),
    # ]
    # lambda_sets = [
        (0.0, 0.1),
        (0.0, 0.01),
        (0.0, 0.001),
        (0.0, 0.0001),
        (1e-4, 0.0),
        (1e-3, 0.0),
        (1e-2, 0.0),
        (1e-1, 0.0),
    ]
    for lambda_set in lambda_sets:
        attacker = EditTokenAttacker(
            model=model,
            tokenizer=tokenizer,
            context_text=context_text,
            prompt_text=prompt_text,
            keywords=keywords,
            num_steps=100,
            lr=5e-1,
            edit_weight=lambda_set,
            device=device
        )
        
        result, softprompt = attacker.run()
        print(f"Lambda: {lambda_set}, Result: {result}")
