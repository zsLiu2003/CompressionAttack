import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import json
from utils import generate_output_from_decoder_memory
from difflib import SequenceMatcher
from transformers import HfArgumentParser
from peft import LoraConfig
from safetensors.torch import load_file
from modeling_icae_multi_span import ICAE, ModelArguments, DataArguments, TrainingArguments

class CounterfactualTester:
    def __init__(self, tokenizer, model, device=None):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def get_memory(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].to(self.device)
        input_embeddings = self.model.tokens_to_embeddings(input_ids.unsqueeze(0)).squeeze(0)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            memory = self.model._compress(inputs_embeds=input_embeddings)
        return memory.squeeze(0)  # shape: [num_labels]
    
    def try_multiple_insertions(base_text, insert_token="Well,"):
        positions = [0, 20, 50, 100]
        variants = []
        for pos in positions:
            tokens = base_text.split()
            if pos < len(tokens):
                tokens.insert(pos, insert_token)
                variants.append(" ".join(tokens))
        return variants

if __name__ == "__main__":
    
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Load ICAE model
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
    model = model.to("cuda").eval()
    tokenizer = model.tokenizer
    file_path = 'path/to/Comattack_dataset/QA_keywords_edit.json'

    tester = CounterfactualTester(tokenizer, model)

    input_text = "CBS broadcast Super Bowl 50 in the U.S., and charged an average of $5 million for a 30-second commercial during the game. The Super Bowl 50 halftime show was headlined by the British rock group Coldplay with special guest performers Beyonc\u00e9 and Bruno Mars, who headlined the Super Bowl XLVII and Super Bowl XLVIII halftime shows, respectively. It was the third-most watched U.S. broadcast ever."
    input_text_attacked = "CBS broadcast Super Bowl 50 in the U.S., and charged an average of $5 million for a 30-second commercial during the game. The Super Bowl 50 halftime show was headlined by the British rock groupTo⁣ with special guest performers Beyoncé and Errorf Mars, who headlined the Super Bowl XLVII and Super Bowl XLVIII halftime shows, respectively. It was the third-most watched U.S. broadcast ever." 
    
    # read from json and extract by key
    # with open(file_path, 'r', encoding='utf-8') as f:
    #     data = f.read()
    #     json_data_list = json.loads(data) # Load the entire list of JSON objects

    # for i, item in enumerate(json_data_list):
    #     input_text = item.get("original", "")
    #     input_text_attacked = item.get("optimized", "")

    # Strategy 1
    # modified_input_text = "Well," + input_text
    # modified_input_text_attacked = "Well," + input_text_attacked
    
    # Strategy 2
    # modified_input_text = input_text.replace("Bruno Mars", "however, Bruno Mars")
    # modified_input_text_attacked = input_text_attacked.replace("Bruno Mars", "however, Bruno Mars")

    # Strategy 3
    # modified_input_text = input_text.replace('Beyoncé','however, Beyoncé')
    # modified_input_text_attacked = input_text_attacked.replace('Beyoncé','however, Beyoncé')
    
    # Strategy 4
    # modified_input_text = input_text.replace('Beyoncé','')
    # modified_input_text_attacked = input_text_attacked.replace('Beyoncé','')
    
    # Strategy 5
    modified_input_text = input_text.replace('rock group','')
    modified_input_text_attacked = input_text_attacked.replace('rock group','')

    input_memory = tester.get_memory(input_text)
    modified_input_memory = tester.get_memory(modified_input_text)
    input_memory_attacked = tester.get_memory(input_text_attacked)
    modified_input_memory_attacked = tester.get_memory(modified_input_text_attacked)
    
    generated_answer = generate_output_from_decoder_memory(
        model=model,
        decoder_memory_embeddings=input_memory.unsqueeze(0),
        prompt_text='PLEASE anser the question ONLY based on the context I provide: What performer lead the Super Bowl XLVIII halftime show?',
        max_out_length=512,
        device="cuda"
    )
    modified_generated_answer = generate_output_from_decoder_memory(
        model=model,
        decoder_memory_embeddings=modified_input_memory.unsqueeze(0),
        prompt_text='PLEASE anser the question ONLY based on the context I provide: What performer lead the Super Bowl XLVIII halftime show?',
        max_out_length=512,
        device="cuda"
    )
    generated_answer_attacked = generate_output_from_decoder_memory(
        model=model,
        decoder_memory_embeddings=input_memory_attacked.unsqueeze(0),
        prompt_text='PLEASE anser the question ONLY based on the context I provide: What performer lead the Super Bowl XLVIII halftime show?',
        max_out_length=512,
        device="cuda"
    )
    modified_generated_answer_attacked = generate_output_from_decoder_memory(
        model=model,
        decoder_memory_embeddings=modified_input_memory_attacked.unsqueeze(0),
        prompt_text='PLEASE anser the question ONLY based on the context I provide: What performer lead the Super Bowl XLVIII halftime show?',
        max_out_length=512,
        device="cuda"
    )
    
    print("answer:", generated_answer)
    print("modified answer:", modified_generated_answer)

    print("attacked answer:", generated_answer_attacked)
    print("modified attacked answer:", modified_generated_answer_attacked)

        # if i==2: break