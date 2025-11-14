# CompressionAttack

### Attack Methodologies

1. **HardCom (Hard Compression Attack)**
   - Targets rule-based/algorithmic compression methods (e.g., Selective Context, LLMLingua)
   - Uses token/word/target-level manipulation and adversarial prompt injection

2. **SoftCom (Soft Compression Attack)**  
   - Targets learned compression models (e.g., AutoCompressor, ICAE)
   - Uses gradient-based optimization to craft adversarial compressed representations
   - Manipulates the learned compression latent space

## 🏗️ Project Structure

```
Compressionattack/
│
├── hardcom/                           # HardCom: Attacks on Rule-based Compression
│   ├── src/
│   │   ├── attacks/                  # Attack implementations
│   │   │   ├── attack_llmlingua.py   # Attack against LLMLingua
│   │   │   └── baseline_prompt.py    # Baseline attacks
│   │   │
│   │   ├── defense/                  # Defense mechanisms
│   │   │   ├── ppl_detection.py      # Perplexity-based detection
│   │   │   ├── prevention_detection.py # prevention-based detection
│   │   │   └── LLM_detection.py      # LLM-based detection
│   │   │
│   │   ├── evaluators/               # Evaluation tools
│   │   │   ├── tool_selection_test.py
│   │   │   ├── QA_test.py
│   │   │   ├── stealthy_*.py         # Stealthiness metrics
│   │   │   └── F1_score.py
│   │   │
│   │   ├── utils/                    # Utility functions
│   │   │   ├── get_ppl.py            # PPL calculation
│   │   │   ├── get_edit_token.py     # Token manipulation
│   │   │   └── inference.py          # Model inference
│   │   │
│   │   └── data/                     # Datasets and prompts
│   │       ├── data.json
│   │       ├── squad_QA_dataset.json
│   │       └── *.txt                 # Prompt templates
│   │
│   ├── examples/                     # Demo applications
│   │   ├── agent/                    # Agent-based examples
│   │   └── Tool_selection/           # Tool selection attacks
│   │
│   ├── config.py                     # Configuration
│   └── README.md
│
├── softcom/                           # SoftCom: Attacks on Learned Compression
│   │
│   ├── AutoCompressors/              # Target: AutoCompressor (compression model)
│   │   ├── auto_compressor.py        # AutoCompressor implementation
│   │   ├── train.py                  # Model training
│   │   ├── evaluate_*.py             # Evaluation on various tasks
│   │   ├── modeling_*.py             # Model architectures
│   │   └── run/                      # Training/eval scripts
│   │
│   ├── ComAttack/                    # Attack implementations against soft compression
│   │   ├── icae_attack*.py           # Attacks using ICAE
│   │   ├── modeling_icae*.py         # ICAE model for attacks
│   │   ├── prompt_benchmark.py       # Benchmarking tools
│   │   │
│   │   └── open_source_code/         # Core attack code
│   │       ├── icae_attack*.py       # Various attack scenarios
│   │       ├── icae_QA*.py           # QA-specific attacks
│   │       ├── icae_recommend*.py    # Recommendation attacks
│   │       ├── evaluate_*.py         # Attack evaluation
│   │       ├── modeling_icae*.py     # ICAE architectures
│   │       ├── dataloader/           # Data loading utilities
│   │       └── *.sh                  # Execution scripts
│   │
│   └── Comattack_dataset/            # Datasets for soft attacks
│       ├── squad/                    # SQuAD QA dataset
│       ├── recommend/                # Product recommendation data
│       └── *.json                    # Attack-specific datasets
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- PyTorch 2.0+
- Transformers 4.30+

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Compressionattack
```

2. Install dependencies:
```bash
# Core dependencies
pip install torch transformers datasets accelerate

# For hard compression attacks
pip install llmlingua sentence-transformers bert-score python-Levenshtein flask

# For soft compression attacks
pip install flash-attn sentencepiece packaging wandb

# Optional: For defense mechanisms
pip install language-tool-python nltk
```

3. Configure model paths:
```bash
cd hardcom
cp config.py.example config.py  # If available
# Edit config.py with your model paths
```

## 📚 Usage

### HardCom: Attacking Rule-based Compression

HardCom attacks exploit algorithmic compression methods that use perplexity or other heuristics to select tokens for removal.

#### Running HardCom Attacks

```bash
cd hardcom

# Configure your model paths
vim config.py  # Set paths for compression and target models

# Attack LLMLingua compression
python src/attacks/attack_llmlingua.py

# Run evaluation on different tasks
python src/evaluators/QA_test.py              # Question Answering
python src/evaluators/tool_selection_test.py  # Tool Selection
python src/evaluators/stealthy_character.py   # Stealthiness metrics
```

### SoftCom: Attacking Learned Compression

SoftCom attacks target neural compression models that learn to compress prompts into continuous representations.

#### Target Compression Models
- **AutoCompressor**: Learns summary vectors for context compression
- **ICAE (In-Context Auto-Encoder)**: End-to-end differentiable compression

#### Running SoftCom Attacks

```bash
cd softcom/ComAttack

# Attack with ICAE on different tasks
bash icae_attack_soft.sh                      # Basic ICAE attack

# QA-specific attacks
bash open_source_code/icae_QA.sh              # Question Answering attack
python open_source_code/evaluate_QA.py        # Evaluate QA attack

# Recommendation attacks
bash open_source_code/icae_recommend_keyword_dataset_improve.sh   # Improve target
bash open_source_code/icae_recommend_keyword_dataset_degrade.sh   # Degrade best
python open_source_code/evaluate_rec_improve.py                   # Evaluate
```

#### Training Target Compression Models (Optional)

If you want to train your own compression models to attack:

```bash
cd softcom/AutoCompressors

bash run/train_llama.sh

# Evaluate compression performance
bash run/eval_llama.sh
python evaluate_qa.py
python evaluate_recommend.py
```
### Defense Mechanisms

The project includes multiple defense methods:

```bash
cd hardcom/src/defense

# Perplexity-based detection
python ppl_detection.py

#Prevention-based detection
python prevention_detection.py

# LLM-based detection
python LLM_detection.py
```

## 🎯 Attack Scenarios

Both HardCom and SoftCom can be applied to various downstream tasks:

### 1. Question Answering (QA) Attack

**HardCom Approach**:
```bash
cd hardcom
python src/attacks/attack_llmlingua.py --task qa --dataset squad
python src/evaluators/QA_test.py
```


**SoftCom Approach**:
```bash
cd softcom/ComAttack
bash open_source_code/icae_QA.sh
python open_source_code/evaluate_QA.py
```

### 2. Product Recommendation Attack

**HardCom Approach**:
```bash
cd hardcom
python src/evaluators/product_recommendation_test.py
```

**SoftCom Approach**:
```bash
cd softcom/ComAttack/open_source_code

# Improve target product ranking
bash icae_recommend_keyword_dataset_improve.sh
python evaluate_rec_improve.py

# Degrade best product ranking  
bash icae_recommend_keyword_dataset_degrade.sh
python evaluate_rec_degrade.py
```

## 📊 Evaluation Metrics

The framework provides comprehensive evaluation metrics:

### Stealthiness Metrics
- **Composite Stealth Score**: `λ · cosine_sim(C, C̃) + (1-λ) · BERTScore(C, C̃)`
- **Semantic Similarity**: Cosine similarity using SentenceTransformers
- **BERT-Score**: F1 score with baseline rescaling
- **Character-level Similarity**: Normalized edit distance

### Attack Success Metrics
- **Attack Success Rate (ASR)**: Percentage of successful adversarial manipulations
- **Task Performance Degradation**: Drop in target task accuracy
- **F1 Score**: Question answering accuracy
- **Recommendation Rank Change**: Position change of target/best items

## 🛡️ Defense Evaluation

Test defense mechanisms against attacks:

```bash
cd hardcom/src/defense

# Calculate clean PPL threshold
python calculate_clean_ppl_threshold.py

# Test PPL-based detection
python ppl_detection.py --threshold 80.0

# Test LLM detection
python LLM_detection.py
```

## 📁 Datasets

The project includes several datasets:
- **SQuAD**: Question answering dataset
- **Product Recommendation**: E-commerce product datasets
- **Tool Selection**: Agent tool selection scenarios
- **Custom Datasets**: Keyword-injected and adversarial datasets

Datasets are located in:
- `hardcom/src/data/`
- `softcom/Comattack_dataset/`

## 🔧 Configuration

Edit `hardcom/config.py` to set up model paths:

```python
# Model paths
COMPRESSION_MODEL_PATH = "path/to/compression-model"
LARGE_MODEL_PATH = "path/to/Qwen3-32B"
LLAMA_PATH = "path/to/model"
MISTRAL_PATH = "path/to/Mistral-7B-Instruct-v0.2"

# CUDA settings
DEFAULT_CUDA_DEVICE = "cuda:0"
```