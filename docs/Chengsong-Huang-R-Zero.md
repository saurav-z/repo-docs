# R-Zero: Revolutionizing LLM Reasoning with Self-Evolution (Learn More at [Original Repo](https://github.com/Chengsong-Huang/R-Zero))

R-Zero is a groundbreaking framework that empowers Large Language Models to autonomously improve their reasoning abilities, starting from scratch without any pre-existing datasets.

**Key Features:**

*   **Fully Autonomous Learning:** R-Zero eliminates the need for labeled data, enabling LLMs to evolve from a base model with no external information.
*   **Co-Evolutionary Loop:** A Challenger-Solver dynamic creates a targeted, adaptive curriculum for continuous improvement.
*   **Demonstrated Performance:** Achieve significant performance boosts on diverse reasoning benchmarks, consistently outperforming baseline models.
*   **Generalization Capabilities:** Skills learned in specific domains (e.g., math) transfer effectively to broader reasoning tasks.
*   **Model Agnostic:** Enhances the reasoning capabilities of different LLM architectures.

## Quickstart Guide

Follow these steps to start using R-Zero:

### 1. Environment Setup and Directory Preparation

```bash
git clone https://github.com/Chengsong-Huang/R-Zero.git
cd R-Zero
pip install -r requirements.txt
export STORAGE_PATH="/path/to/your/storage"  # Set your storage path
export HUGGINGFACENAME="yourhuggingfacename"
mkdir -p "$STORAGE_PATH/evaluation" "$STORAGE_PATH/models" "$STORAGE_PATH/generated_question" "$STORAGE_PATH/temp_results"
```

### 2. API Key Configuration

Add your API keys to `tokens.json` (Hugging Face, WandB) and your OpenAI GPT API key in `evaluation/results_recheck.py` for evaluation.

### 3. Run Experiments

Replicate the experiments using the provided script:

```bash
# Format: bash scripts/main.sh [Base_Model_Name] [Abbreviation]
# Example:
bash scripts/main.sh Qwen/Qwen3-4B-Base qwen3-4b
```

## Impressive Results

(See the original README for the results table)

## Frequently Asked Questions (FAQ)

**Q: What is the hardware setup for the experiments?**

**A:** The experiments were conducted on an 8-GPU server. Modify the code for different hardware.

**Q: How to resolve environment configuration issues?**

**A:** Refer to [EasyR1](https://github.com/hiyouga/EasyR1/tree/main) or use their Docker environment for setup.

**Q: Where are the training logs and model checkpoints saved?**

**A:** Data is saved in the directory specified by the `STORAGE_PATH` environment variable. Datasets are also uploaded to Hugging Face using `HUGGINGFACENAME`.

**Q: What if the code gets stuck during the questioner training process?**

**A:** Restart training from the last saved checkpoint due to potential issues in the `math_verify` lib.

## Acknowledgements

This project is built upon the foundational work of [**EasyR1**](https://github.com/hiyouga/EasyR1/tree/main) and references the evaluation from [**General-Reasoner**](https://github.com/TIGER-AI-Lab/General-Reasoner).

## Citation
(See the original README for the citation)

## Star History
(See the original README for the Star History graph)