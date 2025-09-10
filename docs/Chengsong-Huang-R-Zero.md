# R-Zero: Self-Evolving Reasoning LLM from Zero Data

**Unlock the power of autonomous reasoning with R-Zero, a groundbreaking framework that empowers Large Language Models to learn and improve their reasoning abilities without any initial data.** ([Original Repository](https://github.com/Chengsong-Huang/R-Zero))

## Key Features

*   **Autonomous Learning:** R-Zero starts with a base LLM and evolves its reasoning skills from scratch, requiring no pre-existing datasets or human-annotated solutions.
*   **Co-evolutionary Architecture:** A Challenger-Solver dynamic creates a self-improving loop that generates a tailored curriculum for continuous learning.
*   **Enhanced Performance:** Achieves significant performance gains on various reasoning benchmarks, improving the capabilities of various LLMs.
*   **Strong Generalization:** Reasoning skills learned in specific domains effectively transfer to broader reasoning tasks.
*   **Model Agnostic:** R-Zero can be applied to various LLM backbones, enhancing their reasoning capabilities consistently.

## 🚀 Quickstart

### 1. Set up your environment:

```bash
git clone https://github.com/Chengsong-Huang/R-Zero.git
cd R-Zero
pip install -r requirements.txt

export STORAGE_PATH="/path/to/your/storage" # Set your storage path
export HUGGINGFACENAME="yourhuggingfacename" # Your Hugging Face username
```

Create the necessary directories:

```bash
mkdir -p "$STORAGE_PATH/evaluation" "$STORAGE_PATH/models" "$STORAGE_PATH/generated_question" "$STORAGE_PATH/temp_results"
```

### 2. Add API Keys

*   Populate `tokens.json` with your Hugging Face and WandB API keys.
*   Add your OpenAI GPT API key in `evaluation/results_recheck.py` for evaluation.

### 3. Run Experiments

Replicate the results with a single command:

```bash
# The script takes the base model name and an abbreviation as arguments
# The abbreviation is used for creating a directory to save the model.
# Format: bash scripts/main.sh [Base_Model_Name] [Abbreviation]

# Example using Qwen/Qwen3-4B-Base:
bash scripts/main.sh Qwen/Qwen3-4B-Base qwen3-4b
```

## 📊 Impressive Results

[Include the performance table from the original README here, focusing on the key performance metrics and highlighting the improvements achieved by R-Zero.]

## ❓ FAQ for Developers

*   **Hardware:** Experiments were conducted on an 8-GPU server. Adapt code for larger models or different hardware.
*   **Environment Issues:** Refer to [EasyR1](https://github.com/hiyouga/EasyR1/tree/main) for setup guidance or consider their Docker environment.
*   **Storage:** Training logs, datasets, and checkpoints are saved in the directory specified by `STORAGE_PATH` and datasets are sent to huggingface via `HUGGINGFACENAME`.
*   **Questioner Issues:** If the code gets stuck during questioner training, restart from the last saved checkpoint.

## 🙏 Acknowledgements

This project is built upon the foundational work of [EasyR1](https://github.com/hiyouga/EasyR1/tree/main) and references the evaluation methodology of [General-Reasoner](https://github.com/TIGER-AI-Lab/General-Reasoner).

## 💬 Citation

```
@article{huang2025rzeroselfevolvingreasoningllm,
      title={R-Zero: Self-Evolving Reasoning LLM from Zero Data},
      author={Chengsong Huang and Wenhao Yu and Xiaoyang Wang and Hongming Zhang and Zongxia Li and Ruosen Li and Jiaxin Huang and Haitao Mi and Dong Yu},
      year={2025},
      eprint={2508.05004},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.05004},
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Chengsong-Huang/R-Zero&type=Date)](https://star-history.com/#Chengsong-Huang/R-Zero&Date)