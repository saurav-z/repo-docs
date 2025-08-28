# R-Zero: Self-Evolving Reasoning for LLMs - Train from Zero Data

**R-Zero empowers Large Language Models to autonomously enhance their reasoning abilities without any pre-existing datasets.** ([Original Repo](https://github.com/Chengsong-Huang/R-Zero))

R-Zero introduces a novel framework that enables LLMs to learn and evolve their reasoning skills from scratch, eliminating the need for expensive, human-curated data. Through a self-contained co-evolutionary loop, R-Zero allows LLMs to continuously improve, demonstrating significant performance gains across various benchmarks.

## Key Features

*   🚀 **Fully Autonomous:** Operates without any initial training data or labeled examples.
*   🔄 **Co-Evolutionary Loop:** A dynamic Challenger-Solver system creates a tailored curriculum for continuous improvement.
*   📈 **Proven Performance:** Achieves substantial performance boosts on various reasoning benchmarks.
*   🧠 **Strong Generalization:** Reasoning skills learned in one domain effectively transfer to general reasoning tasks.
*   ⚙️ **Model-Agnostic:** Consistently enhances the performance of various base LLMs.

## 🔥 Updates

*   **2025-8-27:** Analysis on iteration scaling and one model taking on two roles.
*   **2025-8-25:** Code updates for smoother training (using stopit).
*   **2025-8-8:** R-Zero earned `#2 Paper of the day` on [Hugging Face Daily Papers](https://huggingface.co/papers/2508.05004).
*   **2025-8-7:** Paper and code release.

## ⚡️ Quickstart

Get up and running with R-Zero in a few simple steps:

### 1.  Environment Setup

```bash
git clone https://github.com/Chengsong-Huang/R-Zero.git
cd R-Zero
pip install -r requirements.txt

# Set up environment variables
export STORAGE_PATH="/path/to/your/storage"
export HUGGINGFACENAME="yourhuggingfacename"

mkdir -p "$STORAGE_PATH/evaluation" "$STORAGE_PATH/models" "$STORAGE_PATH/generated_question" "$STORAGE_PATH/temp_results"
```

### 2.  API Key Configuration

*   Add your **Hugging Face** and **WandB** API keys to `tokens.json`.
*   Add your **OpenAI GPT** API key to `evaluation/results_recheck.py`.

### 3.  Run the Experiments

Replicate the reported results with a single command:

```bash
# Format: bash scripts/main.sh [Base_Model_Name] [Abbreviation]
# Example:
bash scripts/main.sh Qwen/Qwen3-4B-Base qwen3-4b
```

## 📊 Impressive Results

R-Zero demonstrates significant performance improvements across various reasoning benchmarks. The table below compares the performance of base models, the Zero-Shot Challenger, and R-Zero over multiple iterations.  Peak performance for each model is **bolded**.

| Model Name | Overall AVG | MATH AVG | SuperGPQA | MMLU-Pro | BBEH |
|:---|:---:|:---:|:---:|:---:|:---:|
| ***Qwen3-4B-Base*** | | | | | |
| &emsp;Base Model | 27.10 | 42.58 | 20.88 | 37.38 | 7.57 |
| &emsp;Base Challenger | 30.83 | 44.36 | 24.77 | 47.59 | 6.59 |
| &emsp;R-Zero (Iter 1) | 34.27 | 48.06 | **27.92** | 51.69 | 9.42 |
| &emsp;R-Zero (Iter 2) | **34.92** | 48.44 | 27.72 | **53.75** | 9.76 |
| &emsp;R-Zero (Iter 3) | 34.64 | **49.07** | 27.55 | 51.53 | **10.42** |
| ***Qwen3-8B-Base*** | | | | | |
| &emsp;Base Model | 34.49 | 49.18 | 28.33 | 51.80 | 8.63 |
| &emsp;Base Challenger | 36.43 | 51.87 | 30.12 | 54.14 | 9.60 |
| &emsp;R-Zero (Iter 1) | 37.93 | 53.39 | 31.26 | 57.17 | 9.91 |
| &emsp;R-Zero (Iter 2) | 38.45 | 53.84 | **31.58** | 58.20 | 10.20 |
| &emsp;R-Zero (Iter 3) | **38.73** | **54.69** | 31.38 | **58.23** | **10.60** |
| ***OctoThinker-3B*** | | | | | |
| &emsp;Base Model | 12.27 | 26.64 | 10.09 | 10.87 | 1.46 |
| &emsp;Base Challenger | 14.41 | 27.51 | 11.19 | 14.53 | **4.40** |
| &emsp;R-Zero (Iter 1) | 14.93 | 27.76 | 12.21 | 15.72 | 4.05 |
| &emsp;R-Zero (Iter 2) | 15.11 | 28.20 | 12.43 | 16.08 | 3.74 |
| &emsp;R-Zero (Iter 3) | **15.67** | **29.32** | **12.44** | **16.71** | 4.20 |
| ***OctoThinker-8B*** | | | | | |
| &emsp;Base Model | 16.81 | 32.11 | 13.26 | 20.21 | 1.64 |
| &emsp;Base Challenger | 25.08 | 36.41 | 16.99 | 41.46 | 5.46 |
| &emsp;R-Zero (Iter 1) | 26.44 | 37.80 | 19.15 | **42.05** | 6.77 |
| &emsp;R-Zero (Iter 2) | 26.77 | 38.23 | 19.27 | 41.34 | **8.25** |
| &emsp;R-Zero (Iter 3) | **26.88** | **38.52** | **19.82** | 40.92 | **8.25** |

## ❓ FAQ

**Q: What are the hardware requirements for running experiments?**

**A:** Experiments were conducted on an 8-GPU server. Models should fit on a single GPU (e.g., 4B or 8B models). For larger models or different hardware setups, code modifications may be necessary.

**Q:  What if I run into environment configuration problems during installation?**

**A:**  This project is inspired by [EasyR1](https://github.com/hiyouga/EasyR1/tree/main).  Refer to their setup instructions or consider using their Docker environment for assistance.

**Q: Where are the training logs and checkpoints stored?**

**A:** All generated data, including logs, datasets, and model checkpoints, are saved in the directory specified by the `STORAGE_PATH` environment variable.  Datasets are also sent to Hugging Face via `HUGGINGFACENAME`.

**Q: What if the code gets stuck during the questioner training process?**

**A:** This may be due to a bug in the `math_verify` library. Implement a timeout control to mitigate this, but it may not catch all cases. If this issue arises, restart training from the last saved checkpoint.

## 🙏 Acknowledgements

This framework is built upon the work of [**EasyR1**](https://github.com/hiyouga/EasyR1/tree/main) and incorporates its core functionalities. The evaluation process referenced the work from [**General-Reasoner**](https://github.com/TIGER-AI-Lab/General-Reasoner).

## 💬 Citation

If R-Zero is useful in your work, please cite the paper:

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