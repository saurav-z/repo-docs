# R-Zero: Self-Evolving Reasoning LLM from Zero Data

**R-Zero is a revolutionary framework that empowers Large Language Models to learn reasoning skills autonomously, starting with no pre-existing data.** ([Original Repo](https://github.com/Chengsong-Huang/R-Zero))

## Key Features

*   **Fully Autonomous Learning:** Trains LLMs from scratch, eliminating the need for labeled datasets or pre-existing problem sets.
*   **Co-Evolutionary Dynamics:** Employs a Challenger-Solver loop, creating a dynamic and adaptive curriculum for continuous reasoning improvement.
*   **Proven Performance Gains:** Demonstrates significant performance boosts across various reasoning benchmarks.
*   **Strong Generalization:** Reasoning skills learned are transferable to general reasoning tasks.
*   **Model Agnostic:** Enhances the performance of a variety of base LLMs.

## 🚀 Overview

R-Zero overcomes the limitations of traditional reasoning model training by enabling LLMs to self-improve without any external data. This innovative framework utilizes a co-evolutionary approach:

*   **Challenger:** This component generates challenging problems designed to identify and expose weaknesses in the Solver.
*   **Solver:** This component focuses on solving the increasingly difficult tasks posed by the Challenger, thus continuously improving its reasoning abilities.

This self-contained cycle utilizes techniques such as majority voting for pseudo-labeling and relative policy optimization, leading to a perfectly tailored and adaptive learning process.

## 📊 Impressive Results

R-Zero shows remarkable improvements across several reasoning benchmarks. The table below compares the performance of Base Models, Zero-Shot Challenger baselines, and the iterative R-Zero framework. Peak performance for each model is highlighted in **bold**.

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

## 🛠️ Quickstart Guide

### 1.  Set Up Your Environment and Prepare Directories
```bash
git clone https://github.com/Chengsong-Huang/R-Zero.git
cd R-Zero
pip install -r requirements.txt
export STORAGE_PATH="/path/to/your/storage"
export HUGGINGFACENAME="yourhuggingfacename"
mkdir -p "$STORAGE_PATH/evaluation" "$STORAGE_PATH/models" "$STORAGE_PATH/generated_question" "$STORAGE_PATH/temp_results"
```
### 2.  Add API Keys
*   Insert your Hugging Face and WandB API keys into `tokens.json`.
*   Add your OpenAI GPT API key in `evaluation/results_recheck.py`.

### 3. Run Experiments
```bash
# Format: bash scripts/main.sh [Base_Model_Name] [Abbreviation]
# Example: bash scripts/main.sh Qwen/Qwen3-4B-Base qwen3-4b
```

## ❓ FAQ for Developers

**Q: What hardware setup is recommended for running experiments?**

**A:** The experiments were conducted on an 8-GPU server. Models that can run on a single GPU (e.g., 4B or 8B) were used.

**Q: What to do if environment configuration issues arise during installation?**

**A:** Consider consulting the setup instructions or Docker environment from [EasyR1](https://github.com/hiyouga/EasyR1/tree/main), as R-Zero is inspired by their work.

**Q: Where can I find training logs and model checkpoints?**

**A:** Logs, datasets, and model checkpoints are saved in the directory specified by the `STORAGE_PATH` environment variable. Datasets are also sent to Hugging Face via `HUGGINGFACENAME`.

**Q: What should I do if the code gets stuck during the questioner training process?**

**A:** This may be due to a bug in the `math_verify` library.  Restart training from the last saved checkpoint.

## 🙏 Acknowledgements

We are indebted to the creators of [**EasyR1**](https://github.com/hiyouga/EasyR1/tree/main) for their excellent work, which forms the foundation of this framework. Additionally, we referenced the work from [**General-Reasoner**](https://github.com/TIGER-AI-Lab/General-Reasoner) for our evaluation process.

## 💬 Citation
If you use this work, please cite our paper:
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