# R-Zero: Revolutionizing Reasoning in LLMs with Self-Evolution

**R-Zero is a groundbreaking framework that empowers Large Language Models to learn and improve their reasoning skills autonomously, without relying on any pre-existing data.** ([Original Repo](https://github.com/Chengsong-Huang/R-Zero))

R-Zero offers a novel approach to LLM training, enabling self-improvement through a co-evolutionary Challenger-Solver loop.

## Key Features

*   **Autonomous Learning:** Trains LLMs from scratch, eliminating the need for labeled datasets or pre-existing knowledge.
*   **Co-Evolutionary Architecture:**  A Challenger-Solver dynamic creates an adaptive curriculum, pushing the model to continuously enhance its reasoning abilities.
*   **Proven Performance Boosts:** Achieves significant performance gains on various reasoning benchmarks.
*   **Generalization Capabilities:**  Reasoning skills learned on specific domains (like math) translate to broader reasoning tasks.
*   **Model Agnostic:**  Enhances the performance of diverse backbone LLMs, demonstrating versatility.

## Updates
*   **2025-8-27:** Analysis added on iteration scaling and a model taking on two roles.
*   **2025-8-25:** Code updates for smoother training (implemented by stopit).
*   **2025-8-8:** Featured as `#2 Paper of the day` on [huggingface daily paper](https://huggingface.co/papers/2508.05004).
*   **2025-8-7:** Released our [paper](https://arxiv.org/abs/2508.05004) and code.

## Quickstart Guide

Get started with R-Zero by following these simple steps:

### 1.  Environment Setup & Directory Configuration

```bash
git clone https://github.com/Chengsong-Huang/R-Zero.git
cd R-Zero
pip install -r requirements.txt
export STORAGE_PATH="/path/to/your/storage"
export HUGGINGFACENAME="yourhuggingfacename"

mkdir -p "$STORAGE_PATH/evaluation" "$STORAGE_PATH/models" "$STORAGE_PATH/generated_question" "$STORAGE_PATH/temp_results"
```

### 2.  API Key Configuration

*   Populate your API keys for Hugging Face and WandB (for logging) in `tokens.json`.
*   Add your OpenAI GPT API key in `evaluation/results_recheck.py`.

### 3. Run the Experiments

Replicate the experimental results using this script:

```bash
# Format: bash scripts/main.sh [Base_Model_Name] [Abbreviation]
# Example:
bash scripts/main.sh Qwen/Qwen3-4B-Base qwen3-4b
```

## Impressive Results

R-Zero significantly outperforms base models.  The table below highlights the performance improvements across various benchmarks.  Peak performance for each model is highlighted in **bold**.

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

## FAQ for Developers

### Q: Hardware Setup for Experiments?

**A:** Experiments were conducted on an 8-GPU server.  Adapt the code if using larger models or different hardware.

### Q: Environment Configuration Issues?

**A:** Refer to [EasyR1](https://github.com/hiyouga/EasyR1/tree/main) for setup guidance or their Docker environment for reference.

### Q: Training Logs and Checkpoints?

**A:** All generated data is saved in the directory specified by the `STORAGE_PATH` environment variable, and datasets are sent to Hugging Face via `HUGGINGFACENAME`.

### Q: Code Stuck During Questioner Training?

**A:** This might be due to an infinite loop caused by a bug in the `math_verify` lib.  Restart training from the last checkpoint.

## Acknowledgements

R-Zero is built upon the foundations of [**EasyR1**](https://github.com/hiyouga/EasyR1/tree/main) and references the evaluation work from [**General-Reasoner**](https://github.com/TIGER-AI-Lab/General-Reasoner).

## Citation

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