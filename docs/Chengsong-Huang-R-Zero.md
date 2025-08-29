# R-Zero: Self-Evolving Reasoning LLM from Zero Data

**R-Zero empowers Large Language Models to autonomously learn and evolve their reasoning skills, eliminating the need for any pre-existing data.** Check out the original repository [here](https://github.com/Chengsong-Huang/R-Zero).

[Paper](https://arxiv.org/abs/2508.05004) | [Webpage](https://chengsong-huang.github.io/R-Zero.github.io/)

## 🔥 Key Updates

*   **2025-8-27:** Analysis on iteration scaling and a single model taking on two roles added.
*   **2025-8-25:** Code updates for smoother training (using `stopit`).
*   **2025-8-8:** Featured as `#2 Paper of the day` on [Hugging Face Papers](https://huggingface.co/papers/2508.05004).
*   **2025-8-7:** Paper and code release.

## 🚀 Core Concepts

R-Zero introduces a novel, self-evolving framework enabling LLMs to improve their reasoning capabilities without relying on any pre-existing datasets or labels.  It leverages a co-evolutionary loop, creating a truly self-contained learning system.

### How it Works

The core of R-Zero is a dynamic collaboration between two instances of the same base model:

*   **Challenger 🎯:** Probes the Solver for weaknesses, generating challenging problems tailored to its current abilities.
*   **Solver 🧠:**  Continuously improves by solving the increasingly difficult tasks posed by the Challenger.

This process generates an adaptive curriculum, improving both the Challenger's ability to generate difficult questions and the Solver's skill in providing accurate answers.  The system employs techniques like majority voting for pseudo-labels and relative policy optimization for training.

### Key Features

*   ✅ **Fully Autonomous:** Operates without external data, pre-existing problems, or human-annotated solutions.
*   🔄 **Co-Evolutionary Loop:**  A Challenger-Solver dynamic creates a targeted, adaptive curriculum for continuous improvement.
*   📈 **Proven Performance:** Delivers significant performance gains on reasoning benchmarks.
*   🌍 **Strong Generalization:**  Reasoning skills learned on specific domains (like math) transfer effectively to general reasoning tasks.
*   🛠️ **Model-Agnostic:** Consistently improves the performance of various LLMs.

## ⚡️ Quickstart Guide

Get started with R-Zero in a few simple steps:

### 1. Configure Environment & Prepare Directories

```bash
git clone https://github.com/Chengsong-Huang/R-Zero.git
cd R-Zero
pip install -r requirements.txt
export STORAGE_PATH="/path/to/your/storage" # Set your storage path
export HUGGINGFACENAME="yourhuggingfacename"

mkdir -p "$STORAGE_PATH/evaluation" "$STORAGE_PATH/models" "$STORAGE_PATH/generated_question" "$STORAGE_PATH/temp_results"
```

### 2.  Set API Keys

*   Add your Hugging Face and WandB API keys in `tokens.json`.
*   Add your OpenAI GPT API key in `evaluation/results_recheck.py`.

### 3. Run Experiments

Replicate the experimental results with a single script:

```bash
# The script takes the base model name and an abbreviation as arguments
# The abbreviation is used for creating a directory to save the model.
# Format: bash scripts/main.sh [Base_Model_Name] [Abbreviation]

# Example using Qwen/Qwen3-4B-Base:
bash scripts/main.sh Qwen/Qwen3-4B-Base qwen3-4b
```

## 📊 Performance Results

R-Zero demonstrates significant performance improvements across various models.  The table below highlights the performance gains achieved with R-Zero compared to baseline models. **Bold** indicates peak performance.

| Model Name         | Overall AVG | MATH AVG | SuperGPQA | MMLU-Pro | BBEH |
| :----------------- | :---------: | :------: | :-------: | :------: | :--: |
| ***Qwen3-4B-Base*** |             |          |           |          |      |
| &emsp;Base Model   |    27.10    |   42.58  |    20.88  |   37.38  | 7.57 |
| &emsp;Base Challenger |   30.83    |   44.36  |    24.77  |   47.59  | 6.59 |
| &emsp;R-Zero (Iter 1) |   34.27    |   48.06  |  **27.92**  |   51.69  | 9.42 |
| &emsp;R-Zero (Iter 2) | **34.92**  |   48.44  |    27.72  | **53.75**  | 9.76 |
| &emsp;R-Zero (Iter 3) |   34.64    |  **49.07**  |    27.55  |   51.53  | **10.42**|
| ***Qwen3-8B-Base*** |             |          |           |          |      |
| &emsp;Base Model   |    34.49    |   49.18  |    28.33  |   51.80  | 8.63 |
| &emsp;Base Challenger |   36.43    |   51.87  |    30.12  |   54.14  | 9.60 |
| &emsp;R-Zero (Iter 1) |   37.93    |   53.39  |    31.26  |   57.17  | 9.91 |
| &emsp;R-Zero (Iter 2) |   38.45    |   53.84  |  **31.58**  | **58.20**  | 10.20|
| &emsp;R-Zero (Iter 3) |  **38.73**   |  **54.69**  |    31.38  |   **58.23**  |  **10.60**|
| ***OctoThinker-3B*** |             |          |           |          |      |
| &emsp;Base Model   |    12.27    |   26.64  |    10.09  |   10.87  | 1.46 |
| &emsp;Base Challenger |   14.41    |   27.51  |    11.19  |   14.53  | **4.40** |
| &emsp;R-Zero (Iter 1) |   14.93    |   27.76  |    12.21  |   15.72  | 4.05 |
| &emsp;R-Zero (Iter 2) |   15.11    |   28.20  |    12.43  |   16.08  | 3.74 |
| &emsp;R-Zero (Iter 3) |  **15.67**   |  **29.32**  |  **12.44**  |  **16.71**   | 4.20 |
| ***OctoThinker-8B*** |             |          |           |          |      |
| &emsp;Base Model   |    16.81    |   32.11  |    13.26  |   20.21  | 1.64 |
| &emsp;Base Challenger |   25.08    |   36.41  |    16.99  |   41.46  | 5.46 |
| &emsp;R-Zero (Iter 1) |   26.44    |   37.80  |    19.15  |  **42.05**  | 6.77 |
| &emsp;R-Zero (Iter 2) |   26.77    |   38.23  |    19.27  |   41.34  | **8.25** |
| &emsp;R-Zero (Iter 3) |  **26.88**   |  **38.52**  |  **19.82**  |   40.92  |  **8.25**|

## ❓ FAQ

### Hardware Setup

All experiments were conducted on an 8-GPU server. You might need to modify the code for larger models or different hardware configurations.

### Environment Configuration

Refer to [EasyR1](https://github.com/hiyouga/EasyR1/tree/main) for environment setup guidance or consider using their Docker environment.

### Saving Locations

All generated data, including logs, datasets, and model checkpoints, is saved in the directory specified by the `STORAGE_PATH` environment variable.  Datasets are also sent to Hugging Face via `HUGGINGFACENAME`.

### Code Stuck During Questioner Training

A timeout mechanism has been implemented to address potential infinite loops caused by a bug in the `math_verify` library. If the training appears to be stuck, simply restart from the last saved checkpoint.

## 🙏 Acknowledgements

This project is built upon the excellent work of [**EasyR1**](https://github.com/hiyouga/EasyR1/tree/main) and references the evaluation methodology of [**General-Reasoner**](https://github.com/TIGER-AI-Lab/General-Reasoner).

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