# R-Zero: Revolutionizing LLM Reasoning with Self-Evolving Intelligence

**R-Zero empowers Large Language Models to learn and improve their reasoning abilities autonomously, starting from scratch without any pre-existing data.** ([Original Repository](https://github.com/Chengsong-Huang/R-Zero))

R-Zero is a groundbreaking framework that allows LLMs to evolve their reasoning capabilities without the need for extensive, manually curated datasets. It achieves this through a self-evolving system, a dynamic co-evolutionary loop.

## Key Features

*   **Zero-Shot Learning:** Train reasoning models from scratch, eliminating the need for pre-existing datasets or human-labeled examples.
*   **Co-Evolutionary Architecture:** Employs a Challenger-Solver loop, where the Challenger generates challenging problems, driving continuous improvement in the Solver.
*   **Adaptive Curriculum:** The Challenger learns to create increasingly difficult tasks, creating a tailored learning path for the Solver.
*   **Proven Performance:** Delivers significant performance improvements across various reasoning benchmarks.
*   **Model-Agnostic:** Enhances the reasoning abilities of various backbone LLMs.

## Updates

*   **2025-08-27:** Analysis on iteration scaling and one model taking on two roles added.
*   **2025-08-25:** Code updates for smoother training (stopit implementation).
*   **2025-08-08:** Received `#2 Paper of the day` in Hugging Face Daily Paper.
*   **2025-08-07:** Released paper and code.

## Quickstart Guide

Get started with R-Zero quickly:

### 1. Set Up Your Environment

```bash
git clone https://github.com/Chengsong-Huang/R-Zero.git
cd R-Zero
pip install -r requirements.txt
export STORAGE_PATH="/path/to/your/storage"
export HUGGINGFACENAME="yourhuggingfacename"

mkdir -p "$STORAGE_PATH/evaluation" "$STORAGE_PATH/models" "$STORAGE_PATH/generated_question" "$STORAGE_PATH/temp_results"
```

### 2. Configure API Keys

*   Add your Hugging Face and WandB API keys to `tokens.json`.
*   Include your OpenAI GPT API key in `evaluation/results_recheck.py`.

### 3. Run Experiments

Replicate results using a single script:

```bash
# Format: bash scripts/main.sh [Base_Model_Name] [Abbreviation]
# Example:
bash scripts/main.sh Qwen/Qwen3-4B-Base qwen3-4b
```

## Impressive Results

R-Zero demonstrates substantial improvements over baseline models:

| Model Name           | Overall AVG | MATH AVG | SuperGPQA | MMLU-Pro | BBEH   |
| :------------------- | :----------: | :------: | :--------: | :------: | :-----: |
| ***Qwen3-4B-Base***  |      ...     |    ...   |     ...    |    ...   |   ...   |
| &emsp;Base Model     |     27.10    |   42.58  |    20.88   |   37.38  |  7.57   |
| &emsp;Base Challenger |     30.83    |   44.36  |    24.77   |   47.59  |  6.59   |
| &emsp;R-Zero (Iter 1) |     34.27    |   48.06  |   **27.92**  |   51.69  |  9.42   |
| &emsp;R-Zero (Iter 2) |   **34.92**  |   48.44  |    27.72   |  **53.75**  |  9.76   |
| &emsp;R-Zero (Iter 3) |     34.64    |  **49.07** |    27.55   |   51.53  | **10.42** |
| ***Qwen3-8B-Base***  |      ...     |    ...   |     ...    |    ...   |   ...   |
| &emsp;Base Model     |     34.49    |   49.18  |    28.33   |   51.80  |  8.63   |
| &emsp;Base Challenger |     36.43    |   51.87  |    30.12   |   54.14  |  9.60   |
| &emsp;R-Zero (Iter 1) |     37.93    |   53.39  |    31.26   |   57.17  |  9.91   |
| &emsp;R-Zero (Iter 2) |     38.45    |   53.84  |   **31.58**  |   58.20  | 10.20   |
| &emsp;R-Zero (Iter 3) |   **38.73**  |  **54.69** |    31.38   |  **58.23**  | **10.60** |
| ***OctoThinker-3B*** |      ...     |    ...   |     ...    |    ...   |   ...   |
| &emsp;Base Model     |     12.27    |   26.64  |    10.09   |   10.87  |  1.46   |
| &emsp;Base Challenger |     14.41    |   27.51  |    11.19   |   14.53  |  **4.40**  |
| &emsp;R-Zero (Iter 1) |     14.93    |   27.76  |    12.21   |   15.72  |  4.05   |
| &emsp;R-Zero (Iter 2) |     15.11    |   28.20  |    12.43   |   16.08  |  3.74   |
| &emsp;R-Zero (Iter 3) |   **15.67**  |  **29.32** |   **12.44**  |  **16.71**  |  4.20   |
| ***OctoThinker-8B*** |      ...     |    ...   |     ...    |    ...   |   ...   |
| &emsp;Base Model     |     16.81    |   32.11  |    13.26   |   20.21  |  1.64   |
| &emsp;Base Challenger |     25.08    |   36.41  |    16.99   |   41.46  |  5.46   |
| &emsp;R-Zero (Iter 1) |     26.44    |   37.80  |    19.15   |   **42.05**  |  6.77   |
| &emsp;R-Zero (Iter 2) |     26.77    |   38.23  |    19.27   |   41.34  |  **8.25**  |
| &emsp;R-Zero (Iter 3) |   **26.88**  |  **38.52** |   **19.82**  |   40.92  |  **8.25**  |

## FAQ for Developers

**Q: What hardware is needed for the experiments?**

**A:** Experiments were run on an 8-GPU server.  Adapt the code for larger models or different hardware.

**Q: What if I have environment configuration issues?**

**A:** Check out [EasyR1](https://github.com/hiyouga/EasyR1/tree/main) for their setup instructions or Docker environment as a reference.

**Q: Where are the training logs and model checkpoints saved?**

**A:**  Generated data, including logs, datasets, and model checkpoints, are saved in the `STORAGE_PATH`. Datasets are also sent to Hugging Face via `HUGGINGFACENAME`.

**Q: What if the code gets stuck during the questioner training process?**

**A:** Restart training from the last saved checkpoint.

## Acknowledgements

We are grateful to the work of [**EasyR1**](https://github.com/hiyouga/EasyR1/tree/main) and [**General-Reasoner**](https://github.com/TIGER-AI-Lab/General-Reasoner).

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