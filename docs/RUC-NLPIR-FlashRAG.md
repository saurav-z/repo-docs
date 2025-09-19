# FlashRAG: Your Toolkit for Efficient Retrieval-Augmented Generation (RAG) Research

**Accelerate your RAG research with FlashRAG, a comprehensive Python toolkit designed for rapid prototyping, experimentation, and deployment of advanced RAG models.** ([Original Repo](https://github.com/RUC-NLPIR/FlashRAG))

<div align="center">
<a href="https://arxiv.org/abs/2405.13576" target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv></a>
<a href="https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/" target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Datasets-27b3b4.svg></a>
<a href="https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset" target="_blank"><img src=https://custom-icon-badges.demolab.com/badge/ModelScope%20Datasets-624aff?style=flat&logo=modelscope&logoColor=white></a>
<a href="https://deepwiki.com/RUC-NLPIR/FlashRAG"><img src="https://devin.ai/assets/deepwiki-badge.png" alt="DeepWiki Document" height="20"/></a>
<a href="https://github.com/RUC-NLPIR/FlashRAG/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green"></a>
<a><img alt="Static Badge" src="https://img.shields.io/badge/made_with-Python-blue"></a>
</div>


## Key Features

*   **Extensive and Customizable Framework:** Easily assemble complex RAG pipelines with modular components for retrievers, rerankers, generators, and more.
*   **Comprehensive Benchmark Datasets:** Access **36 pre-processed RAG benchmark datasets** for rigorous model testing and validation.
*   **Advanced RAG Algorithms:** Explore and reproduce **23 state-of-the-art RAG algorithms** with reported results, enabling efficient experimentation.
*   **🚀 Reasoning-Based Methods:**  **NEW!**  Leverage cutting-edge performance with support for **7 reasoning-based methods** that combine retrieval with advanced reasoning capabilities.
*   **Efficient Preprocessing:** Simplify your workflow with pre-built scripts for corpus processing, index building, and document retrieval.
*   **Optimized Execution:** Benefit from integrated tools like vLLM, FastChat, and Faiss for accelerated LLM inference and vector index management.
*   **User-Friendly UI:**  Quickly configure and experiment with different RAG baselines through our intuitive **FlashRAG-UI**.

## Getting Started

### Installation

Install FlashRAG with pip:

```bash
pip install flashrag-dev --pre
```

or clone the repository and install locally:

```bash
git clone https://github.com/RUC-NLPIR/FlashRAG.git
cd FlashRAG
pip install -e .
```

Refer to the original [README](https://github.com/RUC-NLPIR/FlashRAG#wrench-installation) for optional dependencies and detailed installation instructions.

### Quick Start

1.  **Corpus Construction:** Prepare your corpus as a `jsonl` file.
2.  **Index Construction:** Build retrieval indexes using provided scripts.
3.  **Ready-Made Pipeline:** Utilize the pre-built pipeline class for RAG processes by configuring the config and loading the corresponding pipeline.
4.  **Build Your Own Pipeline:** Customize RAG process logic by inheriting from the `BasicPipeline` class.
5.  **Component Usage:** Leverage the input/output formats to embed our components into existing code.

## Core Components

FlashRAG provides a modular architecture, offering a range of components for building RAG systems:

### RAG Components

*   Judger
*   Retriever
*   Reranker
*   Refiner
*   Generator

### Pipelines

FlashRAG categorizes RAG methods based on their inference paths and provides corresponding pipelines:

*   Sequential
*   Conditional
*   Branching
*   Loop

## FlashRAG-UI

Our intuitive user interface makes experimentation easy:

*   **One-Click Configuration Loading**
*   **Quick Method Experience**
*   **Efficient Benchmark Reproduction**

```bash
cd webui
python interface.py
```

## Supporting Methods and Results

We have implemented **23 RAG methods**, including those with complex reasoning, with the following baseline settings:

*   **Generator:** LLAMA3-8B-instruct with input length of 2048
*   **Retriever:** e5-base-v2 as embedding model, retrieve 5 docs per query
*   **Prompt:** A consistent default prompt.

| Method                                   | NQ (EM) | TriviaQA (EM) | Hotpotqa (F1) | ... |
| :--------------------------------------- | :------ | :------------ | :------------ | --- |
| Standard RAG                              | 35.1    | 58.9          | 35.3          | ... |
| [Spring](https://arxiv.org/abs/2405.19670) | 37.9    | 64.6          | 42.6          | ... |
| [Search-R1](https://arxiv.org/abs/2503.09516) | 45.2 | 62.2 | 54.5 | ... |
| [CoRAG](https://arxiv.org/abs/2503.21729) | 40.9 | 63.1 | 56.6 | ... |

Find detailed results and specific settings in the original [README](https://github.com/RUC-NLPIR/FlashRAG#robot-supporting-methods).

## Supporting Datasets

FlashRAG includes **36 preprocessed datasets** commonly used in RAG research. All datasets are available at [Huggingface datasets](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets).

## Citation

Please cite our paper if you use FlashRAG:

```bibtex
@article{FlashRAG,
  author       = {Jiajie Jin and
                  Yutao Zhu and
                  Xinyu Yang and
                  Chenghao Zhang and
                  Zhicheng Dou},
  title        = {FlashRAG: {A} Modular Toolkit for Efficient Retrieval-Augmented Generation
                  Research},
  journal      = {CoRR},
  volume       = {abs/2405.13576},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2405.13576},
  doi          = {10.48550/ARXIV.2405.13576},
  eprinttype    = {arXiv},
  eprint       = {2405.13576},
  timestamp    = {Tue, 18 Jun 2024 09:26:37 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2405-13576.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

**[Explore FlashRAG on GitHub for a deeper dive into its features and functionality.](https://github.com/RUC-NLPIR/FlashRAG)**