# ⚡ FlashRAG: Your Toolkit for Cutting-Edge RAG Research 🚀

> **Unleash the power of Retrieval-Augmented Generation (RAG) with FlashRAG, a comprehensive Python toolkit designed for efficient research and development.  Easily reproduce state-of-the-art RAG methods and build your own custom pipelines.**

[**Original Repository**](https://github.com/RUC-NLPIR/FlashRAG)

[![arXiv](https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv)](https://arxiv.org/abs/2405.13576)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Datasets-27b3b4.svg)](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/)
[![ModelScope Datasets](https://custom-icon-badges.demolab.com/badge/ModelScope%20Datasets-624aff?style=flat&logo=modelscope&logoColor=white)](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset)
[![DeepWiki Document](https://devin.ai/assets/deepwiki-badge.png)](https://deepwiki.com/RUC-NLPIR/FlashRAG)
[![License](https://img.shields.io/badge/LICENSE-MIT-green)](https://github.com/RUC-NLPIR/FlashRAG/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/made_with-Python-blue)](https://www.python.org/)

<p align="center">
  <a href="#wrench-installation">Installation</a> |
  <a href="#sparkles-features">Features</a> |
  <a href="#rocket-quick-start">Quick-Start</a> |
  <a href="#gear-components"> Components</a> |
  <a href="#art-flashrag-ui"> FlashRAG-UI</a> |
  <a href="#robot-supporting-methods"> Supporting Methods</a> |
  <a href="#notebook-supporting-datasets--document-corpus"> Supporting Datasets</a> |
  <a href="#raised_hands-additional-faqs"> FAQs</a>
</p>

FlashRAG empowers researchers and developers to explore, experiment, and advance the field of Retrieval Augmented Generation (RAG).  It provides a modular framework with pre-built components, datasets, and algorithms to accelerate your RAG research.  

<p align="center">
<img src="asset/framework.jpg">
</p>

Access a user-friendly UI for easy configuration and evaluation:
https://github.com/user-attachments/assets/8ca00873-5df2-48a7-b853-89e7b18bc6e9

[![RUC-NLPIR/FlashRAG | Trendshift](https://trendshift.io/api/badge/repositories/10454)](https://trendshift.io/repositories/10454)

## Table of Contents

*   [✨ Key Features](#sparkles-features)
*   [🧭 Roadmap](#mag_right-roadmap)
*   [📜 Changelog](#page_with_curl-changelog)
*   [🛠️ Installation](#wrench-installation)
*   [🚀 Quick Start](#rocket-quick-start)
*   [⚙️ Components](#gear-components)
*   [🎨 FlashRAG-UI](#art-flashrag-ui)
*   [🤖 Supporting Methods](#robot-supporting-methods)
*   [📚 Supporting Datasets & Document Corpus](#notebook-supporting-datasets--document-corpus)
*   [❓ FAQs](#raised_hands-additional-faqs)

## ✨ Key Features

*   **Modular and Extensible:** Build complex RAG pipelines with flexible components like retrievers, rerankers, and generators.
*   **Comprehensive Datasets:**  Access 36 pre-processed benchmark RAG datasets for rigorous evaluation.
*   **Advanced Algorithms:** Leverage **23 state-of-the-art RAG algorithms** with reported results, facilitating result reproduction.
*   **🧠 Reasoning-based Methods:**  **NEW!** Explore **7 reasoning-based methods** that combine retrieval and reasoning capabilities.
*   **Efficient Preprocessing:** Simplify your workflow with ready-to-use scripts for corpus processing, indexing, and document retrieval.
*   **Optimized Performance:** Utilize libraries like vLLM, FastChat, and Faiss for accelerated LLM inference and vector index management.
*   **Easy-to-Use UI:**  Visually configure, test, and evaluate RAG methods with our intuitive user interface.

## 🧭 Roadmap

FlashRAG is continuously evolving; contributions are welcome!

*   [x] Support OpenAI models
*   [x] Provide instructions for each component
*   [x] Integrate sentence Transformers
*   [x] Support multimodal RAG
*   [x] Support reasoning-based methods
*   [ ] Include more RAG approaches
*   [ ] Enhance code adaptability and readability
*   [ ] Add support for api-based retriever (vllm server)

## 📜 Changelog

**Recent Updates:**

*   **[25/08/06] 🎯 NEW!**  Added support for **Reasoning Pipeline**, enhancing performance on multi-hop tasks.
*   **[25/03/21] 🚀 Major Update!** Expanded toolkit to include **23 state-of-the-art RAG algorithms**, including **7 reasoning-based methods**.
*   **[25/02/24] 🔥🔥🔥** Added support for **multimodal RAG** including multimodal LLMs and retrievers.

**(See original README for complete changelog)**

## 🛠️ Installation

```bash
pip install flashrag-dev --pre
```

**(See original README for detailed installation instructions, optional dependencies, and Faiss installation notes.)**

## 🚀 Quick Start

**(See original README for detailed quick start instructions, including Corpus construction, Index Construction (Dense, Sparse, and SPLADE), Pipeline usage, building custom pipelines and component usage.)**

## ⚙️ Components

FlashRAG offers modular components and pipelines for flexible RAG implementation:

**(See original README for detailed table of RAG-Components and Pipelines)**

## 🎨 FlashRAG-UI

**FlashRAG-UI** offers a user-friendly interface to streamline RAG research:

*   **One-Click Configuration:** Load parameters and configuration files with ease.
*   **Quick Method Experience:**  Experiment with corpora, indexes, and various RAG methods.
*   **Efficient Benchmark Reproduction:**  Reproduce and evaluate built-in baselines with ease.

**(See original README for demo images and installation instructions.)**

```bash
cd webui
python interface.py
```

## 🤖 Supporting Methods

FlashRAG supports **23 RAG methods** with consistent settings for easy comparison:

**(See original README for detailed table of Supporting Methods including,  type, metrics and specific settings.  Also includes a table for 7 reasoning-based methods)**

## 📚 Supporting Datasets & Document Corpus

FlashRAG provides extensive dataset and corpus support:

### Datasets

36 pre-processed datasets are available at [Huggingface datasets](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets).

**(See original README for detailed dataset list and structure.)**

### Document Corpus

FlashRAG supports the jsonl format.  For Wikipedia, a comprehensive processing script is provided.

**(See original README for file structure and MS MARCO details.)**

### Index

Preprocessed index (e5-base-v2) for wiki18_100w available on ModelScope:  [FlashRAG_Dataset/retrieval_corpus/wiki18_100w_e5_index.zip](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/file/view/master?id=47985&status=2&fileName=retrieval_corpus%252Fwiki18_100w_e5_index.zip).

## ❓ FAQs

**(See original README for a list of FAQs and their corresponding links.)**

## :bookmark: License

FlashRAG is licensed under the [<u>MIT License</u>](./LICENSE).

## :star2: Citation

```BibTex
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