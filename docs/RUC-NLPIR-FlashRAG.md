# FlashRAG: The Ultimate Python Toolkit for Efficient Retrieval-Augmented Generation (RAG) Research

**Tackle RAG research head-on with FlashRAG, a comprehensive toolkit designed for both researchers and developers.**  Reproduce state-of-the-art results, experiment with cutting-edge methods, and build your own RAG pipelines with ease.  [Explore the original repository](https://github.com/RUC-NLPIR/FlashRAG) for in-depth details.

<div align="center">
<a href="https://arxiv.org/abs/2405.13576" target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv></a>
<a href="https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/" target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Datasets-27b3b4.svg></a>
<a href="https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset" target="_blank"><img src=https://custom-icon-badges.demolab.com/badge/ModelScope%20Datasets-624aff?style=flat&logo=modelscope&logoColor=white></a>
<a href="https://deepwiki.com/RUC-NLPIR/FlashRAG"><img src="https://devin.ai/assets/deepwiki-badge.png" alt="DeepWiki Document" height="20"/></a>
<a href="https://github.com/RUC-NLPIR/FlashRAG/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green"></a>
<a><img alt="Static Badge" src="https://img.shields.io/badge/made_with-Python-blue"></a>
</div>

## Key Features

*   **Extensive & Customizable Framework:**  Build complex RAG pipelines with modular components for retrievers, rerankers, generators, and compressors.
*   **Comprehensive Benchmark Datasets:** Access 36 pre-processed RAG benchmark datasets for robust model evaluation.
*   **Pre-Implemented State-of-the-Art Algorithms:**  Reproduce and experiment with **23 advanced RAG algorithms**, including **7 reasoning-based methods** for complex tasks.
*   **Reasoning-Based Methods:**  Leverage cutting-edge techniques that combine reasoning with retrieval for improved performance, achieving superior performance on complex multi-hop tasks.
*   **Efficient Preprocessing:**  Simplify your RAG workflow with pre-built scripts for corpus processing, index building, and document retrieval.
*   **Optimized Execution:**  Benefit from integrations with vLLM, FastChat, and Faiss for accelerated LLM inference and vector index management.
*   **User-Friendly Interface:**  Easily configure and experiment with RAG baselines using the intuitive **FlashRAG-UI** (see below).

## Sections

*   [Installation](#wrench-installation)
*   [Quick Start](#rocket-quick-start)
*   [Components](#gear-components)
*   [FlashRAG-UI](#art-flashrag-ui)
*   [Supporting Methods](#robot-supporting-methods)
*   [Supporting Datasets & Document Corpus](#notebook-supporting-datasets--document-corpus)
*   [FAQs](#raised_hands-additional-faqs)
*   [Citation](#star2-citation)

## :wrench: Installation

Get started with FlashRAG quickly using pip:

```bash
pip install flashrag-dev --pre
```

Or install from source:

```bash
git clone https://github.com/RUC-NLPIR/FlashRAG.git
cd FlashRAG
pip install -e .
```

(Further installation instructions and optional dependencies detailed in the original README.)

## :rocket: Quick Start

(Concise instructions on corpus construction, index construction, and pipeline usage - as provided in the original README.)

## :gear: Components

(Summary of available RAG components, including retrievers, generators, refiners and pipelines, as described in the original README.)

## :art: FlashRAG-UI

**FlashRAG-UI** provides an easy-to-use interface to configure, test and visualize the results, making RAG research more efficient.

**Key benefits of FlashRAG-UI:**

*   **One-Click Configuration Loading**
*   **Quick Method Experience**
*   **Efficient Benchmark Reproduction**

(Include a brief description and example image)

```bash
cd webui
python interface.py
```

## :robot: Supporting Methods

FlashRAG implements **23 state-of-the-art methods** with standardized settings for easy comparison. This includes **7 reasoning-based methods** that significantly improves performance on complex reasoning tasks.

(Example of results table. Keep the header, show only key methods)

## :notebook: Supporting Datasets & Document Corpus

FlashRAG supports a wide range of datasets and corpora.

*   **Datasets:**  Includes 36 pre-processed RAG benchmark datasets.
*   **Document Corpus:**  Supports JSONL format; provides scripts for processing Wikipedia and other popular sources.

## :raised_hands: FAQs

(Include links to FAQs section as provided in the original README)

## :bookmark: License

FlashRAG is licensed under the [<u>MIT License</u>](./LICENSE).

## :star2: Citation

(Include the BibTex as provided in the original README)

## Star History

(Include the Star History Chart)