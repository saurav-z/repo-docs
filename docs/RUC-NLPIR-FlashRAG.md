# ⚡ FlashRAG: A Python Toolkit for Efficient RAG Research

**Effortlessly advance your Retrieval-Augmented Generation (RAG) research with FlashRAG, a versatile Python toolkit packed with pre-implemented algorithms, datasets, and a user-friendly UI.**

[English | [中文](README_zh.md) ]

<div align="center">
    <a href="https://arxiv.org/abs/2405.13576" target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv></a>
    <a href="https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/" target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Datasets-27b3b4.svg></a>
    <a href="https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset" target="_blank"><img src=https://custom-icon-badges.demolab.com/badge/ModelScope%20Datasets-624aff?style=flat&logo=modelscope&logoColor=white></a>
    <a href="https://deepwiki.com/RUC-NLPIR/FlashRAG"><img src="https://devin.ai/assets/deepwiki-badge.png" alt="DeepWiki Document" height="20"/></a>
    <a href="https://github.com/RUC-NLPIR/FlashRAG/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green"></a>
    <a><img alt="Static Badge" src="https://img.shields.io/badge/made_with-Python-blue"></a>
</div>

<h4 align="center">

<p>
<a href="#wrench-installation">Installation</a> |
<a href="#sparkles-features">Features</a> |
<a href="#rocket-quick-start">Quick-Start</a> |
<a href="#gear-components"> Components</a> |
<a href="#art-flashrag-ui"> FlashRAG-UI</a> |
<a href="#robot-supporting-methods"> Supporting Methods</a> |
<a href="#notebook-supporting-datasets--document-corpus"> Supporting Datasets</a> |
<a href="#raised_hands-additional-faqs"> FAQs</a>
</p>

</h4>

FlashRAG empowers researchers and developers to explore and advance the state-of-the-art in Retrieval Augmented Generation (RAG). This comprehensive toolkit offers a streamlined approach to RAG research, providing:

*   36 pre-processed benchmark RAG datasets
*   23 state-of-the-art RAG algorithms, including 7 reasoning-based methods.

[View the original repository on GitHub](https://github.com/RUC-NLPIR/FlashRAG)

<p align="center">
<img src="asset/framework.jpg">
</p>

With FlashRAG and its integrated resources, you can effortlessly reproduce existing SOTA RAG research or implement your custom RAG processes and components.

<p>
<a href="https://trendshift.io/repositories/10454" target="_blank"><img src="https://trendshift.io/api/badge/repositories/10454" alt="RUC-NLPIR%2FFlashRAG | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

## Key Features

*   **Extensive and Customizable Framework:** Build complex RAG pipelines with modular components (retrievers, rerankers, generators, compressors).
*   **Comprehensive Benchmark Datasets:** Test and validate RAG models with 36 pre-processed benchmark datasets.
*   **Pre-implemented Advanced RAG Algorithms:** Easily reproduce results using **23 state-of-the-art RAG algorithms.**
*   **🚀 Reasoning-based Methods (NEW!):** Achieve superior performance on complex multi-hop tasks with our 7 reasoning-based methods.
*   **Efficient Workflow:** Streamline your RAG workflow with easy-to-use preprocessing tools.
*   **Optimized Execution:** Leverage vLLM, FastChat and Faiss for enhanced speed.
*   **Easy-to-Use UI:** Quickly configure and experiment with supported RAG methods using our intuitive user interface [FlashRAG-UI](#art-flashrag-ui).

## :mag_right: Roadmap
FlashRAG is a evolving project.
- [ ] Inlcude more RAG approaches
- [ ] Enhance code adaptability and readability
- [ ] Add support for api-based retriever (vllm server)

## :page_with_curl: Changelog
*   \[25/08/06] 🎯 **NEW!** Added support for **Reasoning Pipeline**, a new paradigm that combines reasoning ability and retrieval.
*   \[25/03/21] 🚀 **Major Update!** Expanded toolkit to support **23 state-of-the-art RAG algorithms**, including **7 reasoning-based methods**.
*   \[25/02/24] 🔥🔥🔥 Added support for **multimodal RAG**, including MLLMs and multimodal retrievers.

*See the original README for more changelog entries.*

## :wrench: Installation

FlashRAG is easy to install using pip:

```bash
pip install flashrag-dev --pre
```

or:

```bash
git clone https://github.com/RUC-NLPIR/FlashRAG.git
cd FlashRAG
pip install -e .
```

*See the original README for optional dependency instructions*

## :rocket: Quick Start

### Corpus Construction
*See the original README for details on Corpus Construction and Index Construction*

### Index Construction

*See the original README for examples of using Dense Retrieval, Sparse Retrieval, and Sparse Neural Retrieval*

### Using the ready-made pipeline

*See the original README for instructions on using the ready-made pipeline*

### Build your own pipeline!

*See the original README for instructions on building your own pipeline*

### Just use components

*See the original README for instructions on using components*

## :gear: Components

*See the original README for information on the RAG-Components and Pipelines*

## :art: FlashRAG-UI
With FlashRAG-UI, you can easily and quickly configure and experience the supported RAG methods through our meticulously designed visual interface, and evaluate these methods on benchmarks, making complex research work more efficient!

*   **One-Click Configuration Loading** -  Load parameters and configuration files with ease.
*   **Quick Method Experience** - Explore characteristics of various RAG methods, and seamlessly connect different RAG Pipelines to quickly experience their performance and differences.
*   **Efficient Benchmark Reproduction** - Easily reproduce the built-in baseline methods and carefully collected benchmarks.

#### Experience our meticulously designed FlashRAG-UI—both user-friendly and visually appealing:
```bash
cd webui
python interface.py
```

*See the original README for UI images*

## :robot: Supporting Methods

We have implemented **23 works** with a consistent setting of:

-   **Generator:** LLAMA3-8B-instruct with input length of 2048
-   **Retriever:** e5-base-v2 as embedding model, retrieve 5 docs per query
-   **Prompt:** A consistent default prompt, template can be found in the [<u>method details</u>](./docs/original_docs/baseline_details.md).

*See the original README for the table of results*

#### 🚀 Reasoning-based Methods (NEW!)

*See the original README for the table of results*

## :notebook: Supporting Datasets & Document Corpus

### Datasets

*See the original README for the list of datasets*

### Document Corpus

*See the original README for information on Document Corpus and Index*

## :lollipop: Awesome Work using FlashRAG

*   [R1-Searcher](https://github.com/SsmallSong/R1-Searcher), a method that incentivizes the search capability in LLMs via reinforcement learning
*   [ReSearch](https://github.com/Agent-RL/ReSearch), a method that learns to reason with search for LLMs via reinforcement learning
*   [AutoCoA](https://github.com/ADaM-BJTU/AutoCoA), a method that internalizes chain-of-action generation into reasoning models

## :raised_hands: Additional FAQs

*See the original README for the list of FAQs*

## :bookmark: License

FlashRAG is licensed under the [<u>MIT License</u>](./LICENSE).

## :star2: Citation

Please kindly cite our paper if helps your research:

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

## Star History
*See the original README for Star History Chart*