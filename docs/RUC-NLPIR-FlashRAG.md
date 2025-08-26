# FlashRAG: Your Toolkit for Cutting-Edge Retrieval Augmented Generation (RAG) Research

**Unleash the power of Retrieval Augmented Generation with FlashRAG, a Python toolkit empowering you to effortlessly reproduce state-of-the-art research and build your own custom RAG pipelines.**  Explore advanced RAG algorithms, benchmark datasets, and a user-friendly UI – all designed for efficient and effective experimentation.

[Original Repo](https://github.com/RUC-NLPIR/FlashRAG)

[![arXiv](https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv)](https://arxiv.org/abs/2405.13576)
[![HuggingFace Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Datasets-27b3b4.svg)](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/)
[![ModelScope Datasets](https://custom-icon-badges.demolab.com/badge/ModelScope%20Datasets-624aff?style=flat&logo=modelscope&logoColor=white)](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset)
[![DeepWiki Document](https://devin.ai/assets/deepwiki-badge.png)](https://deepwiki.com/RUC-NLPIR/FlashRAG)
[![License](https://img.shields.io/badge/LICENSE-MIT-green)](https://github.com/RUC-NLPIR/FlashRAG/blob/main/LICENSE)
[![Made with Python](https://img.shields.io/badge/made_with-Python-blue)](https://www.python.org/)
[![Trendshift](https://trendshift.io/api/badge/repositories/10454)](https://trendshift.io/repositories/10454)

## Key Features

*   **Comprehensive Framework:** Build RAG pipelines with flexible components including retrievers, rerankers, generators, and compressors.
*   **Extensive Benchmark Datasets:**  Utilize **36 pre-processed RAG datasets** for rigorous model evaluation and comparison.
*   **Advanced RAG Algorithms:**  Reproduce results with **23 pre-implemented, state-of-the-art RAG algorithms**, including:
    *   **NEW!**  **7 reasoning-based methods** that combine retrieval and reasoning for enhanced performance on complex tasks.
*   **Efficient Workflow:** Streamline your RAG workflow with pre-built scripts for corpus processing, index building, and document retrieval.
*   **Optimized Performance:** Leverage vLLM, FastChat, and Faiss for accelerated LLM inference and vector index management.
*   **User-Friendly UI:** Easily configure and experiment with different RAG baselines using our intuitive graphical interface.

## Core Sections

*   [Installation](#wrench-installation)
*   [Quick Start](#rocket-quick-start)
*   [Features](#sparkles-features)
*   [Components](#gear-components)
*   [FlashRAG-UI](#art-flashrag-ui)
*   [Supporting Methods](#robot-supporting-methods)
*   [Supporting Datasets & Document Corpus](#notebook-supporting-datasets--document-corpus)
*   [Roadmap](#mag_right-roadmap)
*   [Changelog](#page_with_curl-changelog)
*   [FAQs](#raised_hands-additional-faqs)
*   [Citation](#star2-citation)
*   [License](#bookmark-license)

## :wrench: Installation

```bash
pip install flashrag-dev --pre
```

For more detailed installation instructions, including optional dependencies like vLLM, sentence-transformers, and faiss, refer to the [Installation](#wrench-installation) section in the original README.

## :rocket: Quick Start

Quickly get started with [Quick Start](#rocket-quick-start) section.

## :gear: Components

[Components](#gear-components) section provides a detailed overview of the available RAG components, including retrievers, generators, and refiners, along with the pipelines that assemble these components.

## :art: FlashRAG-UI

[FlashRAG-UI](#art-flashrag-ui) provides an easy-to-use interface for configuring, experimenting with, and evaluating RAG methods, significantly streamlining research efforts.

## :robot: Supporting Methods

Evaluate and compare your results against the performances of **23 state-of-the-art RAG algorithms** and **7 Reasoning-based Methods** which have shown the best results in different benchmarks. See the [Supporting Methods](#robot-supporting-methods) section.

## :notebook: Supporting Datasets & Document Corpus

Access and utilize **36 pre-processed RAG datasets** and supporting document corpora to facilitate your research and ensure standardized evaluation. All datasets are available at [<u>Huggingface datasets</u>](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets).

## :mag_right: Roadmap

Explore the future directions and planned enhancements for FlashRAG, including supporting more RAG approaches, improving code adaptability, and adding API-based retrievers. See the [Roadmap](#mag_right-roadmap) section.

## :page_with_curl: Changelog

Stay informed about the latest updates, features, and improvements to FlashRAG. See the [Changelog](#page_with_curl-changelog) section.

## :raised_hands: Additional FAQs

Find answers to frequently asked questions, covering topics like parameter settings, corpus building, index creation, and reproducing supporting methods. Access the [FAQs](#raised_hands-additional-faqs) section for more information.

## :bookmark: License

FlashRAG is licensed under the [MIT License](./LICENSE).

## :star2: Citation

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

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=RUC-NLPIR/FlashRAG&type=Date)](https://star-history.com/#RUC-NLPIR/FlashRAG&Date)