# ⚡ FlashRAG: A Python Toolkit for Efficient RAG Research

> **Tired of the complexities of RAG research?** FlashRAG provides a modular and powerful Python toolkit to simplify and accelerate your Retrieval-Augmented Generation (RAG) experiments, allowing you to easily reproduce state-of-the-art results and build your own custom RAG pipelines.

<div align="center">
<a href="https://arxiv.org/abs/2405.13576" target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv></a>
<a href="https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/" target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Datasets-27b3b4.svg></a>
<a href="https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset" target="_blank"><img src=https://custom-icon-badges.demolab.com/badge/ModelScope%20Datasets-624aff?style=flat&logo=modelscope&logoColor=white></a>
<a href="https://deepwiki.com/RUC-NLPIR/FlashRAG"><img src="https://devin.ai/assets/deepwiki-badge.png" alt="DeepWiki Document" height="20"/></a>
<a href="https://github.com/RUC-NLPIR/FlashRAG/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green"></a>
<a><img alt="Static Badge" src="https://img.shields.io/badge/made_with-Python-blue"></a>
</div>

<h4 align="center">
<a href="#wrench-installation">Installation</a> |
<a href="#sparkles-features">Features</a> |
<a href="#rocket-quick-start">Quick-Start</a> |
<a href="#gear-components"> Components</a> |
<a href="#art-flashrag-ui"> FlashRAG-UI</a> |
<a href="#robot-supporting-methods"> Supporting Methods</a> |
<a href="#notebook-supporting-datasets--document-corpus"> Supporting Datasets</a> |
<a href="#raised_hands-additional-faqs"> FAQs</a>
</h4>

<p align="center">
<img src="asset/framework.jpg">
</p>

FlashRAG is your all-in-one solution for RAG research, offering a comprehensive framework and valuable resources to accelerate your work.  The toolkit includes:

*   **36 pre-processed benchmark RAG datasets**
*   **23 state-of-the-art RAG algorithms**
*   **7 reasoning-based methods**

Easily reproduce existing SOTA works, implement custom RAG processes, and contribute to the RAG research community.  Plus, enjoy the user-friendly **FlashRAG-UI** for easy experimentation!

[View the original repository](https://github.com/RUC-NLPIR/FlashRAG)

## Key Features

*   ✨ **Extensive and Customizable Framework**:  Build complex RAG pipelines with modular components: retrievers, rerankers, generators, and compressors.
*   📚 **Comprehensive Benchmark Datasets**:  Evaluate your models on a rich collection of 36 pre-processed RAG benchmark datasets.
*   🚀 **Pre-implemented Advanced RAG Algorithms**:  Reproduce results and experiment with **23 pre-implemented RAG algorithms**, including **7 reasoning-based methods** for advanced performance.
*   🧠 **Reasoning-based Methods**: Leverage innovative methods combining reasoning with retrieval to enhance performance on complex multi-hop tasks.
*   ⚙️ **Efficient Preprocessing**: Streamline your RAG workflow with tools for corpus processing, index building, and document pre-retrieval.
*   ⚡ **Optimized Execution**:  Enhance library efficiency with vLLM, FastChat, and Faiss integration.
*   🖼️ **Easy-to-Use UI**:  Experiment and visualize results through the intuitive and user-friendly FlashRAG-UI.

## Quick Links

*   [Installation](#wrench-installation)
*   [Quick Start](#rocket-quick-start)
*   [Components](#gear-components)
*   [FlashRAG-UI](#art-flashrag-ui)
*   [Supporting Methods](#robot-supporting-methods)
*   [Supporting Datasets & Document Corpus](#notebook-supporting-datasets--document-corpus)
*   [Additional FAQs](#raised_hands-additional-faqs)

## Installation

```bash
pip install flashrag-dev --pre
```

See the original README for additional installation options and dependencies.

## [View the original repository](https://github.com/RUC-NLPIR/FlashRAG)

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