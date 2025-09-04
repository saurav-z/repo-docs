# FlashRAG: Your Toolkit for Cutting-Edge Retrieval Augmented Generation (RAG) Research

**Tackle RAG research challenges head-on!** FlashRAG is a powerful Python toolkit designed for the efficient development, reproduction, and exploration of Retrieval Augmented Generation (RAG) models.  With a rich collection of datasets, pre-implemented algorithms, and an easy-to-use UI, FlashRAG empowers you to push the boundaries of RAG.

[Visit the original repository for more details](https://github.com/RUC-NLPIR/FlashRAG)

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

<p align="center">
<img src="asset/framework.jpg">
</p>

## Key Features

*   **Extensive and Customizable Framework:** Build complex RAG pipelines with modular components like retrievers, rerankers, generators, and compressors.
*   **Comprehensive Benchmark Datasets:** Evaluate your RAG models on a diverse set of 36 pre-processed RAG benchmark datasets.
*   **Pre-implemented Advanced RAG Algorithms:** Easily reproduce results with 23 state-of-the-art RAG algorithms, including methods that combine reasoning ability with retrieval.
*   **🚀 Reasoning-Based Methods:** Explore **7 reasoning-based methods** that achieve superior performance on complex, multi-hop tasks.
*   **Efficient Preprocessing Stage:** Streamline your RAG workflow with pre-built scripts for corpus processing, index building, and document retrieval.
*   **Optimized Execution:** Leverage tools like vLLM and FastChat for LLM inference acceleration and Faiss for efficient vector index management.
*   **Easy-to-Use UI:**  A user-friendly interface (FlashRAG-UI) for quick configuration, experimentation, and evaluation of RAG models.

## Installation

Get started quickly with a simple pip install:

```bash
pip install flashrag-dev --pre
```
Or:

```bash
git clone https://github.com/RUC-NLPIR/FlashRAG.git
cd FlashRAG
pip install -e .
```

## Quick Start

[See original README for detailed Quick Start instructions.]

## Supporting Methods

FlashRAG provides pre-implemented results for 23 methods, including **7 reasoning-based methods**, to enable easy comparison and evaluation:

[Insert the supporting methods table from original README here]

## :art: FlashRAG-UI

With <strong>FlashRAG-UI</strong>, you can easily and quickly configure and experience the supported RAG methods through our meticulously designed visual interface, and evaluate these methods on benchmarks, making complex research work more efficient!

### :star2: Features
- **One-Click Configuration Loading**
  - You can load parameters and configuration files for various RAG methods through simple clicks, selections, and inputs.</li>
  - Supports preview interface for intuitive parameter settings.</li>
  - Provides save functionality to easily store configurations for future use.</li>
- **Quick Method Experience**
  - Quickly load corpora and index files to explore the characteristics and application scenarios of various RAG methods.</li>
  - Supports loading and switching different components and hyperparameters, seamlessly connecting different RAG Pipelines to quickly experience their performance and differences!</li>
- **Efficient Benchmark Reproduction**
  - Easily reproduce the built-in baseline methods and carefully collected benchmarks on FlashRAG-UI.</li>
  - Use cutting-edge research tools directly without complex settings, providing a smooth experience for your research work!</li>

#### Experience our meticulously designed FlashRAG-UI—both user-friendly and visually appealing:
```bash
cd webui
python interface.py
```

## Other Sections (Refer to the Original README for these Sections):
* Components
* Supporting Datasets & Document Corpus
* Additional FAQs
* License
* Citation
* Star History