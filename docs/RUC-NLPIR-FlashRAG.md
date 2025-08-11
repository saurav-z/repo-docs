# FlashRAG: The Ultimate Python Toolkit for Efficient Retrieval-Augmented Generation (RAG) Research

**Tired of struggling with complex RAG implementations?** FlashRAG is a powerful and versatile Python toolkit designed to accelerate your research in Retrieval-Augmented Generation.  **[Check out the original repository](https://github.com/RUC-NLPIR/FlashRAG) to dive deeper!**

<div align="center">
<a href="https://arxiv.org/abs/2405.13576" target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv></a>
<a href="https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/" target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Datasets-27b3b4.svg></a>
<a href="https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset" target="_blank"><img src=https://custom-icon-badges.demolab.com/badge/ModelScope%20Datasets-624aff?style=flat&logo=modelscope&logoColor=white></a>
<a href="https://deepwiki.com/RUC-NLPIR/FlashRAG"><img src="https://devin.ai/assets/deepwiki-badge.png" alt="DeepWiki Document" height="20"/></a>
<a href="https://github.com/RUC-NLPIR/FlashRAG/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green"></a>
<a><img alt="Static Badge" src="https://img.shields.io/badge/made_with-Python-blue"></a>
</div>

## Key Features

*   **Modular and Customizable**:  Assemble complex RAG pipelines with ease using essential components for retrievers, rerankers, generators, and more.
*   **Extensive Benchmark Datasets**: Test and validate your models with 36 pre-processed RAG benchmark datasets.
*   **State-of-the-Art Algorithms**:  Reproduce results with 23 pre-implemented advanced RAG algorithms.
*   **Reasoning-Based Methods**:  **NEW!** Explore 7 cutting-edge reasoning-based methods that boost performance on complex tasks.
*   **Simplified Workflow**: Efficient preprocessing steps for corpus handling, indexing, and document retrieval.
*   **Optimized Performance**: Leverage tools like vLLM, FastChat, and Faiss for faster execution.
*   **User-Friendly Interface**:  Easily configure, experiment, and evaluate RAG baselines with the FlashRAG-UI.

## Sections
*   [Installation](#wrench-installation)
*   [Quick Start](#rocket-quick-start)
*   [Components](#gear-components)
*   [FlashRAG-UI](#art-flashrag-ui)
*   [Supporting Methods](#robot-supporting-methods)
*   [Supporting Datasets & Document Corpus](#notebook-supporting-datasets--document-corpus)
*   [Additional FAQs](#raised_hands-additional-faqs)
*   [License](#bookmark-license)
*   [Citation](#star2-citation)

## :wrench: Installation

Install FlashRAG quickly using pip:

```bash
pip install flashrag-dev --pre
```

For detailed installation instructions, including optional dependencies (vLLM, sentence-transformers, Pyserini), see the [Installation section of the original README](https://github.com/RUC-NLPIR/FlashRAG#wrench-installation).

## :rocket: Quick Start

Jumpstart your RAG experiments. Steps include :
*   Corpus Construction and Formatting
*   Index Building and configuration
*   Using the Ready-Made Pipeline
*   Building your own pipeline
*   Component usage

See the [Quick Start section of the original README](https://github.com/RUC-NLPIR/FlashRAG#rocket-quick-start) for detailed guidance.

## :gear: Components

FlashRAG offers a comprehensive suite of RAG components, including:

*   **Judger**:  SKR Judger
*   **Retriever**: Dense Retriever, BM25 Retriever, Bi-Encoder Reranker, Cross-Encoder Reranker
*   **Refiner**: Extractive Refiner, Abstractive Refiner, LLMLingua Refiner, SelectiveContext Refiner, KG Refiner
*   **Generator**: Encoder-Decoder Generator, Decoder-only Generator, FastChat Generator, vllm Generator

These components can be combined to create various [Pipelines](https://github.com/RUC-NLPIR/FlashRAG#pipelines).

## :art: FlashRAG-UI

The user-friendly FlashRAG-UI allows you to configure, experiment, and evaluate RAG methods visually.

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

To start the UI, run:
```bash
cd webui
python interface.py
```

See screenshots and more details in the [FlashRAG-UI section of the original README](https://github.com/RUC-NLPIR/FlashRAG#art-flashrag-ui).

## :robot: Supporting Methods

FlashRAG includes implementations of 23 RAG methods, with consistent settings for easy comparison. We have implemented **23 works** with a consistent setting of:

-   **Generator:** LLAMA3-8B-instruct with input length of 2048
-   **Retriever:** e5-base-v2 as embedding model, retrieve 5 docs per query
-   **Prompt:** A consistent default prompt, template can be found in the [<u>method details</u>](./docs/original_docs/baseline_details.md).

It's important to note that, to ensure consistency, we have utilized a uniform setting. However, this setting may differ from the original setting of the method, leading to variations in results compared to the original outcomes.

### :rocket: Reasoning-based Methods (NEW!)

Support for 7 reasoning-based methods for multi-hop tasks.

See the [Supporting Methods section of the original README](https://github.com/RUC-NLPIR/FlashRAG#robot-supporting-methods) for a detailed performance table.

## :notebook: Supporting Datasets & Document Corpus

FlashRAG supports 36 RAG datasets and offers a versatile approach for using your own datasets.

### Datasets

All datasets are available on [Hugging Face](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets). Dataset structure in jsonl format.

### Document Corpus

Supports jsonl format for document collections: `{"id":"0", "contents": "..."}`. We offer comprehensive tools for processing and indexing your data, including a script to process Wikipedia dumps.

### Index

Preprocessed index: [FlashRAG\_Dataset/retrieval\_corpus/wiki18\_100w\_e5\_index.zip](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/file/view/master?id=47985&status=2&fileName=retrieval_corpus%252Fwiki18_100w_e5_index.zip).

For more information, see the [Supporting Datasets & Document Corpus section of the original README](https://github.com/RUC-NLPIR/FlashRAG#notebook-supporting-datasets--document-corpus).

## :raised_hands: Additional FAQs

Find answers to common questions:

*   [Configuration settings](docs/original_docs/configuration.md)
*   [Building custom corpora](docs/original_docs/process-wiki.md)
*   [Indexing your corpus](docs/original_docs/building-index.md)
*   [Reproducing supporting methods](docs/original_docs/reproduce_experiment.md)

## :bookmark: License

FlashRAG is licensed under the [MIT License](./LICENSE).

## :star2: Citation

If you use FlashRAG in your research, please cite our paper:

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