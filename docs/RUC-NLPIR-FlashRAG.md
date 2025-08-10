# FlashRAG: Your Toolkit for Efficient Retrieval Augmented Generation (RAG) Research

**Unlock the power of RAG with FlashRAG, a Python toolkit designed for effortless reproduction and cutting-edge development in the RAG domain!**  ([Original Repo](https://github.com/RUC-NLPIR/FlashRAG))

[![arXiv](https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv)](https://arxiv.org/abs/2405.13576)
[![HuggingFace Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Datasets-27b3b4.svg)](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/)
[![ModelScope Datasets](https://custom-icon-badges.demolab.com/badge/ModelScope%20Datasets-624aff?style=flat&logo=modelscope&logoColor=white)](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset)
[![DeepWiki Document](https://devin.ai/assets/deepwiki-badge.png)](https://deepwiki.com/RUC-NLPIR/FlashRAG)
[![License](https://img.shields.io/badge/LICENSE-MIT-green)](https://github.com/RUC-NLPIR/FlashRAG/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/made_with-Python-blue)](https://www.python.org/)

<p align="center">
  <a href="#installation">Installation</a> |
  <a href="#key-features">Key Features</a> |
  <a href="#quick-start">Quick Start</a> |
  <a href="#components">Components</a> |
  <a href="#flashrag-ui">FlashRAG-UI</a> |
  <a href="#supporting-methods">Supporting Methods</a> |
  <a href="#supporting-datasets">Supporting Datasets</a> |
  <a href="#faqs">FAQs</a>
</p>

FlashRAG provides a comprehensive framework for researchers to easily reproduce and develop state-of-the-art RAG models. It offers pre-processed benchmark datasets and a wide array of pre-implemented RAG algorithms, including reasoning-based methods.  Easily implement custom RAG processes and components using our library.

<p align="center">
  <img src="asset/framework.jpg" alt="FlashRAG Framework">
</p>

## Key Features

*   **Extensive Framework:** Includes essential RAG components like retrievers, rerankers, generators, and compressors for flexible pipeline assembly.
*   **36 Pre-processed Benchmark Datasets:** Test and validate your RAG models with a comprehensive collection of datasets.
*   **23 Pre-Implemented Advanced RAG Algorithms:**  Reproduce existing SOTA results with ease, including:
*   **🚀 NEW: Reasoning-based Methods:** Includes **7 reasoning-based methods** that combine reasoning ability with retrieval, significantly improving performance on complex tasks.
*   **Efficient Preprocessing:** Simplifies the RAG workflow with scripts for corpus processing, index building, and document retrieval.
*   **Optimized Execution:** Leverages vLLM, FastChat for LLM acceleration, and Faiss for vector index management.
*   **Easy-to-Use UI:**  Configure, experiment, and evaluate RAG baselines through an intuitive visual interface: [FlashRAG-UI](#flashrag-ui).

## Installation

Get started by installing FlashRAG using pip:

```bash
pip install flashrag-dev --pre
```

For detailed installation instructions and optional dependencies (vLLM, sentence-transformers, pyserini, faiss), please refer to the [installation section of the original README](https://github.com/RUC-NLPIR/FlashRAG#wrench-installation).

## Quick Start

### Index Construction

*   For **Dense Retrieval (e.g., using embedding models):**

```bash
python -m flashrag.retriever.index_builder \
  --retrieval_method e5 \
  --model_path /model/e5-base-v2/ \
  --corpus_path indexes/sample_corpus.jsonl \
  --save_dir indexes/ \
  --use_fp16 \
  --max_length 512 \
  --batch_size 256 \
  --pooling_method mean \
  --faiss_type Flat
```

*   For **Sparse Retrieval (BM25):**

```bash
python -m flashrag.retriever.index_builder \
  --retrieval_method bm25 \
  --corpus_path indexes/sample_corpus.jsonl \
  --bm25_backend bm25s \
  --save_dir indexes/
```

For more detailed guidance and alternative index building options (e.g., SPLADE), please see the [Quick Start section of the original README](https://github.com/RUC-NLPIR/FlashRAG#rocket-quick-start).

## Components

FlashRAG provides a modular architecture with these main components:

#### RAG Components

*   **Judger**: SKR Judger, for judging whether to retrieve using the SKR method.
*   **Retriever**: Dense and Sparse (BM25) Retrievers, and Bi/Cross-Encoder Rerankers.
*   **Refiner**: Extractive and Abstractive Refiners, LLMLingua, SelectiveContext, and KG Refiner
*   **Generator**: Encoder-Decoder and Decoder-only Generators, FastChat, and vLLM.

#### Pipelines

Based on these components, FlashRAG offers various pipeline types, categorized by inference path:

*   Sequential
*   Conditional
*   Branching
*   Loop (including Iterative, Self-Ask, Self-RAG, FLARE, IRCoT, and Reasoning Pipelines)

## FlashRAG-UI

**FlashRAG-UI** offers a user-friendly interface to:

*   Load configurations and experiment with various RAG methods.
*   Quickly load corpora and index files.
*   Reproduce and evaluate benchmark results.

<table align="center">
  <tr>
    <td align="center">
      <img src="./asset/demo_en1.jpg" alt="Image 1" width="505"/>
    </td>
    <td align="center">
      <img src="./asset/demo_en2.jpg" alt="Image 2" width="505"/>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="./asset/demo_en4.png" alt="Image 3" width="500"/>
    </td>
    <td align="center">
      <img src="./asset/demo_en3.jpg" alt="Image 4" width="500"/>
    </td>
  </tr>
</table>

To launch the UI, navigate to the `webui` directory and run:

```bash
cd webui
python interface.py
```

## Supporting Methods

FlashRAG supports a range of RAG methods, including the following, with consistently set generator, retriever, and prompt configurations:

| Method                                                                                    | Type        | NQ (EM) | TriviaQA (EM) | Hotpotqa (F1) | 2Wiki (F1) | PopQA (F1) | WebQA(EM) |
| ----------------------------------------------------------------------------------------- | ----------- | ------- | ------------- | ------------- | ---------- | ---------- | --------- |
| Naive Generation                                                                          | Sequential  | 22.6    | 55.7          | 28.4          | 33.9       | 21.7       | 18.8      |
| Standard RAG                                                                              | Sequential  | 35.1    | 58.9          | 35.3          | 21.0       | 36.7       | 15.7      |
| [AAR-contriever-kilt](https://aclanthology.org/2023.acl-long.136.pdf)                     | Sequential  | 30.1    | 56.8          | 33.4          | 19.8       | 36.1       | 16.1      |
| [LongLLMLingua](https://arxiv.org/abs/2310.06839)                                         | Sequential  | 32.2    | 59.2          | 37.5          | 25.0       | 38.7       | 17.5      |
| [RECOMP-abstractive](https://arxiv.org/pdf/2310.04408)                                    | Sequential  | 33.1    | 56.4          | 37.5          | 32.4       | 39.9       | 20.2      |
| [Selective-Context](https://arxiv.org/abs/2310.06201)                                     | Sequential  | 30.5    | 55.6          | 34.4          | 18.5       | 33.5       | 17.3      |
| [Trace](https://arxiv.org/abs/2406.11460)                                                 | Sequential  | 30.7    | 50.2          | 34.0          | 15.5       | 37.4       | 19.9      |
| [Spring](https://arxiv.org/abs/2405.19670)                                                | Sequential  | 37.9    | 64.6          | 42.6          | 37.3       | 54.8       | 27.7      |
| [SuRe](https://arxiv.org/abs/2404.13081)                                                  | Branching   | 37.1    | 53.2          | 33.4          | 20.6       | 48.1       | 24.2      |
| [REPLUG](https://arxiv.org/abs/2301.12652)                                                | Branching   | 28.9    | 57.7          | 31.2          | 21.1       | 27.8       | 20.2      |
| [SKR](https://aclanthology.org/2023.findings-emnlp.691.pdf)                               | Conditional | 33.2    | 56.0          | 32.4          | 23.4       | 31.7       | 17.0      |
| [Adaptive-RAG](https://aclanthology.org/2024.naacl-long.389.pdf)                          | Conditional | 35.1    | 56.6          | 39.1          | 28.4       | 40.4       | 16.0      |
| [Ret-Robust](https://arxiv.org/abs/2310.01558)                                            | Loop        | 42.9    | 68.2          | 35.8          | 43.4       | 57.2       | 33.7      |
| [Self-RAG](https://arxiv.org/abs/2310.11511)                                              | Loop        | 36.4    | 38.2          | 29.6          | 25.1       | 32.7       | 21.9      |
| [FLARE](https://arxiv.org/abs/2305.06983)                                                 | Loop        | 22.5    | 55.8          | 28.0          | 33.9       | 20.7       | 20.2      |
| [Iter-Retgen](https://arxiv.org/abs/2305.15294), [ITRG](https://arxiv.org/abs/2310.05149) | Loop        | 36.8    | 60.1          | 38.3          | 21.6       | 37.9       | 18.2      |
| [IRCoT](https://aclanthology.org/2023.acl-long.557.pdf)                                   | Loop        | 33.3    | 56.9          | 41.5          | 32.4       | 45.6       | 20.7      |
| [RQRAG](https://arxiv.org/abs/2404.00610) | Loop | 32.6 | 52.5 | 33.5 | 35.8 | 46.4 | 26.2 |

#### 🚀 Reasoning-based Methods (NEW!)

FlashRAG introduces support for these new reasoning-based methods:

| Method                                                                                    | Type        | NQ (EM) | TriviaQA (EM) | PopQA (EM) | Hotpotqa (F1) | 2Wiki (F1) |  Musique (F1) | Bamboogle (F1) |
| ----------------------------------------------------------------------------------------- | ----------- | ------- | ------------- | ------------- | ---------- | ---------- | --------- | ----------------------------------------------- |
| [Search-R1](https://arxiv.org/abs/2503.09516) | Reasoning | 45.2 | 62.2 | 49.2 | 54.5 | 42.6 | 29.2 |  59.9 |
| [R1-Searcher](https://arxiv.org/pdf/2503.05592) | Reasoning | 36.9 | 61.6 | 42.0 | 49.0 | 49.1 | 24.7 | 57.7 |
| [O2-Searcher](https://arxiv.org/pdf/2505.16582) | Reasoning | 41.4 | 51.4 | 46.8 | 43.4 | 48.6 | 19.0 | 47.6 |
| [AutoRefine](https://www.arxiv.org/pdf/2505.11277) | Reasoning | 43.8 | 59.8 | 32.4 | 54.0 | 50.3 | 23.6 | 46.6 |
| [ReaRAG](https://arxiv.org/abs/2503.21729) | Reasoning | 26.3 | 51.8 | 24.6 | 42.9 | 41.6 | 21.2 | 41.9 |
| [CoRAG](https://arxiv.org/abs/2503.21729) | Reasoning | 40.9 | 63.1 | 36.0 | 56.6 | 60.7 | 31.9 | 54.1 |
| [SimpleDeepSearcher](https://arxiv.org/pdf/2505.16834) | Reasoning | 36.1 | 61.6 | 42.0 | 49.0 | 49.1 | 24.7 | 57.7 |

## Supporting Datasets

FlashRAG supports 36 preprocessed datasets. These are available on [Hugging Face Datasets](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets).

## FAQs

Find answers to common questions:

*   How do I set experimental parameters?
*   How do I build a custom corpus (e.g., segmented Wikipedia)?
*   How do I index my own corpus?
*   How do I reproduce the supporting methods?

Find these answers and more in the [FAQs section of the original README](https://github.com/RUC-NLPIR/FlashRAG#raised_hands-additional-faqs).

## License

FlashRAG is licensed under the [MIT License](./LICENSE).

## Citation

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