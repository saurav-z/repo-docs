# ⚡ FlashRAG: Supercharge Your RAG Research ⚡

**Effortlessly reproduce and innovate in Retrieval-Augmented Generation (RAG) with FlashRAG, a powerful and flexible Python toolkit.**  [View the original repository](https://github.com/RUC-NLPIR/FlashRAG)

[![arXiv](https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv)](https://arxiv.org/abs/2405.13576)
[![HuggingFace Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Datasets-27b3b4.svg)](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/)
[![ModelScope Datasets](https://custom-icon-badges.demolab.com/badge/ModelScope%20Datasets-624aff?style=flat&logo=modelscope&logoColor=white)](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset)
[![DeepWiki Document](https://devin.ai/assets/deepwiki-badge.png)](https://deepwiki.com/RUC-NLPIR/FlashRAG)
[![License](https://img.shields.io/badge/LICENSE-MIT-green)](https://github.com/RUC-NLPIR/FlashRAG/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/made_with-Python-blue)](https://www.python.org/)
[![Trendshift](https://trendshift.io/api/badge/repositories/10454)](https://trendshift.io/repositories/10454)

**Key Features:**

*   **Modular & Customizable Framework:** Build complex RAG pipelines with flexible components (retrievers, rerankers, generators, compressors).
*   **Comprehensive Benchmark Datasets:** Access 36 pre-processed RAG datasets for robust model evaluation.
*   **State-of-the-Art RAG Algorithms:** Reproduce and experiment with **23 advanced RAG algorithms**, including **7 reasoning-based methods** for superior performance.
*   **Reasoning-Based Methods:** **NEW!** Explore methods that combine reasoning and retrieval for advanced question-answering capabilities.
*   **Efficient Preprocessing:** Streamline your workflow with tools for corpus processing, index building, and document retrieval.
*   **Optimized Execution:** Benefit from vLLM, FastChat, and Faiss for faster LLM inference and vector index management.
*   **User-Friendly UI:** Easily configure, experiment with, and evaluate RAG methods using FlashRAG-UI.

**Get started today and accelerate your RAG research!**

## Table of Contents

*   [Installation](#wrench-installation)
*   [Quick Start](#rocket-quick-start)
*   [Features](#sparkles-features)
*   [Components](#gear-components)
*   [FlashRAG-UI](#art-flashrag-ui)
*   [Supporting Methods](#robot-supporting-methods)
*   [Supporting Datasets & Document Corpus](#notebook-supporting-datasets--document-corpus)
*   [Additional FAQs](#raised_hands-additional-faqs)
*   [License](#bookmark-license)
*   [Citation](#star2-citation)
*   [Star History](#star-history)

## :wrench: Installation

Install FlashRAG quickly using pip:

```bash
pip install flashrag-dev --pre
```

Or install from source:

```bash
git clone https://github.com/RUC-NLPIR/FlashRAG.git
cd FlashRAG
pip install -e .
```

For optional dependencies (vLLM, sentence-transformers, pyserini, and FAISS), refer to the full installation instructions in the original repository.

## :rocket: Quick Start

### Corpus Construction

Prepare your corpus as a `jsonl` file:

```jsonl
{"id": "0", "contents": "..."}
{"id": "1", "contents": "..."}
```

### Index Construction

Choose your retrieval method:

#### For Dense Retrieval Methods

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

#### For Sparse Retrieval Methods (BM25)

##### Building Index with BM25s

```bash
python -m flashrag.retriever.index_builder \
  --retrieval_method bm25 \
  --corpus_path indexes/sample_corpus.jsonl \
  --bm25_backend bm25s \
  --save_dir indexes/
```

##### Building Index with Pyserini

```bash
python -m flashrag.retriever.index_builder \
  --retrieval_method bm25 \
  --corpus_path indexes/sample_corpus.jsonl \
  --bm25_backend pyserini \
  --save_dir indexes/
```

### For Sparse Neural Retrieval Methods (SPLADE)

##### Install Seismic Index:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh # Install Rust for compiling
pip install pyseismic-lsr # Install Seismic
```

##### Then build the index with Seismic:
```bash
python -m flashrag.retriever.index_builder \ # builder
        --retrieval_method splade \ # Model name to trigger seismic index (splade only available)
        --model_path retriever/splade-v3 \ # Local path or repository path are both supported.
        --corpus_embedded_path data/ms_marco/ms_marco_embedded_corpus.jsonl \  # Use cached embedded corpus if corpus is already available in seismic expected format
        --corpus_path data/ms_marco/ms_marco_corpus.jsonl \ # Corpus path in format {id, contents} jsonl file to be embedded if not already built
        --save_dir indexes/ \ # save index directory
        --use_fp16 \ # tell to use fp16 for splade model
        --max_length 512 \ # max tokens for each document
        --batch_size 4 \ # batch size for splade model (4-5 seems the best size for Tesla T4 16GB)
        --n_postings 1000 \ # seismic number of posting lists
        --centroid_fraction 0.2 \ # seismic centroids
        --min_cluster_size 2 \ # seismic min cluster
        --summary_energy 0.4 \ # seismic energy
        --batched_indexing 10000000 # seismic batch
        --nknn 32 # Optional parameter. Tell to seismic to use also knn graph. if not present seismic will work without knn graph
```

### Using the Ready-Made Pipeline

Configure and run your pipeline using the `Config` and `SequentialPipeline` classes. See the full details in the original documentation.

### Build Your Own Pipeline

Extend the `BasicPipeline` class to create custom RAG workflows.

### Just Use Components

Leverage individual components for flexibility in your own RAG implementations.

## :sparkles: Features

*   **Extensive and Customizable Framework:** Comprehensive components for RAG scenarios, allowing flexible pipeline assembly.
*   **Comprehensive Benchmark Datasets:** 36 pre-processed RAG benchmark datasets for robust model evaluation.
*   **Pre-implemented Advanced RAG Algorithms:** **23 advanced RAG algorithms** with reported results.
*   **🚀 Reasoning-based Methods:** **NEW!** Support for **7 reasoning-based methods** for superior multi-hop task performance.
*   **Efficient Preprocessing Stage:** Simplify RAG workflow preparation with various scripts.
*   **Optimized Execution:** Utilize tools like vLLM, FastChat for efficiency.
*   **Easy to Use UI:** Developed a very easy to use UI to easily and quickly configure and experience the RAG baselines we have implemented, as well as run evaluation scripts on a visual interface.

## :gear: Components

FlashRAG offers a rich set of RAG components for building custom pipelines:

#### RAG-Components

| Type        | Module              | Description                                                                    |
| ----------- | ------------------- | ------------------------------------------------------------------------------ |
| Judger      | SKR Judger          | Judging whether to retrieve using SKR method |
|             |                     |                                                                                |
| Retriever   | Dense Retriever     | Bi-encoder models (dpr, bge, e5) using faiss for search                         |
|             | BM25 Retriever      | Sparse retrieval method based on Lucene                                        |
|             | Bi-Encoder Reranker | Calculate matching score using bi-Encoder                                      |
|             | Cross-Encoder Reranker | Calculate matching score using cross-encoder                                  |
|             |                     |                                                                                |
| Refiner     | Extractive Refiner  | Refine input by extracting important context                                    |
|             | Abstractive Refiner | Refine input through seq2seq model                                              |
|             | LLMLingua Refiner   | LLMLingua-series prompt compressor                                              |
|             | SelectiveContext Refiner   | Selective-Context prompt compressor |
|             | KG Refiner | Use Trace method to construct a knowledge graph                                      |
|             |                     |                                                                                |
| Generator   | Encoder-Decoder Generator | Encoder-Decoder model, supporting Fusion-in-Decoder (FiD)                     |
|             | Decoder-only Generator   | Native transformers implementation                                          |
|             | FastChat Generator   | Accelerate with FastChat                                                       |
|             | vllm Generator     | Accelerate with vllm                                                      |

#### Pipelines

Categorized by inference path, FlashRAG provides pre-built pipelines:

| Type        | Module              | Description                                                                      |
| ----------- | ------------------- | -------------------------------------------------------------------------------- |
| Sequential  | Sequential Pipeline | Linear execution of query, supporting refiner, reranker                         |
| Conditional | Conditional Pipeline | Distinct execution paths based on query types                                     |
| Branching   | REPLUG Pipeline     | Generate answer by integrating probabilities in multiple generation paths |
|   | SuRe Pipeline     | Ranking and merging generated results based on each document       |
| Loop        | Iterative Pipeline  | Alternating retrieval and generation                                             |
|             | Self-Ask Pipeline   | Decompose complex problems into subproblems using self-ask                        |
|             | Self-RAG Pipeline   | Adaptive retrieval, critique, and generation                                     |
|             | FLARE Pipeline      | Dynamic retrieval during the generation process                                  |
|             | IRCoT Pipeline      | Integrate retrieval process with CoT |
|             | Reasoning Pipeline      | Reasoning with retrieval                                               |

## :art: FlashRAG-UI

FlashRAG-UI provides a user-friendly interface for configuring, experimenting with, and evaluating RAG methods.  See the demo images in the original documentation.

### Features

*   One-Click Configuration Loading
*   Quick Method Experience
*   Efficient Benchmark Reproduction

```bash
cd webui
python interface.py
```

## :robot: Supporting Methods

FlashRAG implements **23 RAG methods**, using a consistent setting.

| Method                                                                                    | Type        | NQ (EM) | TriviaQA (EM) | Hotpotqa (F1) | 2Wiki (F1) | PopQA (F1) | WebQA(EM) | Specific setting                                |
| ----------------------------------------------------------------------------------------- | ----------- | ------- | ------------- | ------------- | ---------- | ---------- | --------- | ----------------------------------------------- |
| Naive Generation                                                                          | Sequential  | 22.6    | 55.7          | 28.4          | 33.9       | 21.7       | 18.8      |                                                 |
| Standard RAG                                                                              | Sequential  | 35.1    | 58.9          | 35.3          | 21.0       | 36.7       | 15.7      |                                                 |
| [AAR-contriever-kilt](https://aclanthology.org/2023.acl-long.136.pdf)                     | Sequential  | 30.1    | 56.8          | 33.4          | 19.8       | 36.1       | 16.1      |                                                 |
| [LongLLMLingua](https://arxiv.org/abs/2310.06839)                                         | Sequential  | 32.2    | 59.2          | 37.5          | 25.0       | 38.7       | 17.5      | Compress Ratio=0.5                              |
| [RECOMP-abstractive](https://arxiv.org/pdf/2310.04408)                                    | Sequential  | 33.1    | 56.4          | 37.5          | 32.4       | 39.9       | 20.2      |                                                 |
| [Selective-Context](https://arxiv.org/abs/2310.06201)                                     | Sequential  | 30.5    | 55.6          | 34.4          | 18.5       | 33.5       | 17.3      | Compress Ratio=0.5                              |
| [Trace](https://arxiv.org/abs/2406.11460)                                                 | Sequential  | 30.7    | 50.2          | 34.0          | 15.5       | 37.4       | 19.9      |                                                 |
| [Spring](https://arxiv.org/abs/2405.19670)                                                | Sequential  | 37.9    | 64.6          | 42.6          | 37.3       | 54.8       | 27.7      | Use Llama2-7B-chat with trained embedding table |
| [SuRe](https://arxiv.org/abs/2404.13081)                                                  | Branching   | 37.1    | 53.2          | 33.4          | 20.6       | 48.1       | 24.2      | Use provided prompt                             |
| [REPLUG](https://arxiv.org/abs/2301.12652)                                                | Branching   | 28.9    | 57.7          | 31.2          | 21.1       | 27.8       | 20.2      |                                                 |
| [SKR](https://aclanthology.org/2023.findings-emnlp.691.pdf)                               | Conditional | 33.2    | 56.0          | 32.4          | 23.4       | 31.7       | 17.0      | Use infernece-time training data                |
| [Adaptive-RAG](https://aclanthology.org/2024.naacl-long.389.pdf)                          | Conditional | 35.1    | 56.6          | 39.1          | 28.4       | 40.4       | 16.0      |                                                 |
| [Ret-Robust](https://arxiv.org/abs/2310.01558)                                            | Loop        | 42.9    | 68.2          | 35.8          | 43.4       | 57.2       | 33.7      | Use LLAMA2-13B with trained lora                |
| [Self-RAG](https://arxiv.org/abs/2310.11511)                                              | Loop        | 36.4    | 38.2          | 29.6          | 25.1       | 32.7       | 21.9      | Use trained selfrag-llama2-7B                   |
| [FLARE](https://arxiv.org/abs/2305.06983)                                                 | Loop        | 22.5    | 55.8          | 28.0          | 33.9       | 20.7       | 20.2      |                                                 |
| [Iter-Retgen](https://arxiv.org/abs/2305.15294), [ITRG](https://arxiv.org/abs/2310.05149) | Loop        | 36.8    | 60.1          | 38.3          | 21.6       | 37.9       | 18.2      |                                                 |
| [IRCoT](https://aclanthology.org/2023.acl-long.557.pdf)                                   | Loop        | 33.3    | 56.9          | 41.5          | 32.4       | 45.6       | 20.7      |                                                 |
| [RQRAG](https://arxiv.org/abs/2404.00610)                                   | Loop        | 32.6    | 52.5          | 33.5          | 35.8       | 46.4       | 26.2      |  Use trained rqrag-llama2-7B                                               |

#### 🚀 Reasoning-based Methods (NEW!)

| Method                                                                                    | Type        | NQ (EM) | TriviaQA (EM) | PopQA (EM) | Hotpotqa (F1) | 2Wiki (F1) |  Musique (F1) | Bamboogle (F1) | Specific setting                             |
| ----------------------------------------------------------------------------------------- | ----------- | ------- | ------- | ------------- | ------------- | ---------- | ---------- | --------- | ----------------------------------------------- |
| [Search-R1](https://arxiv.org/abs/2503.09516) | Reasoning | 45.2 | 62.2 | 49.2 | 54.5 | 42.6 | 29.2 |  59.9 | SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo |
| [R1-Searcher](https://arxiv.org/pdf/2503.05592) | Reasoning | 36.9 | 61.6 | 42.0 | 49.0 | 49.1 | 24.7 | 57.7 | Qwen-2.5-7B-base-RAG-RL |
| [O2-Searcher](https://arxiv.org/pdf/2505.16582) | Reasoning | 41.4 | 51.4 | 46.8 | 43.4 | 48.6 | 19.0 | 47.6 | O2-Searcher-Qwen2.5-3B-GRPO |
| [AutoRefine](https://www.arxiv.org/pdf/2505.11277) | Reasoning | 43.8 | 59.8 | 32.4 | 54.0 | 50.3 | 23.6 | 46.6 | AutoRefine-Qwen2.5-3B-Base |
| [ReaRAG](https://arxiv.org/abs/2503.21729) | Reasoning | 26.3 | 51.8 | 24.6 | 42.9 | 41.6 | 21.2 | 41.9 | ReaRAG-9B |
| [CoRAG](https://arxiv.org/abs/2503.21729) | Reasoning | 40.9 | 63.1 | 36.0 | 56.6 | 60.7 | 31.9 | 54.1 | CoRAG-Llama3.1-8B-MultihopQA |
| [SimpleDeepSearcher](https://arxiv.org/pdf/2505.16834) | Reasoning | 36.1 | 61.6 | 42.0 | 49.0 | 49.1 | 24.7 | 57.7 | Qwen-7B-SimpleDeepSearcher |

## :notebook: Supporting Datasets & Document Corpus

### Datasets

Access 36 pre-processed datasets for RAG research on [Huggingface datasets](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets).  Datasets are in `jsonl` format.

### Document Corpus

Supports `jsonl` format.  For Wikipedia, use the provided [script](./docs/original_docs/process-wiki.md) to process it into a clean corpus.

### Index

Preprocessed index available in the ModelScope dataset: [FlashRAG_Dataset/retrieval_corpus/wiki18_100w_e5_index.zip](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/file/view/master?id=47985&status=2&fileName=retrieval_corpus%252Fwiki18_100w_e5_index.zip).

## :raised_hands: Additional FAQs

*   How to set experimental parameters?
*   How to build your own corpus?
*   How to index your own corpus?
*   How to reproduce supporting methods?

## :bookmark: License

FlashRAG is licensed under the [MIT License](./LICENSE).

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

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=RUC-NLPIR/FlashRAG&type=Date)](https://star-history.com/#RUC-NLPIR/FlashRAG&Date)