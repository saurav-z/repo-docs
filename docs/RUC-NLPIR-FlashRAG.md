# FlashRAG: Your Toolkit for Cutting-Edge RAG Research ⚡

**Effortlessly reproduce and explore state-of-the-art Retrieval Augmented Generation (RAG) techniques with FlashRAG, a comprehensive Python toolkit.**  Dive into advanced RAG methods, benchmark datasets, and an intuitive UI, all in one place.  [Get started and explore the original repo!](https://github.com/RUC-NLPIR/FlashRAG)

[![arXiv](https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv)](https://arxiv.org/abs/2405.13576)
[![HuggingFace Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Datasets-27b3b4.svg)](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/)
[![ModelScope Datasets](https://custom-icon-badges.demolab.com/badge/ModelScope%20Datasets-624aff?style=flat&logo=modelscope&logoColor=white)](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset)
[![DeepWiki](https://devin.ai/assets/deepwiki-badge.png)](https://deepwiki.com/RUC-NLPIR/FlashRAG)
[![License](https://img.shields.io/badge/LICENSE-MIT-green)](https://github.com/RUC-NLPIR/FlashRAG/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/made_with-Python-blue)](https://www.python.org/)
[![Trendshift](https://trendshift.io/api/badge/repositories/10454)](https://trendshift.io/repositories/10454)

## Key Features

*   **Modular and Extensible:**  Easily assemble complex RAG pipelines using pre-built components like retrievers, rerankers, generators, and compressors.
*   **Extensive Dataset Support:** Includes 36 pre-processed RAG benchmark datasets for rigorous testing and evaluation.
*   **State-of-the-Art Algorithms:** Explore and reproduce **23 advanced RAG algorithms**, including **7 reasoning-based methods** for superior performance.
*   **Reasoning-Enhanced Retrieval:**  Benefit from cutting-edge methods that integrate reasoning capabilities with retrieval for complex tasks.
*   **Efficient Workflow:** Streamlined preprocessing with tools for corpus processing, index building, and document retrieval.
*   **Optimized Performance:** Leverages vLLM, FastChat, and Faiss for accelerated LLM inference and vector index management.
*   **User-Friendly UI:** Easily configure, experiment with, and evaluate RAG baselines with FlashRAG-UI.

## Quick Links

*   [Installation](#wrench-installation)
*   [Features](#sparkles-features)
*   [Quick Start](#rocket-quick-start)
*   [Components](#gear-components)
*   [FlashRAG-UI](#art-flashrag-ui)
*   [Supporting Methods](#robot-supporting-methods)
*   [Supporting Datasets & Document Corpus](#notebook-supporting-datasets--document-corpus)
*   [FAQs](#raised_hands-additional-faqs)

## Installation

```bash
pip install flashrag-dev --pre
```

For detailed installation instructions, including optional dependencies (vLLM, sentence-transformers, and pyserini), see the [original README](https://github.com/RUC-NLPIR/FlashRAG).

## Quick Start

### Corpus and Index Construction

FlashRAG supports building indexes for both dense and sparse retrieval methods.  Here's how to get started:

1.  **Corpus Preparation:**  Prepare your document corpus in a `jsonl` format:

    ```jsonl
    {"id": "0", "contents": "..."}
    {"id": "1", "contents": "..."}
    ```

2.  **Index Building:**

    *   **Dense Retrieval (e.g., using Faiss):**

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

    *   **Sparse Retrieval (e.g., BM25 using bm25s):**

        ```bash
        python -m flashrag.retriever.index_builder \
          --retrieval_method bm25 \
          --corpus_path indexes/sample_corpus.jsonl \
          --bm25_backend bm25s \
          --save_dir indexes/
        ```

    *   **Sparse Neural Retrieval (e.g., SPLADE):**

        ```bash
        python -m flashrag.retriever.index_builder \
          --retrieval_method splade \
          --model_path retriever/splade-v3 \
          --corpus_embedded_path data/ms_marco/ms_marco_embedded_corpus.jsonl \
          --corpus_path data/ms_marco/ms_marco_corpus.jsonl \
          --save_dir indexes/ \
          --use_fp16 \
          --max_length 512 \
          --batch_size 4 \
          --n_postings 1000 \
          --centroid_fraction 0.2 \
          --min_cluster_size 2 \
          --summary_energy 0.4 \
          --batched_indexing 10000000 \
          --nknn 32
        ```

3.  **Using the ready-made pipeline**: Build your own pipeline with the help of `SequentialPipeline`, see [<u>pipelines</u>](#pipelines), or build your own component to implement complex RAG process. See [<u>basic introduction of the components</u>](./docs/original_docs/basic_usage.md).

For more details on corpus processing, index building, and pipeline usage, refer to the [original README](https://github.com/RUC-NLPIR/FlashRAG).

## Components

FlashRAG provides a comprehensive set of modular components to build your RAG pipelines:

### RAG Components
| Type   | Module           | Description                                               |
| :----- | :--------------- | :-------------------------------------------------------- |
| Judger | SKR Judger       | Determines retrieval using [SKR](https://aclanthology.org/2023.findings-emnlp.691.pdf) |
| Retriever | Dense Retriever  | Uses Bi-encoder models such as dpr, bge, e5 (Faiss)     |
| Retriever | BM25 Retriever   | Sparse retrieval method based on Lucene                 |
| Retriever | Bi-Encoder Reranker | Calculate matching score using bi-Encoder |
| Retriever | Cross-Encoder Reranker | Calculate matching score using cross-encoder |
| Refiner  | Extractive Refiner | Extracts key information                                  |
| Refiner  | Abstractive Refiner| Refines through seq2seq model |
| Refiner  | LLMLingua Refiner| Prompt compressor from [LLMLingua](https://aclanthology.org/2023.emnlp-main.825/)           |
| Refiner  | SelectiveContext Refiner | Prompt compressor from [Selective-Context](https://arxiv.org/abs/2310.06201)        |
| Refiner  | KG Refiner        | Use Trace method for knowledge graph |
| Generator| Encoder-Decoder Generator | Supports [FiD](https://arxiv.org/abs/2007.01282)            |
| Generator| Decoder-only Generator | Native transformers implementation                  |
| Generator| FastChat Generator | Accelerate with [FastChat](https://github.com/lm-sys/FastChat)      |
| Generator| vllm Generator     | Accelerate with [vllm](https://github.com/vllm-project/vllm)        |

### Pipelines

| Type          | Module               | Description                                                                 |
| :------------ | :------------------- | :-------------------------------------------------------------------------- |
| Sequential    | Sequential Pipeline  | Linear RAG process, supports refiner/reranker                             |
| Conditional   | Conditional Pipeline | Different paths based on query type (with judger)                            |
| Branching     | REPLUG Pipeline      | Integrates probabilities for generation                                     |
| Branching     | SuRe Pipeline     | Ranking and merging generated results based on each document                                     |
| Loop          | Iterative Pipeline   | Alternating retrieval and generation                                        |
| Loop          | Self-Ask Pipeline    | Decomposes complex problems using [Self-Ask](https://arxiv.org/abs/2210.03350) |
| Loop          | Self-RAG Pipeline    | Adaptive retrieval, critique, and generation                                |
| Loop          | FLARE Pipeline       | Dynamic retrieval during generation                                         |
| Loop          | IRCoT Pipeline       | Integrates retrieval with CoT                                                |
| Loop          | Reasoning Pipeline   | Reasoning with retrieval    |

## FlashRAG-UI

**FlashRAG-UI provides a user-friendly and visually appealing interface for configuring, experimenting with, and evaluating RAG methods, making complex research more efficient.**

### Features:
*   **One-Click Configuration:** Load parameters with clicks.
*   **Quick Method Experience:** Load corpora and indices to explore different methods.
*   **Efficient Benchmark Reproduction:** Reproduce baselines and experiment easily.

To launch the UI:

```bash
cd webui
python interface.py
```

For visual demos and screenshots, see the [original README](https://github.com/RUC-NLPIR/FlashRAG).

## Supporting Methods

FlashRAG supports **23 RAG methods** with a consistent setup. The results are presented below.

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

### Reasoning-Based Methods (NEW!)

| Method                                                                                    | Type        | NQ (EM) | TriviaQA (EM) | PopQA (EM) | Hotpotqa (F1) | 2Wiki (F1) |  Musique (F1) | Bamboogle (F1) | Specific setting                             |
| ----------------------------------------------------------------------------------------- | ----------- | ------- | ------- | ------------- | ------------- | ---------- | ---------- | --------- | ----------------------------------------------- |
| [Search-R1](https://arxiv.org/abs/2503.09516) | Reasoning | 45.2 | 62.2 | 49.2 | 54.5 | 42.6 | 29.2 |  59.9 | SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo |
| [R1-Searcher](https://arxiv.org/pdf/2503.05592) | Reasoning | 36.9 | 61.6 | 42.0 | 49.0 | 49.1 | 24.7 | 57.7 | Qwen-2.5-7B-base-RAG-RL |
| [O2-Searcher](https://arxiv.org/pdf/2505.16582) | Reasoning | 41.4 | 51.4 | 46.8 | 43.4 | 48.6 | 19.0 | 47.6 | O2-Searcher-Qwen2.5-3B-GRPO |
| [AutoRefine](https://www.arxiv.org/pdf/2505.11277) | Reasoning | 43.8 | 59.8 | 32.4 | 54.0 | 50.3 | 23.6 | 46.6 | AutoRefine-Qwen2.5-3B-Base |
| [ReaRAG](https://arxiv.org/abs/2503.21729) | Reasoning | 26.3 | 51.8 | 24.6 | 42.9 | 41.6 | 21.2 | 41.9 | ReaRAG-9B |
| [CoRAG](https://arxiv.org/abs/2503.21729) | Reasoning | 40.9 | 63.1 | 36.0 | 56.6 | 60.7 | 31.9 | 54.1 | CoRAG-Llama3.1-8B-MultihopQA |
| [SimpleDeepSearcher](https://arxiv.org/pdf/2505.16834) | Reasoning | 36.1 | 61.6 | 42.0 | 49.0 | 49.1 | 24.7 | 57.7 | Qwen-7B-SimpleDeepSearcher |

## Supporting Datasets & Document Corpus

### Datasets

FlashRAG provides 36 pre-processed datasets. All datasets are available at [<u>Huggingface datasets</u>](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets). For each dataset, the format is:

```python
{
  'id': str,
  'question': str,
  'golden_answers': List[str],
  'metadata': dict
}
```

See the [original README](https://github.com/RUC-NLPIR/FlashRAG) for a full list of datasets and sample sizes.

### Document Corpus

FlashRAG supports the `jsonl` format for document collections:

```jsonl
{"id":"0", "contents": "..."}
{"id":"1", "contents": "..."}
```

The `contents` key is essential. For Wikipedia, use the [<u>comprehensive script</u>](./docs/original_docs/process-wiki.md) to process any Wikipedia dump. The index also provides the e5-base-v2 retriever wiki18_100w dataset which can be found on  [<u>ModelScope dataset page</u>](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/file/view/master?id=47985&status=2&fileName=retrieval_corpus%252Fwiki18_100w_e5_index.zip). MS MARCO is also supported.

##  Citation

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