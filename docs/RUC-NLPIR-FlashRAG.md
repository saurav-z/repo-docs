# ⚡ FlashRAG: Your Toolkit for Efficient RAG Research ⚡

> **Unleash the power of Retrieval-Augmented Generation (RAG) with FlashRAG, a comprehensive Python toolkit designed to streamline your research and development in the RAG domain.**

[![arXiv](https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv)](https://arxiv.org/abs/2405.13576)
[![HuggingFace Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Datasets-27b3b4.svg)](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/)
[![ModelScope Datasets](https://custom-icon-badges.demolab.com/badge/ModelScope%20Datasets-624aff?style=flat&logo=modelscope&logoColor=white)](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset)
[![DeepWiki](https://devin.ai/assets/deepwiki-badge.png)](https://deepwiki.com/RUC-NLPIR/FlashRAG)
[![License](https://img.shields.io/badge/LICENSE-MIT-green)](https://github.com/RUC-NLPIR/FlashRAG/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/made_with-Python-blue)](https://www.python.org/)

**[Explore the FlashRAG Repository](https://github.com/RUC-NLPIR/FlashRAG)**

<p align="center">
<img src="asset/framework.jpg" alt="FlashRAG Framework">
</p>

## Key Features

*   **Modular & Customizable:** Build complex RAG pipelines with essential components like retrievers, rerankers, and generators.
*   **Extensive Benchmark Datasets:** Evaluate your models on 36 pre-processed RAG benchmark datasets.
*   **State-of-the-Art Algorithms:** Reproduce and experiment with **23 advanced RAG algorithms**, including **7 reasoning-based methods**.
*   **Reasoning Power:** Harness the potential of **7 reasoning-based methods**, excelling in complex multi-hop tasks.
*   **Efficient Workflow:** Streamline your process with preprocessing scripts for corpus management, index building, and document retrieval.
*   **Optimized Performance:** Leverage tools like vLLM and FastChat for fast LLM inference and Faiss for vector index management.
*   **User-Friendly UI:** Easily configure and experiment with supported RAG methods using our intuitive UI.

## Key Improvements and Contributions in FlashRAG

*   **Reasoning-based Methods Support (NEW!):** Added support for **7 reasoning-based methods** that combine reasoning ability with retrieval.
*   **Multimodal RAG:**  Supported multimodal RAG, including Llava, Qwen, and InternVL.
*   **Expanded Algorithms:** Now supports **23 state-of-the-art RAG algorithms**.

## Sections

*   [Installation](#wrench-installation)
*   [Quick Start](#rocket-quick-start)
*   [Components](#gear-components)
*   [FlashRAG-UI](#art-flashrag-ui)
*   [Supporting Methods](#robot-supporting-methods)
*   [Supporting Datasets & Document Corpus](#notebook-supporting-datasets--document-corpus)
*   [FAQs](#raised_hands-additional-faqs)
*   [License](#bookmark-license)
*   [Citation](#star2-citation)

## :wrench: Installation

Install FlashRAG easily using pip:

```bash
pip install flashrag-dev --pre
```

For detailed installation instructions and optional dependencies (vLLM, sentence-transformers, Pyserini, Faiss), refer to the [original README](https://github.com/RUC-NLPIR/FlashRAG).

## :rocket: Quick Start

1.  **Corpus Construction:** Prepare your data in a `jsonl` format:

    ```jsonl
    {"id": "0", "contents": "..."}
    {"id": "1", "contents": "..."}
    ```

2.  **Index Construction:** Build your index using the provided scripts. Examples for dense and sparse retrieval methods are given in the original README.

## :gear: Components

FlashRAG offers a suite of RAG components to build customized pipelines.

#### RAG-Components

| Type        | Module            | Description                                                                 |
| ----------- | ----------------- | --------------------------------------------------------------------------- |
| Judger      | SKR Judger        | Judging whether to retrieve using SKR method                 |
| Retriever   | Dense Retriever   | Bi-encoder models (dpr, bge, e5, etc.) with Faiss.                             |
| Retriever   | BM25 Retriever    | Sparse retrieval using Lucene                                                   |
| Retriever   | Bi-Encoder Reranker | Calculate matching score using bi-Encoder                    |
| Retriever   | Cross-Encoder Reranker | Calculate matching score using cross-encoder                    |
| Refiner     | Extractive Refiner | Refine input by extracting important context                       |
| Refiner     | Abstractive Refiner | Refine input through seq2seq model                      |
| Refiner     | LLMLingua Refiner | LLMLingua-series prompt compressor                                             |
| Refiner     | SelectiveContext Refiner | Selective-Context prompt compressor                                              |
| Refiner     | KG Refiner | Use Trace method to construct a knowledge graph                                 |
| Generator   | Encoder-Decoder   | Encoder-Decoder model, supporting Fusion-in-Decoder (FiD).                 |
| Generator   | Decoder-only      | Native transformers implementation.                                         |
| Generator   | FastChat          | Accelerate with FastChat.                                                  |
| Generator   | vllm            | Accelerate with vllm.                                                  |

#### Pipelines

FlashRAG's pre-built pipelines simplify RAG implementation.

| Type        | Module             | Description                                                                  |
| ----------- | ------------------ | ---------------------------------------------------------------------------- |
| Sequential  | Sequential Pipeline  | Linear execution of query, supporting refiner, reranker                 |
| Conditional | Conditional Pipeline | Different paths based on query types (with judger).                          |
| Branching   | REPLUG Pipeline     | Integrate probabilities from multiple generation paths.                     |
| Branching   | SuRe Pipeline       | Ranking and merging generated results based on each document |
| Loop        | Iterative Pipeline   | Alternating retrieval and generation.                                      |
| Loop        | Self-Ask Pipeline    | Decompose complex problems with self-ask.                                  |
| Loop        | Self-RAG Pipeline    | Adaptive retrieval, critique, and generation.                               |
| Loop        | FLARE Pipeline       | Dynamic retrieval during generation.                                       |
| Loop        | IRCoT Pipeline       | Integrate retrieval process with CoT                                      |
| Loop        | Reasoning Pipeline | Reasoning with retrieval                                     |

## :art: FlashRAG-UI

Explore FlashRAG's capabilities with the user-friendly **FlashRAG-UI**:

*   **One-Click Configuration:** Easily load parameters and configuration files.
*   **Quick Method Experience:** Load corpora and indexes to explore different RAG methods.
*   **Efficient Benchmark Reproduction:** Reproduce built-in baselines and test on curated benchmarks.

<details>
<summary>Show more</summary>
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
</details>

To start the UI:

```bash
cd webui
python interface.py
```

## :robot: Supporting Methods

FlashRAG supports **23 implemented works**, evaluated with standard settings:

-   **Generator:** LLAMA3-8B-instruct with input length of 2048
-   **Retriever:** e5-base-v2 as embedding model, retrieve 5 docs per query
-   **Prompt:** A consistent default prompt

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

Access **36 pre-processed datasets** for comprehensive RAG evaluation, including:

*   NQ, TriviaQA, PopQA, SQuAD, MSMARCO-QA, NarrativeQA, WikiQA, WebQuestions, and more.

### Document Corpus

Utilize document corpora in `jsonl` format:

```jsonl
{"id":"0", "contents": "..."}
{"id":"1", "contents": "..."}
```

Process Wikipedia and MS MARCO with provided scripts and resources.

Preprocessed index is available for download on ModelScope: [FlashRAG_Dataset/retrieval_corpus/wiki18_100w_e5_index.zip](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/file/view/master?id=47985&status=2&fileName=retrieval_corpus%252Fwiki18_100w_e5_index.zip).

## :raised_hands: Additional FAQs

*   [Configuration Guide](./docs/original_docs/configuration.md)
*   [Corpus Building](./docs/original_docs/process-wiki.md)
*   [Index Building](./docs/original_docs/building-index.md)
*   [Reproducing Methods](./docs/original_docs/reproduce_experiment.md)

## :bookmark: License

FlashRAG is released under the [MIT License](./LICENSE).

## :star2: Citation

Cite our paper if you use FlashRAG:

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