# FlashRAG: Your Toolkit for Cutting-Edge Retrieval-Augmented Generation (RAG) Research

**FlashRAG** is a powerful, open-source Python toolkit designed to accelerate your RAG research, enabling efficient reproduction and development of state-of-the-art RAG models.  **Unlock the potential of your research with FlashRAG, a versatile toolkit providing pre-implemented algorithms, benchmark datasets, and an intuitive UI for rapid RAG experimentation.**

[![arXiv](https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv)](https://arxiv.org/abs/2405.13576)
[![HuggingFace Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Datasets-27b3b4.svg)](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/)
[![ModelScope Datasets](https://custom-icon-badges.demolab.com/badge/ModelScope%20Datasets-624aff?style=flat&logo=modelscope&logoColor=white)](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset)
[![](https://devin.ai/assets/deepwiki-badge.png)](https://deepwiki.com/RUC-NLPIR/FlashRAG)
[![License](https://img.shields.io/badge/LICENSE-MIT-green)](https://github.com/RUC-NLPIR/FlashRAG/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/made_with-Python-blue)](https://www.python.org/)
[![Trendshift](https://trendshift.io/api/badge/repositories/10454)](https://trendshift.io/repositories/10454)

## Key Features

*   **Extensive & Customizable Framework:** Build complex RAG pipelines with modular components (retrievers, rerankers, generators, compressors).
*   **Comprehensive Benchmark Datasets:**  Access **36 pre-processed benchmark datasets** for rigorous evaluation.
*   **Pre-Implemented State-of-the-Art Algorithms:**  Reproduce and experiment with **23 advanced RAG algorithms**, including results.
*   **🚀 Reasoning-Based Methods (NEW!):**  Explore **7 reasoning-based methods** that excel on complex, multi-hop tasks.
*   **Efficient Preprocessing:**  Simplify RAG preparation with easy-to-use corpus processing, indexing, and pre-retrieval scripts.
*   **Optimized Execution:** Leverage vLLM, FastChat, and Faiss for faster LLM inference and vector index management.
*   **Easy-to-Use UI:**  **[FlashRAG-UI](webui/interface.py)**  Quickly configure, experiment, and evaluate RAG baselines.

## Core Components

*   **Judger:**  SKR Judger
*   **Retriever:** Dense, BM25, Bi-Encoder Reranker, Cross-Encoder Reranker, SPLADE
*   **Refiner:** Extractive, Abstractive, LLMLingua, SelectiveContext, KG Refiner
*   **Generator:** Encoder-Decoder, Decoder-only, FastChat, vLLM
*   **Pipelines:** Sequential, Conditional, Branching (REPLUG, SuRe), Loop (Iterative, Self-Ask, Self-RAG, FLARE, IRCoT, Reasoning)

## Table of Contents

*   [🚀 Quick Start](#rocket-quick-start)
*   [🛠️ Installation](#wrench-installation)
*   [✨ Features](#sparkles-features)
*   [🛣️ Roadmap](#mag_right-roadmap)
*   [🔄 Changelog](#page_with_curl-changelog)
*   [⚙️ Components](#gear-components)
*   [🖥️ FlashRAG-UI](#art-flashrag-ui)
*   [🤖 Supported Methods](#robot-supporting-methods)
*   [📚 Supporting Datasets & Document Corpus](#notebook-supporting-datasets--document-corpus)
*   [❓ FAQs](#raised_hands-additional-faqs)
*   [📜 License](#bookmark-license)
*   [📝 Citation](#star2-citation)

## 🚀 Quick Start

*   **Corpus Construction:**
    *   Save your corpus as a `jsonl` file:  `{"id": "0", "contents": "..."} {"id": "1", "contents": "..."}`
    *   Process Wikipedia:  [Processing Wikipedia](./docs/original_docs/process-wiki.md)

*   **Index Construction:**
    *   **Dense Retrieval:**
        ```bash
        python -m flashrag.retriever.index_builder --retrieval_method e5 --model_path /model/e5-base-v2/ --corpus_path indexes/sample_corpus.jsonl --save_dir indexes/ --use_fp16 --max_length 512 --batch_size 256 --pooling_method mean --faiss_type Flat
        ```
        *  Use --pooling_method, --instruction if needed.
        *  Use --sentence_transformer for sentence transformers (pooling method not required).
    *   **Sparse Retrieval (BM25):**
        *   Building index with BM25s
        ```bash
        python -m flashrag.retriever.index_builder --retrieval_method bm25 --corpus_path indexes/sample_corpus.jsonl --bm25_backend bm25s --save_dir indexes/
        ```
        *   Building index with Pyserini
        ```bash
        python -m flashrag.retriever.index_builder --retrieval_method bm25 --corpus_path indexes/sample_corpus.jsonl --bm25_backend pyserini --save_dir indexes/
        ```
    *   **Sparse Neural Retrieval (SPLADE):**
        ```bash
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
        pip install pyseismic-lsr
        python -m flashrag.retriever.index_builder --retrieval_method splade --model_path retriever/splade-v3 --corpus_embedded_path data/ms_marco/ms_marco_embedded_corpus.jsonl --corpus_path data/ms_marco/ms_marco_corpus.jsonl --save_dir indexes/ --use_fp16 --max_length 512 --batch_size 4 --n_postings 1000 --centroid_fraction 0.2 --min_cluster_size 2 --summary_energy 0.4 --batched_indexing 10000000 --nknn 32
        ```

*   **Using the ready-made pipeline:**
    *  Load the configurations.  See [<u>configuration guidance</u>](./docs/original_docs/configuration.md)
    *  Load the dataset and init the pipeline
    *  Execute `pipeline.run(test_data, do_eval=True)`
    *  Build your own pipeline: [<u>basic usage</u>](./docs/original_docs/basic_usage.md).

## 🛠️ Installation

*   **Install FlashRAG:**

    ```bash
    pip install flashrag-dev --pre
    ```
    or
    ```bash
    git clone https://github.com/RUC-NLPIR/FlashRAG.git
    cd FlashRAG
    pip install -e .
    ```

*   **Install optional dependencies:**

    ```bash
    pip install flashrag-dev[full]
    pip install vllm>=0.4.1
    pip install sentence-transformers
    pip install pyserini
    ```

*   **Install FAISS (conda recommended):**

    ```bash
    conda install -c pytorch faiss-cpu=1.8.0  # CPU-only
    conda install -c pytorch -c nvidia faiss-gpu=1.8.0  # GPU
    ```

## ✨ Features

*   **Extensive and Customizable Framework:** Includes essential components for RAG scenarios such as retrievers, rerankers, generators, and compressors, allowing for flexible assembly of complex pipelines.
*   **Comprehensive Benchmark Datasets:** A collection of 36 pre-processed RAG benchmark datasets to test and validate RAG models' performances.
*   **Pre-implemented Advanced RAG Algorithms:** Features **23 advancing RAG algorithms** with reported results, based on our framework. Easily reproducing results under different settings.
*   **🚀 Reasoning-based Methods:** **NEW!** We now support **7 reasoning-based methods** that combine reasoning ability with retrieval, achieving superior performance on complex multi-hop tasks.
*   **Efficient Preprocessing Stage:** Simplifies the RAG workflow preparation by providing various scripts like corpus processing for retrieval, retrieval index building, and pre-retrieval of documents.
*   **Optimized Execution:** The library's efficiency is enhanced with tools like vLLM, FastChat for LLM inference acceleration, and Faiss for vector index management.
*   **Easy to Use UI:** We have developed a very easy to use UI to easily and quickly configure and experience the RAG baselines we have implemented, as well as run evaluation scripts on a visual interface.

## 🛣️ Roadmap

*   [x] Support OpenAI models
*   [x] Provdide instructions for each component
*   [x] Integrate sentence Transformers
*   [x] Support multimodal RAG
*   [x] Support reasoning-based methods
*   [ ] Inlcude more RAG approaches
*   [ ] Enhance code adaptability and readability
*   [ ] Add support for api-based retriever (vllm server)

## 🔄 Changelog

*   **\[25/08/06] 🎯 NEW!** Added support for **Reasoning Pipeline**.
*   **\[25/03/21] 🚀 Major Update!** Expanded toolkit to support **23 state-of-the-art RAG algorithms**, including **7 reasoning-based methods**.
*   **\[25/02/24] 🔥🔥🔥** Added support for **multimodal RAG**.
*   **\[25/01/21]**  Paper accepted to **ACM Web Conference (WWW 2025)**.
*   **\[25/01/12]** Introduced **FlashRAG-UI**.
*   **\[25/01/11]** Added support for [<u>RQRAG</u>](https://arxiv.org/abs/2404.00610).
*   **\[25/01/07]** Support the aggregation of multiple retrievers.
*   **\[25/01/07]** Integrated **Chunkie** library.
*   **\[24/10/21]** Released a version based on the Paddle framework
*   **\[24/10/13]**  Added a new in-domain dataset - [DomainRAG](https://arxiv.org/pdf/2406.05654).
*   **\[24/09/24]** Released a version based on the MindSpore framework
*   **\[24/09/18]** Introduced lightweight `BM25s` package as an alternative.
*   **\[24/09/09]** Add support for Adaptive-RAG
*   **\[24/08/02]**  Added support for Spring
*   **\[24/07/17]** Updated HuggingFace datasets link.
*   **\[24/07/06]** Added support for Trace
*   **\[24/06/19]** Added support for IRCoT.
*   **\[24/06/15]** Provide a [<u>demo</u>](./examples/quick_start/demo_en.py).
*   **\[24/06/11]** Integrated `sentence transformers`.
*   **\[24/06/05]** Provided detailed documents for reproducing existing methods, configurations.
*   **\[24/06/02]** Provided an introduction of FlashRAG for beginners.
*   **\[24/05/31]** Supported Openai-series models as generator.

## ⚙️ Components

FlashRAG offers modular components for building RAG systems:

#### RAG-Components

| Type           | Module                 | Description                                                                         |
| -------------- | ---------------------- | ----------------------------------------------------------------------------------- |
| Judger         | SKR Judger             | Judge whether to retrieve using <a href="https://aclanthology.org/2023.findings-emnlp.691.pdf">SKR</a> method |
| Retriever      | Dense Retriever        | Bi-encoder models (dpr, bge, e5), using faiss for search.                           |
|                | BM25 Retriever         | Sparse retrieval method based on Lucene.                                         |
|                | Bi-Encoder Reranker    | Calculate matching score using bi-Encoder.                                        |
|                | Cross-Encoder Reranker | Calculate matching score using cross-encoder.                                       |
| Refiner        | Extractive Refiner     | Refine input by extracting important context.                                       |
|                | Abstractive Refiner    | Refine input through seq2seq model.                                                 |
|                | LLMLingua Refiner      | <a href="https://aclanthology.org/2023.emnlp-main.825/">LLMLingua-series</a> prompt compressor.    |
|                | SelectiveContext Refiner | <a href="https://arxiv.org/abs/2310.06201">Selective-Context</a> prompt compressor |
| KG Refiner     | KG Refiner        | Use <a hred='https://arxiv.org/abs/2406.11460'>Trace method to construct a knowledge graph |
| Generator      | Encoder-Decoder Gen.  | Encoder-Decoder, supporting <a href="https://arxiv.org/abs/2007.01282">FiD</a>.       |
|                | Decoder-only Gen.      | Native transformers implementation.                                                |
|                | FastChat Generator     | Accelerate with <a href="https://github.com/lm-sys/FastChat">FastChat</a>.       |
|                | vllm Generator         | Accelerate with <a href="https://github.com/vllm-project/vllm">vllm</a>.          |

#### Pipelines

| Type        | Module              | Description                                                                                                 |
| ----------- | ------------------- | ----------------------------------------------------------------------------------------------------------- |
| Sequential  | Sequential Pipeline | Linear execution of query, supporting refiner, reranker.                                                   |
| Conditional | Conditional Pipeline | Distinct execution paths for various query types.                                                             |
| Branching   | REPLUG Pipeline     | Generate answer by integrating probabilities in multiple generation paths.                                      |
|            | SuRe Pipeline      | Ranking and merging generated results based on each document.                                                       |
| Loop        | Iterative Pipeline  | Alternating retrieval and generation.                                                                      |
|            | Self-Ask Pipeline   | Decompose complex problems into subproblems using <a href="https://arxiv.org/abs/2210.03350">self-ask</a>. |
|            | Self-RAG Pipeline   | Adaptive retrieval, critique, and generation.                                                                |
|            | FLARE Pipeline      | Dynamic retrieval during the generation process.                                                              |
|            | IRCoT Pipeline      | Integrate retrieval process with CoT.                                                                           |
|            | Reasoning Pipeline    | Reasoning with retrieval.                                                                         |

## 🖥️ FlashRAG-UI

**[FlashRAG-UI](webui/interface.py)**  offers a user-friendly, visual interface for RAG experimentation.  Easily load, configure, and evaluate RAG methods:

*   **One-Click Configuration Loading**: Load parameters and config files. Supports preview and saving.
*   **Quick Method Experience**: Load corpora and indexes. Supports easy component and hyperparameter switching.
*   **Efficient Benchmark Reproduction**:  Reproduce baselines and evaluate on benchmarks.

```bash
cd webui
python interface.py
```
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

## 🤖 Supported Methods

FlashRAG implements **23 RAG methods** with consistent settings (Llama3-8B-instruct, e5-base-v2, consistent prompt):

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

## 📚 Supporting Datasets & Document Corpus

*   **Datasets:** Pre-processed **36 datasets** for RAG research, available at [<u>Huggingface datasets</u>](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets).
*   **Document Corpus:** Supports jsonl format:  `{"id":"0", "contents": "..."}`.  Wikipedia and MS MARCO examples provided.
*   **Preprocessed Index:**  Available at [ModelScope](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/file/view/master?id=47985&status=2&fileName=retrieval_corpus%252Fwiki18_100w_e5_index.zip).

## ❓ FAQs

*   [Configuration Guide](./docs/original_docs/configuration.md)
*   [Corpus Creation Guide](./docs/original_docs/process-wiki.md)
*   [Indexing Guide](./docs/original_docs/building-index.md)
*   [Reproducing Methods](./docs/original_docs/reproduce_experiment.md)

## 📜 License

FlashRAG is licensed under the [<u>MIT License</u>](./LICENSE).

## 📝 Citation

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