# FlashRAG: A Python Toolkit for Efficient Retrieval-Augmented Generation (RAG) Research

**Tackle complex RAG research with FlashRAG, a powerful Python toolkit boasting 23 state-of-the-art RAG algorithms and 36 pre-processed benchmark datasets.  [Explore the FlashRAG repository](https://github.com/RUC-NLPIR/FlashRAG).**

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


FlashRAG is a comprehensive Python toolkit designed for researchers and developers working with Retrieval Augmented Generation (RAG) models.  It provides a flexible and efficient framework for building, experimenting with, and evaluating RAG systems.

<p align="center">
<img src="asset/framework.jpg">
</p>

## Key Features

*   **Extensive and Customizable Framework:** Build complex pipelines with essential RAG components like retrievers, rerankers, generators, and compressors.
*   **Comprehensive Benchmark Datasets:** Evaluate your models using 36 pre-processed RAG benchmark datasets.
*   **Pre-implemented Advanced RAG Algorithms:**  Reproduce and experiment with **23 state-of-the-art RAG algorithms**, including **7 reasoning-based methods**.
*   **🚀 Reasoning-based Methods:**  Explore cutting-edge RAG approaches that integrate reasoning capabilities for enhanced performance on complex, multi-hop tasks.
*   **Efficient Preprocessing:** Simplify the RAG workflow with pre-built scripts for corpus processing, retrieval index building, and document pre-retrieval.
*   **Optimized Execution:** Leverage tools like vLLM, FastChat for LLM inference acceleration, and Faiss for vector index management for optimal performance.
*   **User-Friendly UI:** Easily configure, experiment with, and evaluate RAG models using our intuitive FlashRAG-UI.

## Installation

```bash
pip install flashrag-dev --pre
```

See the original [README](https://github.com/RUC-NLPIR/FlashRAG) for more installation details, optional dependencies, and specific instructions for CPU/GPU setups.

## Quick Start

### Corpus Construction & Indexing

FlashRAG supports building indexes for both dense and sparse retrieval methods.

*   **Dense Retrieval (e.g., e5, BGE):** Uses `faiss` for efficient indexing.
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

*   **Sparse Retrieval (BM25):** Uses `Pyserini` or `bm25s` for inverted index creation.
    ```bash
    python -m flashrag.retriever.index_builder \
      --retrieval_method bm25 \
      --corpus_path indexes/sample_corpus.jsonl \
      --bm25_backend bm25s \
      --save_dir indexes/
    ```
    ```bash
    python -m flashrag.retriever.index_builder \
      --retrieval_method bm25 \
      --corpus_path indexes/sample_corpus.jsonl \
      --bm25_backend pyserini \
      --save_dir indexes/
    ```

See the original [README](https://github.com/RUC-NLPIR/FlashRAG) for further setup details.

### Ready-Made Pipeline

```python
from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline
from flashrag.prompt import PromptTemplate

config_dict = {'data_dir': 'dataset/'}
my_config = Config(
    config_file_path = 'my_config.yaml',
    config_dict = config_dict
)
all_split = get_dataset(my_config)
test_data = all_split['test']

pipeline = SequentialPipeline(my_config)
```
  See the original [README](https://github.com/RUC-NLPIR/FlashRAG) for more implementation details.

## Components

FlashRAG offers a modular approach, providing a wide range of components to build customized RAG pipelines.

#### RAG-Components

| Type          | Module             | Description                                                 |
| ------------- | ------------------ | ----------------------------------------------------------- |
| Judger      | SKR Judger         |  Determining whether to retrieve using the SKR method            |
| Retriever     | Dense Retriever    | Bi-encoder models (dpr, bge, e5), uses faiss for search     |
| Retriever     | BM25 Retriever     | Sparse retrieval method based on Lucene                    |
| Retriever     | Bi-Encoder Reranker | Calculate matching score using bi-Encoder              |
| Retriever     | Cross-Encoder Reranker | Calculate matching score using cross-encoder              |
| Refiner       | Extractive Refiner | Refine input by extracting important context                 |
| Refiner       | Abstractive Refiner| Refine input through seq2seq model                 |
| Refiner       | LLMLingua Refiner  | LLMLingua-series prompt compressor                         |
| Refiner       | SelectiveContext Refiner  | Selective-Context prompt compressor |
| Refiner       | KG Refiner |Use Trace method to construct a knowledge graph  |
| Generator     | Encoder-Decoder Generator | Encoder-Decoder model, supports FiD                    |
| Generator     | Decoder-only Generator | Native transformers implementation                         |
| Generator     | FastChat Generator   | Accelerate with FastChat                         |
| Generator     | vllm Generator   | Accelerate with vllm |


#### Pipelines

| Type          | Module             | Description                                                 |
| ------------- | ------------------ | ----------------------------------------------------------- |
| Sequential    | Sequential Pipeline   | Linear execution of query, supporting refiner, reranker               |
| Conditional   | Conditional Pipeline  | Distinct execution paths for various query types               |
| Branching     | REPLUG Pipeline     | Generate answer by integrating probabilities in multiple generation paths               |
| Branching     | SuRe Pipeline     | Ranking and merging generated results based on each document               |
| Loop          | Iterative Pipeline  | Alternating retrieval and generation                       |
| Loop          | Self-Ask Pipeline   | Decompose complex problems into subproblems using self-ask  |
| Loop          | Self-RAG Pipeline   | Adaptive retrieval, critique, and generation              |
| Loop          | FLARE Pipeline      | Dynamic retrieval during the generation process              |
| Loop          | IRCoT Pipeline     | Integrate retrieval process with CoT                       |
| Loop          | Reasoning Pipeline  | Reasoning with retrieval                |


## FlashRAG-UI

**FlashRAG-UI** offers a user-friendly and visually appealing interface for configuring, experimenting with, and evaluating RAG methods.

### Key Features

-   **One-Click Configuration Loading:** Easily load parameters and configuration files.
-   **Quick Method Experience:** Quickly explore the characteristics of various RAG methods.
-   **Efficient Benchmark Reproduction:** Reproduce built-in baseline methods and benchmarks.

```bash
cd webui
python interface.py
```
  See the original [README](https://github.com/RUC-NLPIR/FlashRAG) for UI screenshots.

## Supporting Methods

FlashRAG supports **23 RAG methods**, offering a consistent setup for easy comparison.  See below for a sample of results; full details and configurations are in the original documentation.

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
| [RQRAG](https://arxiv.org/abs/2404.00610)                                   | Loop        | 32.6    | 52.5          | 33.5          | 35.8       | 46.4       | 26.2      |

#### 🚀 Reasoning-based Methods (NEW!)

| Method                                                                                    | Type        | NQ (EM) | TriviaQA (EM) | PopQA (EM) | Hotpotqa (F1) | 2Wiki (F1) |  Musique (F1) | Bamboogle (F1) |
| ----------------------------------------------------------------------------------------- | ----------- | ------- | ------- | ------------- | ------------- | ---------- | ---------- | --------- |
| [Search-R1](https://arxiv.org/abs/2503.09516) | Reasoning | 45.2 | 62.2 | 49.2 | 54.5 | 42.6 | 29.2 |  59.9 |
| [R1-Searcher](https://arxiv.org/pdf/2503.05592) | Reasoning | 36.9 | 61.6 | 42.0 | 49.0 | 49.1 | 24.7 | 57.7 |
| [O2-Searcher](https://arxiv.org/pdf/2505.16582) | Reasoning | 41.4 | 51.4 | 46.8 | 43.4 | 48.6 | 19.0 | 47.6 |
| [AutoRefine](https://www.arxiv.org/pdf/2505.11277) | Reasoning | 43.8 | 59.8 | 32.4 | 54.0 | 50.3 | 23.6 | 46.6 |
| [ReaRAG](https://arxiv.org/abs/2503.21729) | Reasoning | 26.3 | 51.8 | 24.6 | 42.9 | 41.6 | 21.2 | 41.9 |
| [CoRAG](https://arxiv.org/abs/2503.21729) | Reasoning | 40.9 | 63.1 | 36.0 | 56.6 | 60.7 | 31.9 | 54.1 |
| [SimpleDeepSearcher](https://arxiv.org/pdf/2505.16834) | Reasoning | 36.1 | 61.6 | 42.0 | 49.0 | 49.1 | 24.7 | 57.7 |


## Supporting Datasets & Document Corpus

### Datasets

FlashRAG provides 36 pre-processed datasets in a consistent `jsonl` format, making it easier to evaluate RAG models. Datasets are available on [Hugging Face](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets).

### Document Corpus

Supports documents in jsonl format. Example:
```jsonl
{"id":"0", "contents": "..."}
{"id":"1", "contents": "..."}
```
The `contents` key is essential for building the index.

See [README](https://github.com/RUC-NLPIR/FlashRAG) for details about Wikipedia and MS MARCO corpus processing. Preprocessed index is also available on ModelScope at [FlashRAG\_Dataset/retrieval\_corpus/wiki18\_100w\_e5\_index.zip](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/file/view/master?id=47985&status=2&fileName=retrieval_corpus%252Fwiki18_100w_e5_index.zip)

## Additional FAQs

*   [How to configure experimental parameters?](./docs/original_docs/configuration.md)
*   [How to build your own corpus, such as a specific segmented Wikipedia?](./docs/original_docs/process-wiki.md)
*   [How to index your own corpus?](./docs/original_docs/building-index.md)
*   [How to reproduce supporting methods?](./docs/original_docs/reproduce_experiment.md)

## License

FlashRAG is released under the [MIT License](./LICENSE).

## Citation

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