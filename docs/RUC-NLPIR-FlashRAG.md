# ⚡ FlashRAG: Your Toolkit for Efficient Retrieval-Augmented Generation (RAG) Research

> **Accelerate your RAG research!** FlashRAG is a powerful Python toolkit designed for reproducing and developing cutting-edge Retrieval-Augmented Generation (RAG) models. Explore state-of-the-art algorithms, benchmark datasets, and a user-friendly UI – all in one place.  [Explore the original repository](https://github.com/RUC-NLPIR/FlashRAG).

[![arXiv](https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv)](https://arxiv.org/abs/2405.13576)
[![HuggingFace Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Datasets-27b3b4.svg)](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/)
[![ModelScope Datasets](https://custom-icon-badges.demolab.com/badge/ModelScope%20Datasets-624aff?style=flat&logo=modelscope&logoColor=white)](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset)
[![DeepWiki Document](https://devin.ai/assets/deepwiki-badge.png)](https://deepwiki.com/RUC-NLPIR/FlashRAG)
[![License](https://img.shields.io/badge/LICENSE-MIT-green)](https://github.com/RUC-NLPIR/FlashRAG/blob/main/LICENSE)
[![Made with Python](https://img.shields.io/badge/made_with-Python-blue)](https://www.python.org/)

<p align="center">
<a href="#wrench-installation">Installation</a> |
<a href="#sparkles-features">Features</a> |
<a href="#rocket-quick-start">Quick-Start</a> |
<a href="#gear-components"> Components</a> |
<a href="#art-flashrag-ui"> FlashRAG-UI</a> |
<a href="#robot-supporting-methods"> Supporting Methods</a> |
<a href="#notebook-supporting-datasets--document-corpus"> Supporting Datasets</a> |
<a href="#raised_hands-additional-faqs"> FAQs</a>
</p>

FlashRAG provides a comprehensive framework for researchers and developers working with RAG models, offering:

*   **36 pre-processed benchmark RAG datasets.**
*   **23 state-of-the-art RAG algorithms.**
*   **7 reasoning-based methods** that integrate reasoning with retrieval, enhancing performance on complex tasks.

<p align="center">
<img src="asset/framework.jpg">
</p>

## Key Features

*   **Extensive and Customizable Framework:** Build complex RAG pipelines with modular components like retrievers, rerankers, generators, and compressors.
*   **Comprehensive Benchmark Datasets:** Evaluate and validate your RAG models with 36 pre-processed datasets.
*   **Pre-implemented Advanced RAG Algorithms:** Easily reproduce results and explore 23 cutting-edge RAG algorithms.
*   **🚀 Reasoning-based Methods:** Explore 7 reasoning-based methods that improve performance on complex multi-hop tasks.
*   **Efficient Preprocessing:** Streamline your workflow with scripts for corpus processing, index building, and document retrieval.
*   **Optimized Execution:** Leverage vLLM, FastChat, and Faiss for faster LLM inference and efficient vector index management.
*   **Easy-to-Use UI:** A visual interface for easy configuration, experimentation, and evaluation of RAG baselines.

## Installation

Get started with FlashRAG in a few simple steps:

```bash
pip install flashrag-dev --pre
```

For more detailed installation instructions, including optional dependencies like `vllm`, `sentence-transformers`, and `pyserini`, refer to the [Installation section](#wrench-installation) in the original README.

## Quick Start

Quickly get up and running with FlashRAG by following these steps:

1.  **Corpus Construction:** Prepare your corpus as a `jsonl` file, with each line representing a document in the format:

```jsonl
{"id": "0", "contents": "..."}
{"id": "1", "contents": "..."}
```

2.  **Index Construction:**  Build your index using the provided scripts.  Examples are given for dense (using Faiss) and sparse retrieval methods (BM25 with Pyserini or `bm25s`), and SPLADE.

```bash
# Dense Retrieval (e.g., with E5 embeddings)
python -m flashrag.retriever.index_builder \
  --retrieval_method e5 \
  --model_path /model/e5-base-v2/ \
  --corpus_path indexes/sample_corpus.jsonl \
  --save_dir indexes/ \
  ...

# BM25 retrieval
python -m flashrag.retriever.index_builder \
  --retrieval_method bm25 \
  --corpus_path indexes/sample_corpus.jsonl \
  --bm25_backend bm25s \
  --save_dir indexes/ 
```

3.  **Using a Ready-Made Pipeline:** Utilize the pre-built `SequentialPipeline` class to implement the RAG process.

```python
from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline
from flashrag.prompt import PromptTemplate

# hybrid load configs
config_dict = {'data_dir': 'dataset/'}
my_config = Config(
    config_file_path = 'my_config.yaml',
    config_dict = config_dict
)
all_split = get_dataset(my_config)
test_data = all_split['test']

pipeline = SequentialPipeline(my_config)

output_dataset = pipeline.run(test_data, do_eval=True)
```

4.  **Build Your Own Pipeline:** Customize the RAG process by inheriting `BasicPipeline` and implementing your own `run` function.

For detailed instructions and more options, see the [Quick Start](#rocket-quick-start) section in the original README.

## Components

FlashRAG provides modular components for building RAG systems.

### RAG Components:

| Type        | Module          | Description                                                                                                                                                                                |
| ----------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Judger      | SKR Judger      | Judging whether to retrieve using [SKR](https://aclanthology.org/2023.findings-emnlp.691.pdf) method |
| Retriever   | Dense Retriever | Bi-encoder models (e.g., dpr, bge, e5) using Faiss for search.                                                                                                                           |
| Retriever   | BM25 Retriever  | Sparse retrieval method based on Lucene.                                                                                                                                                 |
| Retriever   | Bi-Encoder Reranker  | Calculate matching score using bi-Encoder.                                                                                                                                                 |
| Retriever   | Cross-Encoder Reranker  | Calculate matching score using cross-encoder.                                                                                                                                                 |
| Refiner     | Extractive Refiner | Refine input by extracting important context.                                                                                                                                          |
| Refiner     | Abstractive Refiner | Refine input through seq2seq model.                                                                                                                                          |
| Refiner   | LLMLingua Refiner | [LLMLingua-series](https://aclanthology.org/2023.emnlp-main.825/) prompt compressor.                                                                                                                                       |
| Refiner  | SelectiveContext Refiner | [Selective-Context](https://arxiv.org/abs/2310.06201) prompt compressor.                                                                                                                                       |
| Refiner     | KG Refiner      | Use [Trace method to construct a knowledge graph](https://arxiv.org/abs/2406.11460).                                                                                                                              |
| Generator   | Encoder-Decoder Generator  | Encoder-Decoder model, supporting [Fusion-in-Decoder (FiD)](https://arxiv.org/abs/2007.01282).                                                                                   |
| Generator   | Decoder-only Generator  | Native transformers implementation.                                                                                                                                             |
| Generator   | FastChat Generator  | Accelerate with [FastChat](https://github.com/lm-sys/FastChat).                                                                                                                                |
| Generator   | vllm Generator  | Accelerate with [vllm](https://github.com/vllm-project/vllm).                                                                                                                                |


### Pipelines

FlashRAG's pipelines implement different RAG methods.

| Type          | Module              | Description                                                                                                    |
| ------------- | ------------------- | -------------------------------------------------------------------------------------------------------------- |
| Sequential    | Sequential Pipeline | Linear execution of query, supporting refiner, reranker.                                                   |
| Conditional   | Conditional Pipeline | Distinct execution paths based on query type, using a judger module.                                        |
| Branching   | REPLUG Pipeline       | Generate answer by integrating probabilities in multiple generation paths.                                              |
| Branching     | SuRe Pipeline | Ranking and merging generated results based on each document.                               |
| Loop          | Iterative Pipeline  | Alternating retrieval and generation.                                                                        |
| Loop          | Self-Ask Pipeline   | Decompose complex problems into subproblems using [self-ask](https://arxiv.org/abs/2210.03350).                |
| Loop          | Self-RAG Pipeline   | Adaptive retrieval, critique, and generation.                                                               |
| Loop          | FLARE Pipeline      | Dynamic retrieval during the generation process.                                                              |
| Loop          | IRCoT Pipeline      | Integrate retrieval process with CoT.                                                                         |
| Loop          | Reasoning Pipeline  | Reasoning with retrieval.                                                                         |


## FlashRAG-UI

<p>Experience the power of FlashRAG through our user-friendly and visually appealing <strong>FlashRAG-UI</strong>.  Easily configure, experiment with, and evaluate different RAG methods using our intuitive interface!</p>

### Key Features
*   **One-Click Configuration Loading**: Quickly load parameters and configuration files.
*   **Quick Method Experience**: Explore the characteristics of various RAG methods.
*   **Efficient Benchmark Reproduction**: Easily reproduce built-in baseline methods.

<details>
<summary>Show More UI Screenshots</summary>
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

#### Launch FlashRAG-UI:
```bash
cd webui
python interface.py
```

## Supporting Methods

FlashRAG provides implementations of **23 state-of-the-art RAG algorithms**, evaluated with consistent settings for easy comparison.

### Results:

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

### Reasoning-based Methods (NEW!)

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

FlashRAG provides a comprehensive suite of 36 datasets, pre-processed for consistent use in your research.

*   QA (e.g., NQ, TriviaQA)
*   Multi-hop QA (e.g., HotpotQA, 2WikiMultiHopQA)
*   Long-form QA (e.g., ASQA, ELI5)
*   Open-Domain Summarization (WikiASP)
*   Multiple-choice (e.g., MMLU, TruthfulQA)
*   Fact Verification (FEVER)
*   Dialog Generation (WOW)
*   Entity Linking (AIDA CoNll-yago, WNED)
*   Slot Filling (T-REx, Zero-shot RE)
*   In-domain QA (DomainRAG)

Available at [Huggingface datasets](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets).

### Document Corpus

Supports `jsonl` format:

```jsonl
{"id":"0", "contents": "..."}
{"id":"1", "contents": "..."}
```

For Wikipedia, use the provided [processing script](./docs/original_docs/process-wiki.md).  Preprocessed indices are available for the e5-base-v2 retriever on the wiki18_100w dataset ([ModelScope link](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/file/view/master?id=47985&status=2&fileName=retrieval_corpus%252Fwiki18_100w_e5_index.zip)).

## Awesome Work using FlashRAG

*   [R1-Searcher](https://github.com/SsmallSong/R1-Searcher)
*   [ReSearch](https://github.com/Agent-RL/ReSearch)
*   [AutoCoA](https://github.com/ADaM-BJTU/AutoCoA)

## FAQs

*   [How to set experimental parameters?](./docs/original_docs/configuration.md)
*   [How to build my own corpus?](./docs/original_docs/process-wiki.md)
*   [How to index my own corpus?](./docs/original_docs/building-index.md)
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