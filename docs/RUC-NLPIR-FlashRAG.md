# FlashRAG: Your Toolkit for Efficient Retrieval-Augmented Generation (RAG) Research

**Empower your RAG research with FlashRAG – a versatile Python toolkit designed for easy reproduction, customization, and exploration of cutting-edge RAG algorithms.**

[View on GitHub](https://github.com/RUC-NLPIR/FlashRAG) | [Read the Paper](https://arxiv.org/abs/2405.13576) | [Hugging Face Datasets](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/) | [ModelScope Datasets](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset) | [DeepWiki Document](https://deepwiki.com/RUC-NLPIR/FlashRAG) | [MIT License](https://github.com/RUC-NLPIR/FlashRAG/blob/main/LICENSE) | [Python](https://img.shields.io/badge/made_with-Python-blue)

<p align="center">
    <a href="#wrench-installation">Installation</a> |
    <a href="#sparkles-features">Features</a> |
    <a href="#rocket-quick-start">Quick Start</a> |
    <a href="#gear-components">Components</a> |
    <a href="#art-flashrag-ui">FlashRAG-UI</a> |
    <a href="#robot-supporting-methods">Supporting Methods</a> |
    <a href="#notebook-supporting-datasets--document-corpus">Supporting Datasets & Document Corpus</a> |
    <a href="#raised_hands-additional-faqs">FAQs</a>
</p>

FlashRAG is a Python toolkit designed to facilitate the research and development of Retrieval-Augmented Generation (RAG) models.  Reproduce state-of-the-art (SOTA) results, experiment with custom RAG pipelines, and leverage our curated resources to accelerate your research. FlashRAG includes 36 pre-processed benchmark RAG datasets, **23 advanced RAG algorithms**, and an easy-to-use UI.  

<p align="center">
<img src="asset/framework.jpg">
</p>

<a href="https://trendshift.io/repositories/10454" target="_blank"><img src="https://trendshift.io/api/badge/repositories/10454" alt="RUC-NLPIR%2FFlashRAG | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

## Key Features:

*   **Extensive and Customizable Framework:** Build complex RAG pipelines with our modular components: retrievers, rerankers, generators, and compressors.
*   **Comprehensive Benchmark Datasets:** Evaluate your models using 36 pre-processed RAG benchmark datasets.
*   **Pre-implemented SOTA Algorithms:** Reproduce results easily with 23 state-of-the-art RAG algorithms, including **7 reasoning-based methods.**
*   **🚀 Reasoning-based Methods:**  Leverage the power of reasoning with **7 reasoning-based methods** that achieve superior performance on complex, multi-hop tasks.
*   **Efficient Preprocessing:** Streamline your workflow with scripts for corpus processing, index building, and document retrieval.
*   **Optimized Execution:** Utilize tools like vLLM and FastChat for LLM inference acceleration and Faiss for vector index management, accelerating your experimentation.
*   **Easy-to-Use UI:**  Quickly configure and experiment with our RAG baselines through a user-friendly visual interface.

## Table of Contents

*   [Installation](#wrench-installation)
*   [Quick Start](#rocket-quick-start)
*   [Features](#sparkles-features)
*   [Components](#gear-components)
*   [FlashRAG-UI](#art-flashrag-ui)
*   [Supporting Methods](#robot-supporting-methods)
*   [Supporting Datasets & Document Corpus](#notebook-supporting-datasets--document-corpus)
*   [FAQs](#raised_hands-additional-faqs)
*   [License](#bookmark-license)
*   [Citation](#star2-citation)

## :wrench: Installation

Install FlashRAG easily with pip:

```bash
pip install flashrag-dev --pre
```

Or install from source:

```bash
git clone https://github.com/RUC-NLPIR/FlashRAG.git
cd FlashRAG
pip install -e .
```

For optional dependencies (vLLM, sentence-transformers, pyserini, and faiss), run:

```bash
pip install flashrag-dev[full]
pip install vllm>=0.4.1
pip install sentence-transformers
pip install pyserini
# CPU-only:
conda install -c pytorch faiss-cpu=1.8.0
# GPU:
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```

## :rocket: Quick Start

### Corpus Construction

Save your corpus as a `jsonl` file:

```jsonl
{"id": "0", "contents": "..."}
{"id": "1", "contents": "..."}
```

### Index Construction

**Dense Retrieval (e.g., E5, BGE):**

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

**Sparse Retrieval (BM25):**

```bash
python -m flashrag.retriever.index_builder \
  --retrieval_method bm25 \
  --corpus_path indexes/sample_corpus.jsonl \
  --bm25_backend bm25s \
  --save_dir indexes/
```

### Using the Ready-Made Pipeline

Load configurations (using `my_config.yaml`):

```python
from flashrag.config import Config
config_dict = {'data_dir': 'dataset/'}
my_config = Config(config_file_path = 'my_config.yaml', config_dict = config_dict)
```

Load dataset and initialize pipeline:

```python
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline
from flashrag.prompt import PromptTemplate
from flashrag.config import Config
config_dict = {'data_dir': 'dataset/'}
my_config = Config(config_file_path = 'my_config.yaml', config_dict = config_dict)
all_split = get_dataset(my_config)
test_data = all_split['test']
pipeline = SequentialPipeline(my_config)
```

Execute and get results:

```python
output_dataset = pipeline.run(test_data, do_eval=True)
```

## :gear: Components

FlashRAG provides modular components for building custom RAG pipelines:

#### RAG-Components

| Type        | Module           | Description                                                                  |
| ----------- | ---------------- | ---------------------------------------------------------------------------- |
| Judger      | SKR Judger      | Judging whether to retrieve using <a href="https://aclanthology.org/2023.findings-emnlp.691.pdf">SKR</a> method |
| Retriever   | Dense Retriever  | Bi-encoder models such as dpr, bge, e5, using faiss for search               |
|             | BM25 Retriever   | Sparse retrieval method based on Lucene                                     |
|             | Bi-Encoder Reranker| Calculate matching score using bi-Encoder                                |
|             | Cross-Encoder Reranker| Calculate matching score using cross-encoder                                |
| Refiner     | Extractive Refiner | Refine input by extracting important context                               |
|             | Abstractive Refiner | Refine input through seq2seq model                                        |
|             | LLMLingua Refiner  | <a href="https://aclanthology.org/2023.emnlp-main.825/">LLMLingua-series</a> prompt compressor |
|             | SelectiveContext Refiner | <a href="https://arxiv.org/abs/2310.06201">Selective-Context</a> prompt compressor |
|             | KG Refiner | Use <a hred='https://arxiv.org/abs/2406.11460'>Trace method to construct a knowledge graph                                |
| Generator   | Encoder-Decoder Generator | Encoder-Decoder model, supporting <a href="https://arxiv.org/abs/2007.01282">Fusion-in-Decoder (FiD)</a> |
|             | Decoder-only Generator | Native transformers implementation                                       |
|             | FastChat Generator   | Accelerate with <a href="https://github.com/lm-sys/FastChat">FastChat</a>            |
|             | vllm Generator     | Accelerate with <a href="https://github.com/vllm-project/vllm">vllm</a>                 |

#### Pipelines

| Type        | Module               | Description                                                                                                                |
| ----------- | -------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Sequential  | Sequential Pipeline    | Linear execution of RAG process, supporting refiner, reranker                                                             |
| Conditional | Conditional Pipeline   | Implements different paths for different types of input queries                                                            |
| Branching   | REPLUG Pipeline        | Generate answer by integrating probabilities in multiple generation paths                                                   |
|  | SuRe Pipeline        | Ranking and merging generated results based on each document                                                   |
| Loop        | Iterative Pipeline     | Alternating retrieval and generation                                                                                       |
|             | Self-Ask Pipeline      | Decompose complex problems into subproblems using <a href="https://arxiv.org/abs/2210.03350">self-ask</a>                  |
|             | Self-RAG Pipeline      | Adaptive retrieval, critique, and generation                                                                              |
|             | FLARE Pipeline         | Dynamic retrieval during the generation process                                                                          |
|             | IRCoT Pipeline         | Integrate retrieval process with CoT                                                                                       |
|             | Reasoning Pipeline      | Reasoning with retrieval                                                                                                   |

## :art: FlashRAG-UI

FlashRAG-UI provides an intuitive, user-friendly interface for configuring, experimenting with, and evaluating RAG methods.

### Features:

*   One-Click Configuration Loading
*   Quick Method Experience
*   Efficient Benchmark Reproduction

#### Experience our meticulously designed FlashRAG-UI—both user-friendly and visually appealing:
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

## :robot: Supporting Methods

We have implemented **23 works**, using the consistent settings described above.

#### 🚀 Reasoning-based Methods (NEW!)

We now support **7 reasoning-based methods** that combine reasoning ability with retrieval, achieving superior performance on complex multi-hop tasks:

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

We provide 36 pre-processed datasets in a consistent `jsonl` format.  Datasets are available on [Huggingface datasets](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets).

### Document Corpus

Supports `jsonl` format:

```jsonl
{"id":"0", "contents": "..."}
{"id":"1", "contents": "..."}
```

Wikipedia and MS MARCO are common choices, and we provide a script for processing Wikipedia. Preprocessed indexes are also available for convenience: [FlashRAG_Dataset/retrieval_corpus/wiki18_100w_e5_index.zip](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/file/view/master?id=47985&status=2&fileName=retrieval_corpus%252Fwiki18_100w_e5_index.zip).

## :raised_hands: Additional FAQs

*   [How to set different experimental parameters?](./docs/original_docs/configuration.md)
*   [How to build my own corpus?](./docs/original_docs/process-wiki.md)
*   [How to index my own corpus?](./docs/original_docs/building-index.md)
*   [How to reproduce supporting methods?](./docs/original_docs/reproduce_experiment.md)

## :bookmark: License

FlashRAG is licensed under the [<u>MIT License</u>](./LICENSE).

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