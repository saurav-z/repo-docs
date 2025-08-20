# FlashRAG: A Python Toolkit for Efficient Retrieval-Augmented Generation (RAG) Research

> **Enhance your RAG research with FlashRAG, a modular and efficient Python toolkit designed for easy implementation and reproduction of state-of-the-art RAG models.**  Explore the [FlashRAG repository](https://github.com/RUC-NLPIR/FlashRAG) for more details.

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

FlashRAG is a comprehensive Python toolkit designed to streamline Retrieval-Augmented Generation (RAG) research. It provides an easy-to-use framework for reproducing existing state-of-the-art (SOTA) works, implementing custom RAG pipelines, and accelerating the development of novel RAG algorithms.  It includes a user-friendly UI for easy configuration, experiment, and evaluation.

<p align="center">
<img src="asset/framework.jpg">
</p>

<p>
<a href="https://trendshift.io/repositories/10454" target="_blank"><img src="https://trendshift.io/api/badge/repositories/10454" alt="RUC-NLPIR%2FFlashRAG | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

## Key Features

*   **Modular & Customizable Framework:**  Build complex RAG pipelines with flexible components including retrievers, rerankers, generators, and compressors.
*   **Extensive Dataset Support:**  Utilize **36 pre-processed benchmark RAG datasets** for thorough model testing and validation.
*   **Pre-implemented SOTA Algorithms:**  Reproduce results easily with **23 state-of-the-art RAG algorithms** with reported results, based on our framework.
*   **Reasoning-Based Methods**: Leverage **7 reasoning-based methods** that combine retrieval with advanced reasoning capabilities for superior performance, particularly on complex, multi-hop tasks.
*   **Efficient Workflow:** Simplify your RAG workflow preparation, index building, and document retrieval via integrated preprocessing steps.
*   **Optimized Execution:** Boost your workflow with vLLM, FastChat, and Faiss for LLM inference acceleration, and vector index management.
*   **Easy-to-Use UI**: Configure, test, and evaluate RAG baselines quickly with a user-friendly visual interface (FlashRAG-UI).

## Installation

FlashRAG is easily installed using pip:

```bash
pip install flashrag-dev --pre
```

or by cloning the repository and installing locally:

```bash
git clone https://github.com/RUC-NLPIR/FlashRAG.git
cd FlashRAG
pip install -e .
```

Optional dependencies for enhanced functionality:

```bash
# Install all extra dependencies
pip install flashrag-dev[full]

# Install vllm for faster speed
pip install vllm>=0.4.1

# Install sentence-transformers
pip install sentence-transformers

# Install pyserini for bm25
pip install pyserini
```

### Conda installation for Faiss

Due to incompatibility when installing `faiss` using `pip`, it is necessary to use the following conda command for installation.

```bash
# CPU-only version
conda install -c pytorch faiss-cpu=1.8.0

# GPU(+CPU) version
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```

## Quick Start

### 1. Corpus Construction

Prepare your corpus as a `jsonl` file:

```jsonl
{"id": "0", "contents": "..."}
{"id": "1", "contents": "..."}
```

For Wikipedia, refer to our documentation on [Processing Wikipedia](./docs/original_docs/process-wiki.md).

### 2. Index Construction

**Dense Retrieval (e.g., with embedding models):**

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

**SPLADE:**

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
### 3. Using the Ready-Made Pipeline

Load configurations, dataset, and pipeline:

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

Customize prompts using `PromptTemplate`:

```python
prompt_templete = PromptTemplate(
    config,
    system_prompt = "Answer the question based on the given document. Only give me the answer and do not output any other words.\nThe following are given documents.\n\n{reference}",
    user_prompt = "Question: {question}\nAnswer:"
)
pipeline = SequentialPipeline(
  my_config,
  prompt_template = prompt_templete
)
```

Run the pipeline:

```python
output_dataset = pipeline.run(test_data, do_eval=True)
```

### 4. Build Your Own Pipeline

Create custom pipelines by inheriting `BasicPipeline` and implementing the `run` function.  See [<u>documentation</u>](./docs/original_docs/basic_usage.md) for component details.

### 5. Just Use Components

Integrate individual FlashRAG components into your existing code. Consult the [<u>documentation</u>](./docs/original_docs/basic_usage.md) for component input/output formats.

## Components

FlashRAG provides modular components for RAG pipelines:

#### RAG-Components

<table>
  <thead>
    <tr>
      <th>Type</th>
      <th>Module</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="1">Judger</td>
      <td>SKR Judger</td>
      <td>Judging whether to retrieve using <a href="https://aclanthology.org/2023.findings-emnlp.691.pdf">SKR</a> method</td>
    </tr>
    <tr>
      <td rowspan="4">Retriever</td>
      <td>Dense Retriever</td>
      <td>Bi-encoder models such as dpr, bge, e5, using faiss for search</td>
    </tr>
    <tr>
      <td>BM25 Retriever</td>
      <td>Sparse retrieval method based on Lucene</td>
    </tr>
    <tr>
      <td>Bi-Encoder Reranker</td>
      <td>Calculate matching score using bi-Encoder</td>
    </tr>
    <tr>
      <td>Cross-Encoder Reranker</td>
      <td>Calculate matching score using cross-encoder</td>
    </tr>
    <tr>
      <td rowspan="5">Refiner</td>
      <td>Extractive Refiner</td>
      <td>Refine input by extracting important context</td>
    </tr>
    <tr>
      <td>Abstractive Refiner</td>
      <td>Refine input through seq2seq model</td>
    </tr>
    <tr>
      <td>LLMLingua Refiner</td>
      <td><a href="https://aclanthology.org/2023.emnlp-main.825/">LLMLingua-series</a> prompt compressor</td>
    </tr>
    <tr>
      <td>SelectiveContext Refiner</td>
      <td><a href="https://arxiv.org/abs/2310.06201">Selective-Context</a> prompt compressor</td>
    </tr>
    <tr>
      <td> KG Refiner </td>
      <td>Use <a hred='https://arxiv.org/abs/2406.11460'>Trace method to construct a knowledge graph</td>
    <tr>
      <td rowspan="4">Generator</td>
      <td>Encoder-Decoder Generator</td>
      <td>Encoder-Decoder model, supporting <a href="https://arxiv.org/abs/2007.01282">Fusion-in-Decoder (FiD)</a></td>
    </tr>
    <tr>
      <td>Decoder-only Generator</td>
      <td>Native transformers implementation</td>
    </tr>
    <tr>
      <td>FastChat Generator</td>
      <td>Accelerate with <a href="https://github.com/lm-sys/FastChat">FastChat</a></td>
    </tr>
    <tr>
      <td>vllm Generator</td>
      <td>Accelerate with <a href="https://github.com/vllm-project/vllm">vllm</a></td>
    </tr>
  </tbody>
</table>

#### Pipelines

RAG pipelines are categorized based on their inference paths:

*   **Sequential**: Linear RAG processes.
*   **Conditional**: Multiple paths based on query type.
*   **Branching**: Parallel paths, merging responses.
*   **Loop**: Iterative retrieval and generation.

Relevant pipeline modules are implemented in each category.

<table>
    <thead>
        <tr>
            <th>Type</th>
            <th>Module</th>
            <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="1">Sequential</td>
            <td>Sequential Pipeline</td>
            <td>Linear execution of query, supporting refiner, reranker</td>
        </tr>
        <tr>
            <td rowspan="1">Conditional</td>
            <td>Conditional Pipeline</td>
            <td>With a judger module, distinct execution paths for various query types</td>
        </tr>
        <tr>
            <td rowspan="2">Branching</td>
            <td>REPLUG Pipeline</td>
            <td>Generate answer by integrating probabilities in multiple generation paths</td>
        </tr>
          <td>SuRe Pipeline</td>
          <td>Ranking and merging generated results based on each document</td>
        </tr>
        <tr>
            <td rowspan="6">Loop</td>
            <td>Iterative Pipeline</td>
            <td>Alternating retrieval and generation</td>
        </tr>
        <tr>
            <td>Self-Ask Pipeline</td>
            <td>Decompose complex problems into subproblems using <a href="https://arxiv.org/abs/2210.03350">self-ask</a> </td>
        </tr>
        <tr>
            <td>Self-RAG Pipeline</td>
            <td>Adaptive retrieval, critique, and generation</td>
        </tr>
        <tr>
            <td>FLARE Pipeline</td>
            <td>Dynamic retrieval during the generation process</td>
        </tr>
        <tr>
            <td>IRCoT Pipeline</td>
            <td>Integrate retrieval process with CoT</td>
        </tr>
        <tr>
            <td>Reasoning Pipeline</td>
            <td>Reasoning with retrieval</td>
        </tr>
    </tbody>
</table>

## FlashRAG-UI

The FlashRAG-UI enables easy configuration, method exploration, and benchmark evaluation through an intuitive visual interface.

### Key Features:

*   **One-Click Configuration:** Load parameters and configuration files.
*   **Quick Method Experience:** Easily load corpora and indexes, switch components, and test hyperparameters.
*   **Efficient Benchmark Reproduction:** Effortlessly reproduce baselines and benchmarks with simple settings.

#### Screenshots

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

#### Running the UI

```bash
cd webui
python interface.py
```

## Supporting Methods

The following performance results were achieved using a consistent setting:

*   **Generator:** LLAMA3-8B-instruct
*   **Retriever:** e5-base-v2
*   **Prompt:** Consistent default prompt (see [<u>method details</u>](./docs/original_docs/baseline_details.md)).

## Supporting Methods (Results)

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

## Supporting Datasets & Document Corpus

### Datasets

FlashRAG supports **36 diverse datasets** for evaluating RAG models. All datasets are available on [<u>Huggingface datasets</u>](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets).

### Document Corpus

Supports JSONL format for retrieval document collections. Includes a [<u>comprehensive script</u>](./docs/original_docs/process-wiki.md) for processing Wikipedia.

Preprocessed index available:  [FlashRAG_Dataset/retrieval_corpus/wiki18_100w_e5_index.zip](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/file/view/master?id=47985&status=2&fileName=retrieval_corpus%252Fwiki18_100w_e5_index.zip).

## Awesome Work Using FlashRAG

*   [R1-Searcher](https://github.com/SsmallSong/R1-Searcher)
*   [ReSearch](https://github.com/Agent-RL/ReSearch)
*   [AutoCoA](https://github.com/ADaM-BJTU/AutoCoA)

## FAQs

*   [How to set experimental parameters?](./docs/original_docs/configuration.md)
*   [How to build my own corpus?](./docs/original_docs/process-wiki.md)
*   [How to index my own corpus?](./docs/original_docs/building-index.md)
*   [How to reproduce supporting methods?](./docs/original_docs/reproduce_experiment.md)

## License

FlashRAG is licensed under the [<u>MIT License</u>](./LICENSE).

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