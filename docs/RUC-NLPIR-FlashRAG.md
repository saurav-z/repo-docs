# ⚡FlashRAG: Your Toolkit for Efficient RAG Research 🚀

> **Effortlessly explore and advance Retrieval Augmented Generation (RAG) with FlashRAG, a Python toolkit designed for both research and practical application.**

[English | [中文](README_zh.md)]

<div align="center">
  <a href="https://arxiv.org/abs/2405.13576" target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv></a>
  <a href="https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/" target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Datasets-27b3b4.svg></a>
  <a href="https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset" target="_blank"><img src=https://custom-icon-badges.demolab.com/badge/ModelScope%20Datasets-624aff?style=flat&logo=modelscope&logoColor=white></a>
  <a href="https://deepwiki.com/RUC-NLPIR/FlashRAG"><img src="https://devin.ai/assets/deepwiki-badge.png" alt="DeepWiki Document" height="20"/></a>
  <a href="https://github.com/RUC-NLPIR/FlashRAG/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green"></a>
  <a><img alt="Static Badge" src="https://img.shields.io/badge/made_with-Python-blue"></a>
</div>

<h4 align="center">
  <a href="#wrench-installation">Installation</a> |
  <a href="#sparkles-features">Features</a> |
  <a href="#rocket-quick-start">Quick-Start</a> |
  <a href="#gear-components"> Components</a> |
  <a href="#art-flashrag-ui"> FlashRAG-UI</a> |
  <a href="#robot-supporting-methods"> Supporting Methods</a> |
  <a href="#notebook-supporting-datasets--document-corpus"> Supporting Datasets</a> |
  <a href="#raised_hands-additional-faqs"> FAQs</a>
</h4>

FlashRAG is a comprehensive Python toolkit designed to simplify the reproduction, development, and exploration of Retrieval Augmented Generation (RAG) research.  It offers a modular framework with pre-built components and a wealth of resources, including 36 pre-processed benchmark RAG datasets and **23 state-of-the-art RAG algorithms**, including **7 reasoning-based methods** that combine reasoning ability with retrieval.

<p align="center">
<img src="asset/framework.jpg" alt="FlashRAG Framework">
</p>

Easily reproduce state-of-the-art (SOTA) results or craft your custom RAG pipelines.  Enhance your research with our easy-to-use UI:

https://github.com/user-attachments/assets/8ca00873-5df2-48a7-b853-89e7b18bc6e9

<p>
<a href="https://trendshift.io/repositories/10454" target="_blank"><img src="https://trendshift.io/api/badge/repositories/10454" alt="RUC-NLPIR%2FFlashRAG | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

## Key Features

*   **Extensive and Customizable Framework:**  Build complex RAG pipelines with modular components for retrievers, rerankers, generators, and compressors.
*   **Comprehensive Benchmark Datasets:** Evaluate and validate your models with 36 pre-processed RAG benchmark datasets.
*   **Pre-implemented Advanced RAG Algorithms:**  Reproduce results easily with 23 advanced RAG algorithms.
*   **🚀 Reasoning-based Methods:** **NEW!**  Leverage 7 reasoning-based methods to significantly boost performance on complex multi-hop tasks.
*   **Efficient Preprocessing:** Simplify RAG workflow preparation with tools for corpus processing, index building, and document pre-retrieval.
*   **Optimized Execution:**  Accelerate performance with vLLM, FastChat, and Faiss integration.
*   **Easy-to-Use UI:**  Visually configure, experiment with, and evaluate RAG baselines using our intuitive UI.

## :link: Navigation

*   [Features](#sparkles-features)
*   [Roadmap](#mag_right-roadmap)
*   [Changelog](#page_with_curl-changelog)
*   [Installation](#wrench-installation)
*   [Quick Start](#rocket-quick-start)
*   [Components](#gear-components)
*   [FlashRAG-UI](#art-flashrag-ui)
*   [Supporting Methods](#robot-supporting-methods)
*   [Supporting Datasets & Document Corpus](#notebook-supporting-datasets--document-corpus)
*   [Additional FAQs](#raised_hands-additional-faqs)
*   [License](#bookmark-license)
*   [Citation](#star2-citation)

## :mag_right: Roadmap

FlashRAG is actively being developed; we welcome community contributions!

*   [x] Support OpenAI models
*   [x] Provide instructions for each component
*   [x] Integrate sentence Transformers
*   [x] Support multimodal RAG
*   [x] Support reasoning-based methods
*   [ ] Inlcude more RAG approaches
*   [ ] Enhance code adaptability and readability
*   [ ] Add support for api-based retriever (vllm server)

## :page_with_curl: Changelog

**Recent Updates:**

*   **[25/08/06] 🎯 NEW! Reasoning Pipeline:** Added support for reasoning-based methods, achieving near 60 F1 on multi-hop inference datasets like HotpotQA (see [**result table**](#robot-supporting-methods)).
*   **[25/03/21] 🚀 Major Update:** Expanded toolkit to support 23 state-of-the-art RAG algorithms, including 7 reasoning-based methods.
*   **[25/02/24] 🔥🔥🔥 Multimodal RAG Support:**  Added support for multimodal RAG, including MLLMs (Llava, Qwen, InternVL) and CLIP-based multimodal retrievers. See the new arxiv article and documentation for details.
*   **[25/01/21] WWW 2025 Acceptance:** Our technical paper [FlashRAG: A Python Toolkit for Efficient RAG Research](https://arxiv.org/abs/2405.13576) has been accepted to the Resource Track of the 2025 ACM Web Conference (WWW 2025).
*   **[25/01/12] FlashRAG-UI Release:**  Introducing an easy-to-use UI for configuring and experimenting with supported RAG methods.
*   **[25/01/11] RQRAG Method Support:** Added support for the [<u>RQRAG</u>](https://arxiv.org/abs/2404.00610) method.
*   **[25/01/07] Multi-Retriever and Chunkie Integration:** Support for aggregating multiple retrievers and integration of the Chunkie library.
*   **[24/10/21 & 24/09/24] Paddle & MindSpore Frameworks:** Released versions supporting Chinese hardware platforms (see [FlashRAG Paddle](https://github.com/RUC-NLPIR/FlashRAG-Paddle) and [FlashRAG MindSpore](https://github.com/RUC-NLPIR/FlashRAG-MindSpore)).
*   **[24/10/13] DomainRAG Dataset:** Added the new in-domain dataset [DomainRAG](https://arxiv.org/pdf/2406.05654) for domain-specific RAG testing.

<details>
<summary>Show more</summary>

*   **[24/09/18] BM25s Package Integration:** Introduced `BM25s` package for faster and easier retrieval, deprecating Pyserini in future versions.
*   **[24/09/09] Adaptive-RAG Support:** Added support for the [<u>Adaptive-RAG</u>](https://aclanthology.org/2024.naacl-long.389.pdf) method.
*   **[24/08/02] Spring Support:**  Added support for the [<u>Spring</u>](https://arxiv.org/abs/2405.19670) method.
*   **[24/07/17] Dataset Link Update:** Updated dataset link.
*   **[24/07/06] Trace Support:**  Added support for the [<u>Trace</u>](https://arxiv.org/abs/2406.11460) method.
*   **[24/06/19] IRCoT Support:**  Added support for the [<u>IRCoT</u>](https://arxiv.org/abs/2212.10509) method.
*   **[24/06/15] Demo:**  Provided a [<u>demo</u>](./examples/quick_start/demo_en.py).
*   **[24/06/11] Sentence Transformers Integration:** Integrated `sentence transformers`.
*   **[24/06/05] Detailed Documentation:** Provided detailed documentation for reproducing existing methods and configuration settings.
*   **[24/06/02] Introduction for Beginners:** Provided an introduction of FlashRAG for beginners.
*   **[24/05/31] OpenAI Model Support:** Supported Openai-series models as generator.

</details>

## :wrench: Installation

[![PyPI - Version](https://img.shields.io/pypi/v/flashrag-dev)](https://pypi.org/project/flashrag-dev/)
[![PyPI - Downloads](https://img.shields.io/pypi/dw/flashrag-dev)](https://pypi.org/project/flashrag-dev/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/flashrag-dev)](https://pypi.org/project/flashrag-dev/)

**Install FlashRAG using pip:**

```bash
pip install flashrag-dev --pre
```

**Alternatively, clone and install from GitHub (Python 3.10+ required):**

```bash
git clone https://github.com/RUC-NLPIR/FlashRAG.git
cd FlashRAG
pip install -e .
```

**Install Optional Dependencies (vLLM, Sentence Transformers, Pyserini):**

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

**Install Faiss (CPU or GPU):**

```bash
# CPU-only version
conda install -c pytorch faiss-cpu=1.8.0

# GPU(+CPU) version
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```

**Note:**  Installing the latest version of `faiss` might be impossible on certain systems. See the Faiss documentation for more information.

## :rocket: Quick Start

### Corpus Construction

Build a `jsonl` file with each line representing a document:

```jsonl
{"id": "0", "contents": "..."}
{"id": "1", "contents": "..."}
```

See [Processing Wikipedia](./docs/original_docs/process-wiki.md) for an example of how to convert Wikipedia to an indexable format.

### Index Construction

**For Dense Retrieval Methods (e.g., embedding models):**

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

*   `--pooling_method`:  Specify the pooling method (`mean`, `pooler`, or `cls`).  If omitted, it will be automatically selected based on model.  **Ensure accuracy by specifying it.**
*   `---instruction`: If your embedding model needs additional instructions for query encoding, specify them here. Auto-filled for **E5** and **BGE** models.

If the retrieval model supports `sentence transformers`, use:

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
  --sentence_transformer \
  --faiss_type Flat
```

**For Sparse Retrieval Methods (BM25):**
For BM25 index, do not need to specify `model_path`.

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

### Using the ready-made pipeline

Configure the RAG process with the config:

```python
from flashrag.config import Config

# hybrid load configs
config_dict = {'data_dir': 'dataset/'}
my_config = Config(
    config_file_path = 'my_config.yaml',
    config_dict = config_dict
```

*   **Variables take precedence over files.**  See [<u>configuration guidance</u>](./docs/original_docs/configuration.md) and the [<u>basic yaml file</u>](./flashrag/config/basic_config.yaml) for configuration.

Load the dataset and initialize the pipeline:

```python
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline
from flashrag.prompt import PromptTemplate
from flashrag.config import Config

config_dict = {'data_dir': 'dataset/'}
my_config = Config(
    config_file_path = 'my_config.yaml',
    config_dict = config_dict
)
all_split = get_dataset(my_config)
test_data = all_split['test']

pipeline = SequentialPipeline(my_config)
```

Use `PromptTemplete` to customize your input prompt:

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

Run the pipeline and obtain the results:

```python
output_dataset = pipeline.run(test_data, do_eval=True)
```

`output_dataset` contains intermediate results and metric scores.

### Build your own pipeline!

Inherit `BasicPipeline`, initialize components, and implement the `run` function:

```python
from flashrag.pipeline import BasicPipeline
from flashrag.utils import get_retriever, get_generator

class ToyPipeline(BasicPipeline):
  def __init__(self, config, prompt_templete=None):
    # Load your own components
    pass

  def run(self, dataset, do_eval=True):
    # Complete your own process logic

    # get attribute in dataset using `.`
    input_query = dataset.question
    ...
    # use `update_output` to save intermeidate data
    dataset.update_output("pred",pred_answer_list)
    dataset = self.evaluate(dataset, do_eval=do_eval)
    return dataset
```

### Just use components

Use our components for embedding: see the [<u>basic introduction of the components</u>](./docs/original_docs/basic_usage.md) for input/output formats.

## :gear: Components

FlashRAG provides core RAG components, including retrievers, generators, refiners, and pipelines.

#### RAG-Components

| Type          | Module                 | Description                                                                   |
| ------------- | ---------------------- | ----------------------------------------------------------------------------- |
| Judger        | SKR Judger             | Judging whether to retrieve using <a href="https://aclanthology.org/2023.findings-emnlp.691.pdf">SKR</a> method |
| Retriever     | Dense Retriever        | Bi-encoder models (dpr, bge, e5, using faiss)                               |
|               | BM25 Retriever       | Sparse retrieval (Lucene based)                                             |
|               | Bi-Encoder Reranker    | Calculate matching score using bi-Encoder                                       |
|               | Cross-Encoder Reranker | Calculate matching score using cross-encoder                                    |
| Refiner       | Extractive Refiner     | Extracting important context.                                              |
|               | Abstractive Refiner    | Refine input through seq2seq model.                                         |
|               | LLMLingua Refiner      | <a href="https://aclanthology.org/2023.emnlp-main.825/">LLMLingua-series</a> prompt compressor                                |
|               | SelectiveContext Refiner | <a href="https://arxiv.org/abs/2310.06201">Selective-Context</a> prompt compressor                               |
|               | KG Refiner | Use <a hred='https://arxiv.org/abs/2406.11460'>Trace method to construct a knowledge graph.          |
| Generator     | Encoder-Decoder Generator | Supports <a href="https://arxiv.org/abs/2007.01282">Fusion-in-Decoder (FiD)</a>               |
|               | Decoder-only Generator | Native transformers implementation                                              |
|               | FastChat Generator     | Accelerate with <a href="https://github.com/lm-sys/FastChat">FastChat</a>     |
|               | vllm Generator         | Accelerate with <a href="https://github.com/vllm-project/vllm">vllm</a>         |

#### Pipelines

RAG methods are categorized into four types based on inference paths.  We've implemented corresponding pipelines for each.

| Type        | Module               | Description                                           |
| ----------- | -------------------- | ----------------------------------------------------- |
| Sequential  | Sequential Pipeline  | Linear execution, supporting refiner/reranker        |
| Conditional | Conditional Pipeline | Distinct execution paths for query types             |
| Branching   | REPLUG Pipeline      | Integrate probabilities in multiple generation paths  |
|             | SuRe Pipeline        | Ranking and merging generated results                |
| Loop        | Iterative Pipeline   | Alternating retrieval and generation                  |
|             | Self-Ask Pipeline    | Decompose problems using <a href="https://arxiv.org/abs/2210.03350">self-ask</a> |
|             | Self-RAG Pipeline    | Adaptive retrieval, critique, and generation          |
|             | FLARE Pipeline       | Dynamic retrieval during generation                   |
|             | IRCoT Pipeline       | Integrate retrieval with CoT                         |
|             | Reasoning Pipeline   | Reasoning with retrieval. |

## :art: FlashRAG-UI

<p>With <strong>FlashRAG-UI</strong>, you can easily and quickly configure and experience the supported RAG methods through our meticulously designed visual interface, and evaluate these methods on benchmarks, making complex research work more efficient!</p>

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

#### Experience our meticulously designed FlashRAG-UI—both user-friendly and visually appealing:
```bash
cd webui
python interface.py
```

## :robot: Supporting Methods

We've implemented **23 works**, using consistent settings:

-   **Generator:** LLAMA3-8B-instruct (input length 2048)
-   **Retriever:** e5-base-v2 (5 docs per query)
-   **Prompt:** Consistent default prompt (see [<u>method details</u>](./docs/original_docs/baseline_details.md)).

For open-source methods, we implemented their processes using our framework. For methods where the author did not provide source code, we will try our best to follow the methods in the original paper for implementation.

For necessary settings and hyperparameters specific to some methods, we have documented them in the **specific settings** column. For more details, please consult our [<u>reproduce guidance</u>](./docs/original_docs/reproduce_experiment.md) and [<u>method details</u>](./docs/original_docs/baseline_details.md).

Note that, to ensure consistency, we have utilized a uniform setting. However, this setting may differ from the original setting of the method, leading to variations in results compared to the original outcomes.

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

FlashRAG provides 36 pre-processed