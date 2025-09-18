# FlashRAG: Your Toolkit for Cutting-Edge Retrieval-Augmented Generation (RAG) Research

**Effortlessly reproduce state-of-the-art RAG models and accelerate your research with FlashRAG, a comprehensive Python toolkit designed for efficiency and customization.**  [Explore the Original Repo](https://github.com/RUC-NLPIR/FlashRAG)

[![arXiv](https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv)](https://arxiv.org/abs/2405.13576)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Datasets-27b3b4.svg)](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/)
[![ModelScope Datasets](https://custom-icon-badges.demolab.com/badge/ModelScope%20Datasets-624aff?style=flat&logo=modelscope&logoColor=white)](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset)
[![DeepWiki Document](https://devin.ai/assets/deepwiki-badge.png)](https://deepwiki.com/RUC-NLPIR/FlashRAG)
[![License](https://img.shields.io/badge/LICENSE-MIT-green)](https://github.com/RUC-NLPIR/FlashRAG/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/made_with-Python-blue)](https://www.python.org/)

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

FlashRAG provides a modular and efficient framework for RAG research, empowering you to easily reproduce existing SOTA results or build your custom RAG pipelines.  It includes a wealth of resources, including pre-processed datasets and pre-implemented algorithms.

<p align="center">
<img src="asset/framework.jpg">
</p>

<p>
<a href="https://trendshift.io/repositories/10454" target="_blank"><img src="https://trendshift.io/api/badge/repositories/10454" alt="RUC-NLPIR%2FFlashRAG | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

## Table of Contents

*   [✨ Features](#sparkles-features)
*   [🛣️ Roadmap](#mag_right-roadmap)
*   [🔄 Changelog](#page_with_curl-changelog)
*   [🛠️ Installation](#wrench-installation)
*   [🚀 Quick Start](#rocket-quick-start)
*   [⚙️ Components](#gear-components)
*   [🎨 FlashRAG-UI](#art-flashrag-ui)
*   [🤖 Supporting Methods](#robot-supporting-methods)
*   [📒 Supporting Datasets & Document Corpus](#notebook-supporting-datasets--document-corpus)
*   [🙋‍♀️ FAQs](#raised_hands-additional-faqs)
*   [📜 License](#bookmark-license)
*   [⭐ Citation](#star2-citation)

## ✨ Features

*   **Modular & Flexible Framework:** Easily assemble complex RAG pipelines using components like retrievers, rerankers, generators, and compressors.
*   **Rich Benchmark Datasets:** Test and validate your models with a collection of **36 pre-processed RAG benchmark datasets**.
*   **Advanced Algorithms:**  Reproduce results with **23 pre-implemented state-of-the-art RAG algorithms**, saving you time and effort.
*   **🧠 Reasoning-based Methods:**  **NEW!**  Support for **7 reasoning-based methods** that combine reasoning with retrieval, boosting performance on complex tasks.
*   **Streamlined Preprocessing:** Simplify your workflow with tools for corpus processing, index building, and document retrieval.
*   **Optimized Execution:**  Leverage vLLM, FastChat, and Faiss for accelerated LLM inference and efficient vector index management.
*   **User-Friendly UI:** The easy-to-use **FlashRAG-UI** provides a visual interface for configuring, experimenting with, and evaluating RAG baselines.

## 🛣️ Roadmap

FlashRAG is continuously evolving; contributions are welcomed!

*   [x] Support OpenAI models
*   [x] Provide instructions for each component
*   [x] Integrate Sentence Transformers
*   [x] Support multimodal RAG
*   [x] Support reasoning-based methods
*   [ ] Include more RAG approaches
*   [ ] Enhance code adaptability and readability
*   [ ] Add support for API-based retriever (vLLM server)

## 🔄 Changelog

*   **[25/08/06] 🎯 NEW!**  Added support for **Reasoning Pipeline**, a novel approach combining reasoning with retrieval, achieving state-of-the-art results on multi-hop datasets.
*   **[25/03/21] 🚀 Major Update!** Expanded toolkit to support **23 state-of-the-art RAG algorithms**, including **7 reasoning-based methods**.
*   **[25/02/24] 🔥🔥🔥**  Added support for **multimodal RAG**, including MLLMs (Llava, Qwen, InternVL) and multimodal retrievers with Clip architecture.
*   **[25/01/21]**  Technical paper [FlashRAG: A Python Toolkit for Efficient RAG Research](https://arxiv.org/abs/2405.13576) accepted to ACM Web Conference (WWW 2025).
*   **[25/01/12]** Introduced **FlashRAG-UI**, a user-friendly interface.
*   **[25/01/11]**  Added support for [<u>RQRAG</u>](https://arxiv.org/abs/2404.00610) method.
*   **[25/01/07]**  Supported the aggregation of multiple retrievers and added the flexible corpus chunking library [**Chunkie**](https://github.com/chonkie-ai/chonkie?tab=readme-ov-file#usage).
*   **[24/10/21]** Released a version based on the Paddle framework.
*   **[24/10/13]** Added the new in-domain dataset and corpus - [DomainRAG](https://arxiv.org/pdf/2406.05654).
*   **[24/09/24]** Released a version based on the MindSpore framework.

<details>
<summary>Show more</summary>

*   **[24/09/18]** Introduced `BM25s` package for faster and easier bm25 retrieval.
*   **[24/09/09]** Added support for [<u>Adaptive-RAG</u>](https://aclanthology.org/2024.naacl-long.389.pdf).
*   **[24/08/02]** Added support for [<u>Spring</u>](https://arxiv.org/abs/2405.19670).
*   **[24/07/17]** Updated Hugging Face dataset links.
*   **[24/07/06]** Added support for [<u>Trace</u>](https://arxiv.org/abs/2406.11460).
*   **[24/06/19]** Added support for [<u>IRCoT</u>](https://arxiv.org/abs/2212.10509).
*   **[24/06/15]** Provided a [<u>demo</u>](./examples/quick_start/demo_en.py).
*   **[24/06/11]** Integrated `sentence transformers` in the retriever module.
*   **[24/06/05]** Provided detailed documentation for reproducing existing methods and configurations.
*   **[24/06/02]** Provided an introduction of FlashRAG for beginners in English, Chinese, and Korean.
*   **[24/05/31]** Supported OpenAI-series models as generators.

</details>

## 🛠️ Installation

[![PyPI - Version](https://img.shields.io/pypi/v/flashrag-dev)](https://pypi.org/project/flashrag-dev/)
[![PyPI - Downloads](https://img.shields.io/pypi/dw/flashrag-dev)](https://pypi.org/project/flashrag-dev/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/flashrag-dev)](https://pypi.org/project/flashrag-dev/)

Install FlashRAG easily with pip:

```bash
pip install flashrag-dev --pre
```

Or, install from source:

```bash
git clone https://github.com/RUC-NLPIR/FlashRAG.git
cd FlashRAG
pip install -e .
```

Optional dependencies for advanced features:

```bash
pip install flashrag-dev[full]  # Install all extra dependencies
pip install vllm>=0.4.1       # Install vLLM for faster inference
pip install sentence-transformers # Install sentence-transformers
pip install pyserini            # Install pyserini for BM25
```

Install `faiss` for dense retrieval (CPU/GPU):

```bash
# CPU-only
conda install -c pytorch faiss-cpu=1.8.0

# GPU(+CPU)
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```
*Note:  Compatibility issues may exist with the latest `faiss` version.*

## 🚀 Quick Start

### 1. Corpus Construction

Create a `jsonl` file with documents in the following format:

```jsonl
{"id": "0", "contents": "..."}
{"id": "1", "contents": "..."}
```

Process Wikipedia using our script: [Processing Wikipedia](./docs/original_docs/process-wiki.md).

### 2. Index Construction

Build an index using the following commands:

*   **Dense Retrieval:**  Use `faiss` for methods like E5 and BGE.

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

*   `--pooling_method`: Specify the pooling method (`mean`, `pooler`, `cls`) for your embedding model if it is not determined automatically.
*   `--instruction`:  Add instructions for models like E5 and BGE if required.
*   **Sparse Retrieval (BM25):**

```bash
python -m flashrag.retriever.index_builder \
  --retrieval_method bm25 \
  --corpus_path indexes/sample_corpus.jsonl \
  --bm25_backend bm25s \  # or pyserini
  --save_dir indexes/
```
### 3. Using the ready-made pipeline
```python
from flashrag.config import Config

# hybrid load configs
config_dict = {'data_dir': 'dataset/'}
my_config = Config(
    config_file_path = 'my_config.yaml',
    config_dict = config_dict
```

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

```python
output_dataset = pipeline.run(test_data, do_eval=True)
```

### 4. Build your own pipeline!

Build your own pipeline by inheriting from `BasicPipeline`, initializing the components you need, and completing the `run` function.

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

### 5. Just use components

If you already have your own code and only want to use our components to embed the original code, you can refer to the [<u>basic introduction of the components</u>](./docs/original_docs/basic_usage.md) to obtain the input and output formats of each component.

## ⚙️ Components

FlashRAG provides versatile components for building RAG systems.

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

FlashRAG organizes RAG methods into pipeline categories:

-   **Sequential:** Query-retriever-generator
-   **Conditional:** Judger-based, different paths for queries.
-   **Branching:** Parallel paths, results merged.
-   **Loop:** Iterative retrieval and generation.

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

## 🎨 FlashRAG-UI

**FlashRAG-UI** provides an intuitive interface for easy experimentation and evaluation:

###  ✨ Features
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

#### Experience our meticulously designed FlashRAG-UI:
```bash
cd webui
python interface.py
```

## 🤖 Supporting Methods

FlashRAG implements **23 RAG methods** with the following base settings:

*   **Generator:** LLAMA3-8B-instruct, input length 2048
*   **Retriever:** e5-base-v2, retrieve 5 docs per query
*   **Prompt:** Default prompt (details in [<u>method details</u>](./docs/original_docs/baseline_details.md).)

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

#### 🧠 Reasoning-based Methods (NEW!)

| Method                                                                                    | Type        | NQ (EM) | TriviaQA (EM) | PopQA (EM) | Hotpotqa (F1) | 2Wiki (F1) |  Musique (F1) | Bamboogle (F1) | Specific setting                             |
| ----------------------------------------------------------------------------------------- | ----------- | ------- | ------- | ------------- | ------------- | ---------- | ---------- | --------- | ----------------------------------------------- |
| [Search-R1](https://arxiv.org/abs/2503.09516) | Reasoning | 45.2 | 62.2 | 49.2 | 54.5 | 42.6 | 29.2 |  59.9 | SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo |
| [R1-Searcher](https://arxiv.org/pdf/2503.05592) | Reasoning | 36.9 | 61.6 | 42.0 | 49.0 | 49.1 | 24.7 | 57.7 | Qwen-2.5-7B-base-RAG-RL |
| [O2-Searcher](https://arxiv.org/pdf/2505.16582) | Reasoning | 41.4 | 51.4 | 46.8 | 43.4 | 48.6 | 19.0 | 47.6 | O2-Searcher-Qwen2.5-3B-GRPO |
| [AutoRefine](https://www.arxiv.org/pdf/2505.11277) | Reasoning | 43.8 | 59.8 | 32.4 | 54.0 | 50.3 | 23.6 | 46.6 | AutoRefine-Qwen2.5-3B-Base |
| [ReaRAG](https://arxiv.org/abs/2503.21729) | Reasoning | 26.3 | 51.8 | 24.6 | 42.9 | 41.6 | 21.2 | 41.9 | ReaRAG-9B |
| [CoRAG](https://arxiv.org/abs/2503.21729) | Reasoning | 40.9 | 63.1 | 36.0 | 56.6 | 60.7 | 31.9 | 54.1 | CoRAG-Llama3.1-8B-MultihopQA |
| [SimpleDeepSearcher](https://arxiv.org/pdf/2505.16834) | Reasoning | 36.1 | 61.6 | 42.0 | 49.0 | 49.1 | 24.7 | 57.7 | Qwen-7B-SimpleDeepSearcher |

*Note:*  Results may vary from original papers due to consistent settings.  See  [<u>reproduce guidance</u>](./docs/original_docs/reproduce_experiment.md) and [<u>method details</u>](./docs/original_docs/baseline_details.md).

## 📒 Supporting Datasets & Document Corpus

### Datasets

FlashRAG provides 36 pre-processed datasets in `jsonl` format:

```python
{
  'id': str,
  'question': str,
  'golden_answers': List[str],
  'metadata': dict
}
```

Datasets include:

| Task                      | Dataset Name    | Knowledge Source | # Train   | # Dev   | # Test |
| ------------------------- | --------------- | ---------------- | --------- | ------- | ------ |
| QA                        | NQ              | wiki             | 79,168    | 8,757   | 3,610  |
| QA                        | TriviaQA        | wiki & web       | 78,785    | 8,837   | 11,313 |
| QA                        | PopQA           | wiki             | /         | /       | 14,267 |
| QA                        | SQuAD           | wiki             | 87,599    | 10,570  | /      |
| QA                        | MSMARCO-QA      | web              | 808,731   | 101,093 | /      |
| QA                        | NarrativeQA     | books and story  | 32,747    | 3,461   | 10,557 |
| QA                        | WikiQA          | wiki             | 20,360    | 2,733   | 6,165  |
| QA                        | WebQuestions    | Google Freebase  | 3,778     | /       | 2,032  |
| QA                        | AmbigQA         | wiki             | 10,036    | 2,002   | /      |
| QA                        | SIQA            | -                | 33,410    | 1,954   | /      |
| QA                        | CommonSenseQA   | -                | 9,741     | 1,221   | /      |
| QA                        | BoolQ           | wiki             | 9,427     | 3,270   | /      |
| QA                        | PIQA            | -                | 16,113    | 1,838   | /      |
| QA                        | Fermi           | wiki             | 8,000     | 1,000   | 1,000  |
| multi-hop QA              | HotpotQA        | wiki             | 90,447    | 7,405   | /      |
| multi-hop QA              | 2WikiMultiHopQA | wiki             | 15,000    | 12,576  | /      |
| multi-hop QA              | Musique         | wiki             | 19,938    | 2,417   | /      |
| multi-hop QA              | Bamboogle       | wiki             | /         | /       | 125    |
| multi-hop QA              | StrategyQA      | wiki             | 2290      | /       | /      |
| Long-form QA              | ASQA            | wiki             | 4,353     | 948     | /      |
| Long-form QA              | ELI5            | Reddit           | 272,634   | 1,507   | /      |
| Long-form QA              | WikiPassageQA   | wiki             | 3,332     | 417     | 416    |
| Open-Domain Summarization | WikiASP         | wiki             | 300,636   | 37,046  | 37,368 |
| multiple-choice           | MMLU            | -                | 99