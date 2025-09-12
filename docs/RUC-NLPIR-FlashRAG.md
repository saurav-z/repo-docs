# ⚡ FlashRAG: Supercharge Your RAG Research with this Powerful Toolkit

**Tired of reinventing the wheel in Retrieval-Augmented Generation (RAG) research?** FlashRAG is a comprehensive Python toolkit designed to streamline your RAG experiments, providing a modular and efficient framework for reproducing state-of-the-art results and developing innovative RAG solutions. [Explore the FlashRAG repository!](https://github.com/RUC-NLPIR/FlashRAG)

<div align="center">
  <a href="https://arxiv.org/abs/2405.13576" target="_blank"><img src="https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Datasets-27b3b4.svg" alt="Hugging Face Datasets"></a>
  <a href="https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset" target="_blank"><img src="https://custom-icon-badges.demolab.com/badge/ModelScope%20Datasets-624aff?style=flat&logo=modelscope&logoColor=white" alt="ModelScope Datasets"></a>
  <a href="https://deepwiki.com/RUC-NLPIR/FlashRAG"><img src="https://devin.ai/assets/deepwiki-badge.png" alt="DeepWiki Document" height="20"/></a>
  <a href="https://github.com/RUC-NLPIR/FlashRAG/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green" alt="License"></a>
  <img alt="Made with Python" src="https://img.shields.io/badge/made_with-Python-blue" alt="Python">
</div>

<h4 align="center">
  <a href="#wrench-installation">Installation</a> |
  <a href="#sparkles-features">Features</a> |
  <a href="#rocket-quick-start">Quick Start</a> |
  <a href="#gear-components">Components</a> |
  <a href="#art-flashrag-ui">FlashRAG-UI</a> |
  <a href="#robot-supporting-methods">Supporting Methods</a> |
  <a href="#notebook-supporting-datasets--document-corpus">Supporting Datasets</a> |
  <a href="#raised_hands-additional-faqs">FAQs</a>
</h4>

<p align="center">
  <img src="asset/framework.jpg" alt="FlashRAG Framework">
</p>

FlashRAG empowers researchers and developers to:

*   **Reproduce SOTA RAG results effortlessly**
*   **Build custom RAG pipelines with ease**
*   **Experiment with cutting-edge RAG algorithms**

**Key Features:**

*   ✨ **Extensive and Customizable Framework:** Assemble complex RAG pipelines with modular components like retrievers, rerankers, generators, and compressors.
*   📚 **Comprehensive Benchmark Datasets:**  Access 36 pre-processed RAG datasets for robust model evaluation.
*   🚀 **Pre-implemented Advanced RAG Algorithms:**  Reproduce results with 23 state-of-the-art RAG algorithms, including **7 reasoning-based methods** for superior performance on complex tasks.
*   🧠 **Reasoning-based Methods:** Leverage cutting-edge techniques that combine reasoning capabilities with retrieval.
*   ⚙️ **Efficient Preprocessing and Optimized Execution:** Streamline your workflow with efficient tools such as vLLM, FastChat, and Faiss.
*   🖥️ **Easy-to-Use UI:**  Quickly configure, experiment with, and evaluate RAG methods using the intuitive FlashRAG-UI.

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

*   [x] Support OpenAI models
*   [x] Provide instructions for each component
*   [x] Integrate sentence Transformers
*   [x] Support multimodal RAG
*   [x] Support reasoning-based methods
*   [ ] Include more RAG approaches
*   [ ] Enhance code adaptability and readability
*   [ ] Add support for api-based retriever (vllm server)

## :page_with_curl: Changelog

*   **[25/08/06]** 🎯 **NEW!** Added support for **Reasoning Pipeline**, integrating reasoning ability and retrieval (e.g., [R1-Searcher](https://github.com/SsmallSong/R1-Searcher)).  Achieves close to 60 F1 on multi-hop datasets like HotpotQA.  See the [result table](#robot-supporting-methods).
*   **[25/03/21]** 🚀 **Major Update!** Expanded toolkit to include **23 state-of-the-art RAG algorithms**, including **7 reasoning-based methods**, representing a major milestone.
*   **[25/02/24]** 🔥🔥🔥 Added support for **multimodal RAG**, including [MLLMs like Llava, Qwen, InternVL](https://ruc-nlpir.github.io/FlashRAG/#/zh-cn/component/generator?id=%e5%a4%9a%e6%a8%a1%e6%80%81%e7%94%9f%e6%88%90%e5%99%a8) and [multimodal retrievers with Clip architecture](https://ruc-nlpir.github.io/FlashRAG/#/zh-cn/component/retriever?id=%e5%a4%9a%e6%a8%a1%e6%80%81%e6%a3%80%e7%b4%a2%e5%99%a8). More details in our new Arxiv article and documentation.
*   **[25/01/21]** Paper [FlashRAG: A Python Toolkit for Efficient RAG Research](https://arxiv.org/abs/2405.13576) accepted to the Resource Track of the 2025 **ACM Web Conference (WWW 2025)**.
*   **[25/01/12]** Introduced **FlashRAG-UI**, an easy-to-use interface for configuring, experiencing, and evaluating RAG methods.
*   **[25/01/11]** Added support for [<u>RQRAG</u>](https://arxiv.org/abs/2404.00610).
*   **[25/01/07]** Support for aggregation of multiple retrievers.
*   **[25/01/07]** Integrated [**Chunkie**](https://github.com/chonkie-ai/chonkie?tab=readme-ov-file#usage) for flexible corpus chunking.
*   **[24/10/21]** Released a version based on the Paddle framework. See [FlashRAG Paddle](https://github.com/RUC-NLPIR/FlashRAG-Paddle).
*   **[24/10/13]** Added a new in-domain dataset - [DomainRAG](https://arxiv.org/pdf/2406.05654).
*   **[24/09/24]** Released a version based on the MindSpore framework. See [FlashRAG MindSpore](https://github.com/RUC-NLPIR/FlashRAG-MindSpore).

<details>
<summary>Show more</summary>

*   **[24/09/18]** Introduced `BM25s` package for BM25 retrieval.
*   **[24/09/09]** Added support for [<u>Adaptive-RAG</u>](https://aclanthology.org/2024.naacl-long.389.pdf).
*   **[24/08/02]** Added support for [<u>Spring</u>](https://arxiv.org/abs/2405.19670).
*   **[24/07/17]** Updated dataset link.
*   **[24/07/06]** Added support for [<u>Trace</u>](https://arxiv.org/abs/2406.11460).
*   **[24/06/19]** Added support for [<u>IRCoT</u>](https://arxiv.org/abs/2212.10509).
*   **[24/06/15]** Provided a [<u>demo</u>](./examples/quick_start/demo_en.py).
*   **[24/06/11]** Integrated `sentence transformers` in the retriever module.
*   **[24/06/05]** Provided detailed documentation.
*   **[24/06/02]** Provided an introduction of FlashRAG for beginners.
*   **[24/05/31]** Supported Openai-series models as generator.

</details>

## :wrench: Installation

[![PyPI - Version](https://img.shields.io/pypi/v/flashrag-dev)](https://pypi.org/project/flashrag-dev/)
[![PyPI - Downloads](https://img.shields.io/pypi/dw/flashrag-dev)](https://pypi.org/project/flashrag-dev/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/flashrag-dev)](https://pypi.org/project/flashrag-dev/)

Install FlashRAG using pip:

```bash
pip install flashrag-dev --pre
```

Alternatively, clone the repository and install (Python 3.10+ required):

```bash
git clone https://github.com/RUC-NLPIR/FlashRAG.git
cd FlashRAG
pip install -e .
```

Install optional dependencies for vLLM, sentence-transformers, and Pyserini:

```bash
pip install flashrag-dev[full] # Install all extra dependencies
pip install vllm>=0.4.1   # vLLM for faster speed
pip install sentence-transformers  # sentence-transformers
pip install pyserini       # pyserini for bm25
```

For Faiss installation, use conda:

```bash
conda install -c pytorch faiss-cpu=1.8.0  # CPU-only
conda install -c pytorch -c nvidia faiss-gpu=1.8.0 # GPU(+CPU)
```

*Note: Latest `faiss` versions may not install on all systems.*

## :rocket: Quick Start

### Corpus Construction

Create a `jsonl` file for your corpus, with each line representing a document:

```jsonl
{"id": "0", "contents": "..."}
{"id": "1", "contents": "..."}
```

See [Processing Wikipedia](./docs/original_docs/process-wiki.md) for Wikipedia corpus conversion.

### Index Construction

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

*   `--pooling_method`: Specify ('mean', 'pooler', 'cls'). If not provided, it attempts automatic selection, but user-specified settings are recommended.
*   `--instruction`:  Required for certain embedding models; automatically provided for E5 and BGE.

If you use models from the `sentence transformers` library, use:

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

### Using the ready-made pipeline

Load the process config and set the prompt:

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

prompt_templete = PromptTemplate(
    my_config,
    system_prompt = "Answer the question based on the given document. Only give me the answer and do not output any other words.\nThe following are given documents.\n\n{reference}",
    user_prompt = "Question: {question}\nAnswer:"
)
pipeline = SequentialPipeline(
  my_config,
  prompt_template = prompt_templete
)

output_dataset = pipeline.run(test_data, do_eval=True)
```

### Build your own pipeline!

```python
from flashrag.pipeline import BasicPipeline
from flashrag.utils import get_retriever, get_generator

class ToyPipeline(BasicPipeline):
  def __init__(self, config, prompt_templete=None):
    # Load your own components
    pass

  def run(self, dataset, do_eval=True):
    # Complete your own process logic
    input_query = dataset.question
    ...
    dataset.update_output("pred",pred_answer_list)
    dataset = self.evaluate(dataset, do_eval=do_eval)
    return dataset
```

Refer to the [<u>basic usage guide</u>](./docs/original_docs/basic_usage.md) for component input/output formats.

### Just use components

See the [<u>basic component introduction</u>](./docs/original_docs/basic_usage.md) for component usage.

## :gear: Components

FlashRAG provides modular components for RAG, including retrievers, generators, and refiners.

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

RAG methods are categorized based on their inference paths (Sequential, Conditional, Branching, Loop).

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

## :art: FlashRAG-UI

**FlashRAG-UI** provides a user-friendly interface for configuring, experimenting with, and evaluating RAG methods.

### :star2: Features

*   **One-Click Configuration Loading** Load parameter and configuration files.
*   **Quick Method Experience**  Load corpora and index files.
*   **Efficient Benchmark Reproduction** Reproduce built-in baselines and benchmarks.

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

#### Run the UI:

```bash
cd webui
python interface.py
```

## :robot: Supporting Methods

We have implemented **23 works** with a consistent setting of:

*   **Generator:** LLAMA3-8B-instruct (2048 input length)
*   **Retriever:** e5-base-v2 (5 docs per query)
*   **Prompt:** Default prompt (see [<u>method details</u>](./docs/original_docs/baseline_details.md)).

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

We provide 36 pre-processed datasets, available on [<u>Huggingface datasets</u>](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets).

```python
{
  'id': str,
  'question': str,
  'golden_answers': List[str],
  'metadata': dict
}
```

| Task                      | Dataset Name    | Knowledge Source | # Train   | # Dev   | # Test |
| ------------------------- | --------------- | ---------------- | --------- | ------- | ------ |
| QA                        | NQ              | wiki             | 79,168    | 8,757   | 3,610  |
| QA                        | TriviaQA        | wiki & web       | 78,785    | 8,837   | 11,313 |
| QA                        | PopQA           | wiki             | /         | /       | 14,267 |
| QA                        | SQuAD           | wiki             | 87,599    | 10,570  | /      |
| QA                        | MSMARCO-QA      | web              | 808,731   | 101,093 | /      |
| QA                        | NarrativeQA     | books and story  | 32,747    | 3,461   | 10,557 |
| QA                        | WikiQA          | wiki             | 20,360    | 2,733   | 6,165  |
| QA