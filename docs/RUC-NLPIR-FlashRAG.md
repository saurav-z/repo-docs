# FlashRAG: Your Toolkit for Cutting-Edge Retrieval-Augmented Generation (RAG) Research

**Tackle the complexities of RAG with FlashRAG, a versatile Python toolkit enabling efficient research, experimentation, and rapid prototyping of state-of-the-art RAG models.** Explore 23 pre-implemented RAG algorithms, including 7 reasoning-based methods, on 36 pre-processed benchmark datasets.

[**Visit the original repository on GitHub!**](https://github.com/RUC-NLPIR/FlashRAG)

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

## Key Features

*   **Modular and Customizable Framework:** Design and assemble your own RAG pipelines with flexible components like retrievers, rerankers, generators, and compressors.
*   **Comprehensive Benchmark Datasets:** Evaluate your models on a wide range of 36 pre-processed RAG benchmark datasets.
*   **Pre-implemented State-of-the-Art Algorithms:** Easily reproduce and experiment with **23 advanced RAG algorithms**, including **7 reasoning-based methods**, to achieve cutting-edge performance.
*   **Reasoning-Based Methods:** Leverage advanced models that combine reasoning abilities with retrieval, leading to superior performance in complex, multi-hop tasks.
*   **Efficient Workflow:** Streamline your RAG research with optimized preprocessing, efficient execution (vLLM, FastChat, Faiss), and UI for easy experimentation.
*   **User-Friendly UI:** Quickly configure, experiment with, and evaluate RAG baselines through an intuitive visual interface.

## Navigation

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

## Roadmap

*   [x] Support OpenAI models
*   [x] Provdide instructions for each component
*   [x] Integrate sentence Transformers
*   [x] Support multimodal RAG
*   [x] Support reasoning-based methods
*   [ ] Inlcude more RAG approaches
*   [ ] Enhance code adaptability and readability
*   [ ] Add support for api-based retriever (vllm server)

## Changelog
-   [25/08/06] 🎯 **NEW!** We have added support for **Reasoning Pipeline**, which is a new paradigm that combines reasoning ability and retrieval, representing work that includes [R1-Searcher](https://github.com/SsmallSong/R1-Searcher), [Search-R1](https://github.com/PeterGriffinJin/Search-R1),.... We evaluate the performance of the pipeline on various RAG benchmarks, it can achieve F1 scores close to 60 on multi hop inference datasets such as HotpotQA. See it in [**result table**](#robot-supporting-methods).

-   [25/03/21] 🚀 **Major Update!** We have expanded our toolkit to support **23 state-of-the-art RAG algorithms**, including **7 reasoning-based methods** that significantly improve performance on complex reasoning tasks. This represents a major milestone in our toolkit's evolution!

-   [25/02/24] 🔥🔥🔥 We have added support for **multimodal RAG**, including [**MLLMs like Llava, Qwen, InternVL**](https://ruc-nlpir.github.io/FlashRAG/#/zh-cn/component/generator?id=%e5%a4%9a%e6%a8%a1%e6%80%81%e7%94%9f%e6%88%90%e5%99%a8), and various [**multimodal retrievers with Clip architecture**](https://ruc-nlpir.github.io/FlashRAG/#/zh-cn/component/retriever?id=%e5%a4%9a%e6%a8%a1%e6%80%81%e6%a3%80%e7%b4%a2%e5%99%a8). More information can be found in our new version of arxiv article and our documentation. Try it!

-   [25/01/21] Our technical paper [FlashRAG: A Python Toolkit for Efficient RAG Research](https://arxiv.org/abs/2405.13576) is honored to have been accepted to the Resource Track of the 2025 **ACM Web Conference (WWW 2025)**. Please Check it out!

-   [25/01/12] Introduce <strong>FlashRAG-UI</strong>, an easy to use interface. You can easily and quickly configure and experience the supported RAG methods and evaluate them on the benchmarks.

-   [25/01/11] We have added support for a new method [<u>RQRAG</u>](https://arxiv.org/abs/2404.00610) method, see it in [**reproduce\_experiment**](docs/original\_docs/reproduce\_experiment.md).

-   [25/01/07] We have currently support the aggregation of multiple retrievers, see it in [**multi retriever usage**](https://github.com/RUC-NLPIR/FlashRAG/blob/main/docs/original_docs/multi\_retriever\_usage.md).

-   [25/01/07] We have integrated a very flexible and lightweight corpus chunking library [**Chunkie**](https://github.com/chonkie-ai/chonkie?tab=readme-ov-file#usage), which supports various custom chunking methods (tokens, sentences, semantic, etc.). Use it in [<u>chunking doc corpus</u>](docs/original_docs/chunk-doc-corpus.md).

-   [24/10/21] We have released a version based on the Paddle framework that supports Chinese hardware platforms. Please refer to [FlashRAG Paddle](https://github.com/RUC-NLPIR/FlashRAG-Paddle) for details.

-   [24/10/13] A new in-domain dataset and corpus - [DomainRAG](https://arxiv.org/pdf/2406.05654) have been added to the dataset. The dataset is based on the internal enrollment data of Renmin University of China, covering seven types of tasks, which can be used for conducting domain-specific RAG testing.

-   [24/09/24] We have released a version based on the MindSpore framework that supports Chinese hardware platforms. Please refer to [FlashRAG MindSpore](https://github.com/RUC-NLPIR/FlashRAG-MindSpore) for details.

<details>
<summary>Show more</summary>

-   [24/09/18] Due to the complexity and limitations of installing Pyserini in certain environments, we have introduced a lightweight `BM25s` package as an alternative (faster and easier to use). The retriever based on Pyserini will be deprecated in future versions. To use retriever with `bm25s`, just set `bm25_backend` to `bm25s` in config.

-   [24/09/09] We add support for a new method [<u>Adaptive-RAG</u>](https://aclanthology.org/2024.naacl-long.389.pdf), which can automatically select the RAG process to execute based on the type of query. See it result in [<u>result table</u>](#robot-supporting-methods).

-   [24/08/02] We add support for a new method [<u>Spring</u>](https://arxiv.org/abs/2405.19670), significantly improve the performance of LLM by adding only a few token embeddings. See it result in [<u>result table</u>](#robot-supporting-methods).

-   [24/07/17] Due to some unknown issues with HuggingFace, our original dataset link has been invalid. We have updated it. Please check the [new link](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/) if you encounter any problems.

-   [24/07/06] We add support for a new method: [<u>Trace</u>](https://arxiv.org/abs/2406.11460), which refine text by constructing a knowledge graph. See it [<u>results</u>](#robot-supporting-methods) and [<u>details</u>](./docs/original_docs/baseline_details.md).

-   [24/06/19] We add support for a new method: [<u>IRCoT</u>](https://arxiv.org/abs/2212.10509), and update the [<u>result table</u>](#robot-supporting-methods).

-   [24/06/15] We provide a [<u>demo</u>](./examples/quick_start/demo_en.py) to perform the RAG process using our toolkit.

-   [24/06/11] We have integrated `sentence transformers` in the retriever module. Now it's easier to use the retriever without setting pooling methods.

-   [24/06/05] We have provided detailed document for reproducing existing methods (see [how to reproduce](./docs/original_docs/reproduce_experiment.md), [baseline details](./docs/original_docs/baseline_details.md)), and [<u>configurations settings</u>](./docs/original_docs/configuration.md).

-   [24/06/02] We have provided an introduction of FlashRAG for beginners, see [<u>an introduction to flashrag</u>](./docs/original_docs/introduction_for_beginners_en.md) ([<u>中文版</u>](./docs/original_docs/introduction_for_beginners_zh.md) [<u>한국어</u>](./docs/original_docs/introduction_for_beginners_kr.md)).

-   [24/05/31] We supported Openai-series models as generator.

</details>

## Installation

```bash
pip install flashrag-dev --pre
```

or

```bash
git clone https://github.com/RUC-NLPIR/FlashRAG.git
cd FlashRAG
pip install -e .
```

### Optional Dependencies
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

### Faiss Installation
```bash
# CPU-only version
conda install -c pytorch faiss-cpu=1.8.0

# GPU(+CPU) version
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```

## Quick Start

### Corpus Construction

```jsonl
{"id": "0", "contents": "..."}
{"id": "1", "contents": "..."}
```

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

    # get attribute in dataset using `.`
    input_query = dataset.question
    ...
    # use `update_output` to save intermeidate data
    dataset.update_output("pred",pred_answer_list)
    dataset = self.evaluate(dataset, do_eval=do_eval)
    return dataset
```

### Just use components
## Components

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
With <strong>FlashRAG-UI</strong>, you can easily and quickly configure and experience the supported RAG methods through our meticulously designed visual interface, and evaluate these methods on benchmarks, making complex research work more efficient!

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

## Supporting Methods

### Key Results

We have implemented **23 works**, with the following consistent settings:

*   **Generator:** LLAMA3-8B-instruct with input length of 2048
*   **Retriever:** e5-base-v2 as embedding model, retrieve 5 docs per query
*   **Prompt:** A consistent default prompt (template can be found in the [<u>method details</u>](./docs/original_docs/baseline_details.md)).

For open-source methods, we implemented their processes using our framework. For methods where the author did not provide source code, we will try our best to follow the methods in the original paper for implementation.

For necessary settings and hyperparameters specific to some methods, we have documented them in the **specific settings** column. For more details, please consult our [<u>reproduce guidance</u>](./docs/original_docs/reproduce_experiment.md) and [<u>method details</u>](./docs/original_docs/baseline_details.md).

It's important to note that, to ensure consistency, we have utilized a uniform setting. However, this setting may differ from the original setting of the method, leading to variations in results compared to the original outcomes.

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
| [CoRAG](https://arxiv.org/abs/2503.21729) | Reasoning | 40.9 | 63.1 | 36.0 | 56.6 | 60.7 | 31.9 | 5