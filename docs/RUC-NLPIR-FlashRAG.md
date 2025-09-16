# ⚡️ FlashRAG: Unlock Efficient RAG Research with Python ⚡️

[![arXiv](https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv)](https://arxiv.org/abs/2405.13576)
[![HuggingFace Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Datasets-27b3b4.svg)](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/)
[![ModelScope Datasets](https://custom-icon-badges.demolab.com/badge/ModelScope%20Datasets-624aff?style=flat&logo=modelscope&logoColor=white)](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset)
[![DeepWiki Document](https://devin.ai/assets/deepwiki-badge.png)](https://deepwiki.com/RUC-NLPIR/FlashRAG)
[![License](https://img.shields.io/badge/LICENSE-MIT-green)](https://github.com/RUC-NLPIR/FlashRAG/blob/main/LICENSE)
[![Made with Python](https://img.shields.io/badge/made_with-Python-blue)](https://www.python.org/)

**FlashRAG is a powerful Python toolkit designed to accelerate your research in Retrieval-Augmented Generation (RAG).**  Effortlessly reproduce cutting-edge RAG models, build custom pipelines, and leverage a comprehensive suite of resources for faster, more effective experimentation.  [Explore the original repo](https://github.com/RUC-NLPIR/FlashRAG).

**Key Features:**

*   **Modular and Customizable:** Build complex RAG pipelines with flexible components for retrievers, rerankers, generators, and more.
*   **Extensive Benchmark Datasets:** Evaluate your models with 36 pre-processed RAG benchmark datasets.
*   **Pre-Implemented Advanced Algorithms:** Reproduce and experiment with **23 state-of-the-art RAG algorithms**, including **7 reasoning-based methods**.
*   **Reasoning-Based Methods:**  Now supporting **7 reasoning-based methods** that leverage reasoning capabilities for superior performance on complex tasks.
*   **Efficient Workflow:** Streamline your RAG pipeline with pre-processing scripts for corpus management and index building.
*   **Optimized Performance:** Accelerate LLM inference with tools like vLLM and FastChat, and leverage Faiss for efficient vector indexing.
*   **User-Friendly UI:**  Easily configure, experiment with, and evaluate RAG methods using the intuitive **FlashRAG-UI**.

**Key Updates**

*   **[Aug 2024]**  Added Support for Reasoning Pipeline and improved Hotpotqa with F1 scores close to 60.
*   **[Mar 2024]** Support for 23 state-of-the-art RAG algorithms, including 7 reasoning-based methods.
*   **[Feb 2024]** Multimodal RAG support with MLLMs and Multimodal retrievers.
*   **[Jan 2024]** FlashRAG-UI release.

**Jump to:**

*   [Installation](#wrench-installation)
*   [Quick Start](#rocket-quick-start)
*   [Features](#sparkles-features)
*   [Components](#gear-components)
*   [FlashRAG-UI](#art-flashrag-ui)
*   [Supported Methods](#robot-supporting-methods)
*   [Supporting Datasets](#notebook-supporting-datasets--document-corpus)
*   [FAQs](#raised_hands-additional-faqs)

<details>
<summary>Show more</summary>

## :wrench: Installation

FlashRAG is easily installed using pip:

```bash
pip install flashrag-dev --pre
```

Or clone the repository and install:

```bash
git clone https://github.com/RUC-NLPIR/FlashRAG.git
cd FlashRAG
pip install -e .
```

Install optional dependencies for enhanced functionality:

```bash
pip install flashrag-dev[full]
pip install vllm>=0.4.1  # For faster inference
pip install sentence-transformers
pip install pyserini # for BM25
conda install -c pytorch faiss-cpu=1.8.0 #For CPU only
conda install -c pytorch -c nvidia faiss-gpu=1.8.0 # For GPU(+CPU)
```

Note: Refer to the official Faiss documentation for compatibility details.

## :rocket: Quick Start

### Corpus Preparation

Create a `jsonl` file with each line containing a document in the format:

```jsonl
{"id": "0", "contents": "..."}
{"id": "1", "contents": "..."}
```

### Index Building

#### Dense Retrieval

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

#### Sparse Retrieval (BM25)

```bash
python -m flashrag.retriever.index_builder \
  --retrieval_method bm25 \
  --corpus_path indexes/sample_corpus.jsonl \
  --bm25_backend bm25s \
  --save_dir indexes/
```

### Ready-Made Pipeline

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

prompt_templete = PromptTemplate(
    config,
    system_prompt = "Answer the question based on the given document. Only give me the answer and do not output any other words.\nThe following are given documents.\n\n{reference}",
    user_prompt = "Question: {question}\nAnswer:"
)
pipeline = SequentialPipeline(
  my_config,
  prompt_template = prompt_templete
)

output_dataset = pipeline.run(test_data, do_eval=True)
```

### Build Your Own Pipeline

Customize your RAG workflow by inheriting `BasicPipeline` and implementing the `run` function.

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

## :sparkles: Features

*   **Extensive and Customizable Framework:** Includes essential RAG components for flexible pipeline assembly.
*   **Comprehensive Benchmark Datasets:** Access 36 pre-processed RAG datasets for robust model evaluation.
*   **Pre-implemented Advanced RAG Algorithms:** Features **23 advanced RAG algorithms** with reported results.
*   **🚀 Reasoning-based Methods**:  Support for **7 reasoning-based methods** that combine reasoning ability with retrieval.
*   **Efficient Preprocessing:** Simplify RAG workflow preparation with corpus and index building scripts.
*   **Optimized Execution:**  Enhance performance with vLLM, FastChat, and Faiss.
*   **User-Friendly UI:** An easy-to-use UI to configure and experiment with implemented baselines.

## :mag_right: Roadmap

*   [x] Support OpenAI models
*   [x] Provide instructions for each component
*   [x] Integrate sentence Transformers
*   [x] Support multimodal RAG
*   [x] Support reasoning-based methods
*   [ ] Inlcude more RAG approaches
*   [ ] Enhance code adaptability and readability
*   [ ] Add support for api-based retriever (vllm server)

## :page_with_curl: Changelog
See the [Changelog](https://github.com/RUC-NLPIR/FlashRAG/blob/main/README.md) for detailed update information.

## :gear: Components

**RAG-Components**

| Type        | Module            | Description                                                                                                                                            |
| ----------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Judger      | SKR Judger        | Judging whether to retrieve using <a href="https://aclanthology.org/2023.findings-emnlp.691.pdf">SKR</a> method                                        |
| Retriever   | Dense Retriever   | Bi-encoder models such as dpr, bge, e5, using faiss for search                                                                                         |
| Retriever   | BM25 Retriever    | Sparse retrieval method based on Lucene                                                                                                                |
| Retriever   | Bi-Encoder Reranker | Calculate matching score using bi-Encoder                                                                                                              |
| Retriever   | Cross-Encoder Reranker | Calculate matching score using cross-encoder                                                                                                              |
| Refiner     | Extractive Refiner| Refine input by extracting important context                                                                                                          |
| Refiner     | Abstractive Refiner| Refine input through seq2seq model                                                                                                          |
| Refiner     | LLMLingua Refiner | <a href="https://aclanthology.org/2023.emnlp-main.825/">LLMLingua-series</a> prompt compressor                                                           |
| Refiner     | SelectiveContext Refiner | <a href="https://arxiv.org/abs/2310.06201">Selective-Context</a> prompt compressor                                                        |
| Refiner     | KG Refiner        | Use <a hred='https://arxiv.org/abs/2406.11460'>Trace method to construct a knowledge graph                                                         |
| Generator   | Encoder-Decoder   | Encoder-Decoder model, supporting <a href="https://arxiv.org/abs/2007.01282">Fusion-in-Decoder (FiD)</a>                                                 |
| Generator   | Decoder-only      | Native transformers implementation                                                                                                                     |
| Generator   | FastChat          | Accelerate with <a href="https://github.com/lm-sys/FastChat">FastChat</a>                                                                               |
| Generator   | vllm              | Accelerate with <a href="https://github.com/vllm-project/vllm">vllm</a>                                                                                   |

**Pipelines**

| Type        | Module                | Description                                                                                                                                                           |
| ----------- | --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Sequential  | Sequential Pipeline     | Linear execution of query, supporting refiner, reranker                                                                                                              |
| Conditional | Conditional Pipeline    | Distinct execution paths for various query types                                                                                                                      |
| Branching   | REPLUG Pipeline       | Generate answer by integrating probabilities in multiple generation paths                                                                                             |
| Branching   | SuRe Pipeline       | Ranking and merging generated results based on each document                                                                                             |
| Loop        | Iterative Pipeline    | Alternating retrieval and generation                                                                                                                                |
| Loop        | Self-Ask Pipeline     | Decompose complex problems into subproblems using <a href="https://arxiv.org/abs/2210.03350">self-ask</a>                                                            |
| Loop        | Self-RAG Pipeline     | Adaptive retrieval, critique, and generation                                                                                                                             |
| Loop        | FLARE Pipeline        | Dynamic retrieval during the generation process                                                                                                                         |
| Loop        | IRCoT Pipeline        | Integrate retrieval process with CoT                                                                                                                                    |
| Loop        | Reasoning Pipeline    | Reasoning with retrieval                                                                                                                                            |

## :art: FlashRAG-UI

Easily configure and experience RAG methods through an intuitive visual interface:

**Features:**

*   **One-Click Configuration Loading**
    *   Load parameters and configuration files.
    *   Preview interface for parameter settings.
    *   Save configurations.
*   **Quick Method Experience**
    *   Load corpora and index files.
    *   Supports loading and switching different components and hyperparameters.
*   **Efficient Benchmark Reproduction**
    *   Reproduce the built-in baseline methods and benchmarks.

Experience FlashRAG-UI:

```bash
cd webui
python interface.py
```

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

## :robot: Supporting Methods

*   **23 Implemented Methods**

    *   Generator: LLAMA3-8B-instruct (2048 input length)
    *   Retriever: e5-base-v2 (top 5 docs)
    *   Prompt: Consistent default prompt (see [method details](./docs/original_docs/baseline_details.md))

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
    | [RQRAG](https://arxiv.org/abs/2404.00610) | Loop        | 32.6    | 52.5          | 33.5          | 35.8       | 46.4       | 26.2      |  Use trained rqrag-llama2-7B                                               |

*   **🚀 Reasoning-based Methods (NEW!)**

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

Access 36 pre-processed datasets:

| Task                      | Dataset Name    | Knowledge Source | # Train   | # Dev   | # Test |
| ------------------------- | --------------- | ---------------- | --------- | ------- | ------ |
| QA                        | NQ              | wiki             | 79,168    | 8,757   | 3,610  |
| QA                        | TriviaQA        | wiki & web       | 78,785    | 8,837   | 11,313 |
| QA                        | PopQA           | wiki             | /         | /       | 14,267 |
| ...                       | ...             | ...              | ...       | ...     | ...    |

All datasets are available at [<u>Huggingface datasets</u>](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets).

### Document Corpus

Supports `jsonl` format.

```jsonl
{"id":"0", "contents": "..."}
{"id":"1", "contents": "..."}
```

*   Wikipedia & MS MARCO are commonly used.

*   Processed Wikipedia:  [<u>comprehensive script</u>](./docs/original_docs/process-wiki.md).
*   MS MARCO:  [<u>hosting link</u>](https://huggingface.co/datasets/Tevatron/msmarco-passage-corpus).

### Index

Preprocessed index available [here](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/file/view/master?id=47985&status=2&fileName=retrieval_corpus%252Fwiki18_100w_e5_index.zip).

## :raised_hands: Additional FAQs

*   [How to set experimental parameters?](./docs/original_docs/configuration.md)
*   [How to build your own corpus?](./docs/original_docs/process-wiki.md)
*   [How to index your own corpus?](./docs/original_docs/building-index.md)
*   [How to reproduce supporting methods?](./docs/original_docs/reproduce_experiment.md)

## :bookmark: License

This project is licensed under the [<u>MIT License</u>](./LICENSE).
</details>