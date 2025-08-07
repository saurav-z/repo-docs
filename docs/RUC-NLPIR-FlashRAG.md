# FlashRAG: Your Toolkit for Efficient RAG Research

> Unleash the power of Retrieval-Augmented Generation (RAG) with FlashRAG, a Python toolkit designed to accelerate your research and development in the RAG domain.  [Explore the FlashRAG repository on GitHub](https://github.com/RUC-NLPIR/FlashRAG).

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

FlashRAG empowers researchers to easily reproduce state-of-the-art (SOTA) RAG models and build their own customized RAG pipelines. This toolkit provides a comprehensive framework, pre-processed datasets, and pre-implemented algorithms for efficient RAG research.

<p align="center">
<img src="asset/framework.jpg">
</p>

## Key Features

*   **Extensive and Customizable Framework:** Assemble complex pipelines with modular components like retrievers, rerankers, generators, and compressors.
*   **Comprehensive Benchmark Datasets:** Evaluate RAG models with 36 pre-processed benchmark datasets.
*   **Pre-implemented Advanced RAG Algorithms:** Reproduce results quickly with 23 advanced RAG algorithms.
*   **🚀 Reasoning-Based Methods:**  **NEW!**  Explore cutting-edge research with support for 7 reasoning-based methods, combining reasoning and retrieval.
*   **Efficient Preprocessing:** Simplify your RAG workflow with streamlined corpus and index preparation.
*   **Optimized Execution:** Accelerate inference using vLLM, FastChat for LLM inference, and Faiss for vector indexing.
*   **Easy-to-Use UI:** Easily configure, experiment with, and evaluate RAG baselines using a user-friendly visual interface.

## :wrench: Installation

Get started by installing FlashRAG using pip:

```bash
pip install flashrag-dev --pre
```

Or clone the repository and install:

```bash
git clone https://github.com/RUC-NLPIR/FlashRAG.git
cd FlashRAG
pip install -e .
```

For optional dependencies like vLLM, sentence-transformers, and pyserini, use:

```bash
pip install flashrag-dev[full]
```

Follow the instructions in the original README for specific installation instructions regarding `faiss`.

## :rocket: Quick Start

FlashRAG provides ready-to-use components for building RAG pipelines.

### Corpus Construction

1.  Prepare your corpus as a `jsonl` file.

    ```jsonl
    {"id": "0", "contents": "..."}
    {"id": "1", "contents": "..."}
    ```

2.  See [Processing Wikipedia](./docs/original_docs/process-wiki.md) to convert your data.

### Index Construction

Choose your retrieval method and build an index using the provided scripts.

#### Dense Retrieval Methods

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

*   Refer to the original documentation for parameter details.

#### Sparse Retrieval Methods (BM25)

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

### Use Ready-Made Pipeline

Load a pipeline and configure it with a yaml or dictionary config

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

# Customize prompts
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

Inherit `BasicPipeline` and create your custom RAG process.

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

## :gear: Components

FlashRAG provides the flexibility to create custom pipelines:

#### RAG Components

| Type        | Module            | Description                                                        |
| ----------- | ----------------- | ------------------------------------------------------------------ |
| Judger      | SKR Judger        | Judge whether to retrieve using <a href="https://aclanthology.org/2023.findings-emnlp.691.pdf">SKR</a> method           |
| Retriever   | Dense Retriever   | Bi-encoder models like dpr, bge, e5, using faiss.                  |
|             | BM25 Retriever    | Sparse retrieval method based on Lucene.                           |
|             | Bi-Encoder Reranker| Calculate matching score using bi-encoder          |
|             | Cross-Encoder Reranker| Calculate matching score using cross-encoder         |
| Refiner     | Extractive Refiner| Refine input by extracting important context            |
|             | Abstractive Refiner| Refine input through seq2seq model           |
|             | LLMLingua Refiner   | <a href="https://aclanthology.org/2023.emnlp-main.825/">LLMLingua-series</a> prompt compressor |
|             | SelectiveContext Refiner   | <a href="https://arxiv.org/abs/2310.06201">Selective-Context</a> prompt compressor |
|             | KG Refiner        | Use <a hred='https://arxiv.org/abs/2406.11460'>Trace method to construct a knowledge graph |
| Generator   | Encoder-Decoder Generator | Encoder-Decoder model, supporting <a href="https://arxiv.org/abs/2007.01282">Fusion-in-Decoder (FiD)</a> |
|             | Decoder-only Generator | Native transformers implementation                 |
|             | FastChat Generator  | Accelerate with <a href="https://github.com/lm-sys/FastChat">FastChat</a> |
|             | vllm Generator      | Accelerate with <a href="https://github.com/vllm-project/vllm">vllm</a> |

#### Pipelines

| Type         | Module             | Description                                                                      |
| ------------ | ------------------ | -------------------------------------------------------------------------------- |
| Sequential   | Sequential Pipeline| Linear execution, supporting refiner, reranker                                 |
| Conditional  | Conditional Pipeline| Different paths for different query types                                        |
| Branching    | REPLUG Pipeline    | Integrate probabilities in multiple generation paths.                           |
| Branching    | SuRe Pipeline    | Ranking and merging generated results based on each document.                           |
| Loop         | Iterative Pipeline | Alternating retrieval and generation                                             |
|              | Self-Ask Pipeline  | Decompose complex problems using <a href="https://arxiv.org/abs/2210.03350">self-ask</a>      |
|              | Self-RAG Pipeline  | Adaptive retrieval, critique, and generation                                      |
|              | FLARE Pipeline     | Dynamic retrieval during generation                                               |
|              | IRCoT Pipeline     | Integrate retrieval with CoT                                                      |
|              | Reasoning Pipeline   | Reasoning with retrieval         |

## :art: FlashRAG-UI

FlashRAG-UI is a user-friendly interface to easily and quickly configure and experience RAG methods.

### Features

*   One-Click configuration
*   Quick Method Experience
*   Efficient Benchmark Reproduction

```bash
cd webui
python interface.py
```

## :robot: Supporting Methods

FlashRAG implements **23 works** with the following default settings:

-   **Generator:** LLAMA3-8B-instruct with input length of 2048
-   **Retriever:** e5-base-v2 as embedding model, retrieve 5 docs per query
-   **Prompt:** A consistent default prompt (see [<u>method details</u>](./docs/original_docs/baseline_details.md)).

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

FlashRAG supports a wide range of datasets for RAG research, available on [Huggingface datasets](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets).

| Task                      | Dataset Name    | Knowledge Source | # Train   | # Dev   | # Test |
| ------------------------- | --------------- | ---------------- | --------- | ------- | ------ |
| QA                        | NQ              | wiki             | 79,168    | 8,757   | 3,610  |
| QA                        | TriviaQA        | wiki & web       | 78,785    | 8,837   | 11,313 |
| QA                        | PopQA           | wiki             | /         | /       | 14,267 |
| ...                       | ...             | ...              | ...       | ...     | ...    |

### Document Corpus

Use JSONL format:

```jsonl
{"id":"0", "contents": "..."}
{"id":"1", "contents": "..."}
```

For Wikipedia, refer to  [<u>comprehensive script</u>](./docs/original_docs/process-wiki.md).

### Index

Download the preprocessed index from [FlashRAG\_Dataset/retrieval\_corpus/wiki18\_100w\_e5\_index.zip](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/file/view/master?id=47985&status=2&fileName=retrieval_corpus%252Fwiki18_100w_e5_index.zip).

## :lollipop: Awesome Work using FlashRAG

- [R1-Searcher](https://github.com/SsmallSong/R1-Searcher)
- [ReSearch](https://github.com/Agent-RL/ReSearch)
- [AutoCoA](https://github.com/ADaM-BJTU/AutoCoA)

## :raised_hands: Additional FAQs

-   [How should I set different experimental parameters?](./docs/original_docs/configuration.md)
-   [How to build my own corpus, such as a specific segmented Wikipedia?](./docs/original_docs/process-wiki.md)
-   [How to index my own corpus?](./docs/original_docs/building-index.md)
-   [How to reproduce supporting methods?](./docs/original_docs/reproduce_experiment.md)

## :bookmark: License

FlashRAG is licensed under the [<u>MIT License</u>](./LICENSE).

## :star2: Citation

If you use FlashRAG in your research, please cite our paper:

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