# FlashRAG: Your Toolkit for Efficient RAG Research

**Unlock the power of Retrieval-Augmented Generation (RAG) with FlashRAG, a Python toolkit designed for efficient RAG research and development. [Explore the original repository on GitHub](https://github.com/RUC-NLPIR/FlashRAG)!**

<div align="center">
<a href="https://arxiv.org/abs/2405.13576" target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv></a>
<a href="https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/" target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Datasets-27b3b4.svg></a>
<a href="https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset" target="_blank"><img src=https://custom-icon-badges.demolab.com/badge/ModelScope%20Datasets-624aff?style=flat&logo=modelscope&logoColor=white></a>
<a href="https://deepwiki.com/RUC-NLPIR/FlashRAG"><img src="https://devin.ai/assets/deepwiki-badge.png" alt="DeepWiki Document" height="20"/></a>
<a href="https://github.com/RUC-NLPIR/FlashRAG/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green"></a>
<a><img alt="Static Badge" src="https://img.shields.io/badge/made_with-Python-blue"></a>
</div>

## Key Features:

*   **Flexible Framework:** Assemble complex RAG pipelines with modular components for retrievers, rerankers, generators, and more.
*   **Comprehensive Datasets:** Access 36 pre-processed benchmark RAG datasets to evaluate and validate your models.
*   **Advanced RAG Algorithms:** Explore **23 pre-implemented state-of-the-art RAG algorithms**, including proven performance baselines.
*   **Reasoning-Driven Performance:** *NEW!* Benefit from **7 reasoning-based methods** that enhance retrieval with reasoning abilities, excelling on complex tasks.
*   **Streamlined Workflow:** Simplify RAG preparation with efficient tools for corpus processing, index building, and document retrieval.
*   **Optimized Execution:** Leverage vLLM, FastChat, and Faiss for accelerated LLM inference and vector index management.
*   **User-Friendly UI:** Utilize an easy-to-use UI to configure, experiment with, and evaluate RAG methods quickly.

## Table of Contents

*   [Installation](#wrench-installation)
*   [Features](#sparkles-features)
*   [Quick Start](#rocket-quick-start)
*   [Components](#gear-components)
*   [FlashRAG-UI](#art-flashrag-ui)
*   [Supporting Methods](#robot-supporting-methods)
*   [Supporting Datasets & Document Corpus](#notebook-supporting-datasets--document-corpus)
*   [FAQs](#raised_hands-additional-faqs)
*   [License](#bookmark-license)
*   [Citation](#star2-citation)

## :wrench: Installation

FlashRAG can be easily installed using pip:

```bash
pip install flashrag-dev --pre
```

Alternatively, clone the repository and install:

```bash
git clone https://github.com/RUC-NLPIR/FlashRAG.git
cd FlashRAG
pip install -e .
```
*Optional Dependencies:*

```bash
pip install flashrag-dev[full]
pip install vllm>=0.4.1
pip install sentence-transformers
pip install pyserini
```

Use conda for `faiss`:

```bash
conda install -c pytorch faiss-cpu=1.8.0
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```
## :rocket: Quick Start

Get started by constructing your corpus, building an index, and utilizing pre-built pipelines or components.

### Corpus Construction
Save your corpus as a `jsonl` file.

```jsonl
{"id": "0", "contents": "..."}
{"id": "1", "contents": "..."}
```

### Index Construction

*For Dense Retrieval Methods:*

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

*For Sparse Retrieval Methods (BM25):*

```bash
python -m flashrag.retriever.index_builder \
  --retrieval_method bm25 \
  --corpus_path indexes/sample_corpus.jsonl \
  --bm25_backend bm25s \
  --save_dir indexes/ 
```
### For Sparse Neural Retrieval Methods (SPLADE)

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
First, load the process's config.
```python
from flashrag.config import Config

# hybrid load configs
config_dict = {'data_dir': 'dataset/'}
my_config = Config(
    config_file_path = 'my_config.yaml',
    config_dict = config_dict
```
Load the corresponding dataset and initialize the pipeline
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

## :gear: Components

FlashRAG offers various RAG components including retrievers, generators, and refiners. It provides pre-built pipelines and the flexibility to create custom pipelines using these components.

#### RAG-Components

| Type        | Module              | Description                                                                                                                                             |
| ----------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Judger      | SKR Judger          | Judging whether to retrieve using <a href="https://aclanthology.org/2023.findings-emnlp.691.pdf">SKR</a> method                               |
| Retriever   | Dense Retriever     | Bi-encoder models (dpr, bge, e5, using faiss)                                                                                                       |
|             | BM25 Retriever      | Sparse retrieval method based on Lucene                                                                                                                   |
|             | Bi-Encoder Reranker | Calculate matching score using bi-Encoder                                                                                                                 |
|             | Cross-Encoder Reranker | Calculate matching score using cross-encoder                                                                                                                 |
| Refiner     | Extractive Refiner  | Refine input by extracting important context                                                                                                         |
|             | Abstractive Refiner | Refine input through seq2seq model                                                                                                                      |
|             | LLMLingua Refiner   | <a href="https://aclanthology.org/2023.emnlp-main.825/">LLMLingua-series</a> prompt compressor                                                            |
|             | SelectiveContext Refiner   | <a href="https://arxiv.org/abs/2310.06201">Selective-Context</a> prompt compressor                                                            |
|             | KG Refiner   | Use <a hred='https://arxiv.org/abs/2406.11460'>Trace method to construct a knowledge graph</td>
| Generator   | Encoder-Decoder Generator | Encoder-Decoder model, supporting <a href="https://arxiv.org/abs/2007.01282">Fusion-in-Decoder (FiD)</a>                                              |
|             | Decoder-only Generator  | Native transformers implementation                                                                                                                     |
|             | FastChat Generator    | Accelerate with <a href="https://github.com/lm-sys/FastChat">FastChat</a>                                                                                 |
|             | vllm Generator        | Accelerate with <a href="https://github.com/vllm-project/vllm">vllm</a>                                                                                     |

#### Pipelines

| Type        | Module              | Description                                                           |
| ----------- | ------------------- | --------------------------------------------------------------------- |
| Sequential  | Sequential Pipeline | Linear execution of RAG process                                       |
| Conditional | Conditional Pipeline| Different paths based on query type                                  |
| Branching   | REPLUG Pipeline     | Integrate probabilities in multiple generation paths                  |
| Branching   | SuRe Pipeline     | Ranking and merging generated results based on each document                  |
| Loop        | Iterative Pipeline  | Alternating retrieval and generation                                |
| Loop        | Self-Ask Pipeline   | Decompose complex problems into subproblems using self-ask            |
| Loop        | Self-RAG Pipeline   | Adaptive retrieval, critique, and generation                        |
| Loop        | FLARE Pipeline      | Dynamic retrieval during the generation process                       |
| Loop        | IRCoT Pipeline      | Integrate retrieval process with CoT                                   |
| Loop        | Reasoning Pipeline  | Reasoning with retrieval                                              |

## :art: FlashRAG-UI

**Enhance your RAG research with FlashRAG-UI, a user-friendly interface for easy method configuration, experimentation, and evaluation!**

### Features:

-   **Quick Configuration:** Load and save configurations with ease.
-   **Efficient Method Exploration:** Rapidly test methods with your data.
-   **Simplified Benchmark Reproduction:** Replicate baseline methods and explore benchmarks.

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

*   **Datasets:** 36 pre-processed datasets for RAG research, available on Hugging Face: [Huggingface datasets](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets).
*   **Document Corpus:** Supports JSONL format. Includes Wikipedia and MS MARCO.
*   **Preprocessed Index:**  Access a preprocessed index built with e5-base-v2 on the wiki18_100w dataset, available in the ModelScope dataset page: [FlashRAG_Dataset/retrieval_corpus/wiki18_100w_e5_index.zip](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/file/view/master?id=47985&status=2&fileName=retrieval_corpus%252Fwiki18_100w_e5_index.zip).

## :raised_hands: Additional FAQs

*   [How to set experimental parameters?](./docs/original_docs/configuration.md)
*   [How to build my own corpus?](./docs/original_docs/process-wiki.md)
*   [How to index my own corpus?](./docs/original_docs/building-index.md)
*   [How to reproduce supporting methods?](./docs/original_docs/reproduce_experiment.md)

## :bookmark: License

FlashRAG is licensed under the [<u>MIT License</u>](./LICENSE).

## :star2: Citation

If you find FlashRAG helpful, please cite our paper:

```BibTex
@inproceedings{FlashRAG,
  author       = {Jiajie Jin and
                  Yutao Zhu and
                  Zhicheng Dou and
                  Guanting Dong and
                  Xinyu Yang and
                  Chenghao Zhang and
                  Tong Zhao and
                  Zhao Yang and
                  Ji{-}Rong Wen},
  editor       = {Guodong Long and
                  Michale Blumestein and
                  Yi Chang and
                  Liane Lewin{-}Eytan and
                  Zi Helen Huang and
                  Elad Yom{-}Tov},
  title        = {FlashRAG: {A} Modular Toolkit for Efficient Retrieval-Augmented Generation
                  Research},
  booktitle    = {Companion Proceedings of the {ACM} on Web Conference 2025, {WWW} 2025,
                  Sydney, NSW, Australia, 28 April 2025 - 2 May 2025},
  pages        = {737--740},
  publisher    = {{ACM}},
  year         = {2025},
  url          = {https://doi.org/10.1145/3701716.3715313},
  doi          = {10.1145/3701716.3715313}
}