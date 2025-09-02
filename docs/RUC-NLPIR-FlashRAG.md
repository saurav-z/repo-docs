# FlashRAG: Your Toolkit for Efficient RAG Research

<!--  Div alignment centered for the main title -->
<div align="center">
  ⚡️ **FlashRAG empowers you to build, experiment, and advance Retrieval-Augmented Generation (RAG) models.**
</div>

<div align="center">
  <!-- Badges for quick reference -->
  <a href="https://arxiv.org/abs/2405.13576" target="_blank"><img src="https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Datasets-27b3b4.svg" alt="Hugging Face Datasets"></a>
  <a href="https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset" target="_blank"><img src="https://custom-icon-badges.demolab.com/badge/ModelScope%20Datasets-624aff?style=flat&logo=modelscope&logoColor=white" alt="ModelScope Datasets"></a>
  <a href="https://deepwiki.com/RUC-NLPIR/FlashRAG"><img src="https://devin.ai/assets/deepwiki-badge.png" alt="DeepWiki Document" height="20"></a>
  <a href="https://github.com/RUC-NLPIR/FlashRAG/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green"></a>
  <a><img alt="Static Badge" src="https://img.shields.io/badge/made_with-Python-blue"></a>
</div>

<div align="center">
  <!-- Navigation -->
  <p>
    <a href="#wrench-installation">Installation</a> |
    <a href="#sparkles-features">Features</a> |
    <a href="#rocket-quick-start">Quick Start</a> |
    <a href="#gear-components">Components</a> |
    <a href="#art-flashrag-ui">FlashRAG-UI</a> |
    <a href="#robot-supporting-methods">Supporting Methods</a> |
    <a href="#notebook-supporting-datasets--document-corpus">Supporting Datasets</a> |
    <a href="#raised_hands-additional-faqs">FAQs</a>
  </p>
</div>

FlashRAG is a powerful Python toolkit designed to accelerate Retrieval-Augmented Generation (RAG) research and development.  It simplifies the process of reproducing state-of-the-art (SOTA) RAG models and allows for the seamless implementation of custom RAG pipelines. FlashRAG provides:

*   **36 pre-processed benchmark RAG datasets.**
*   **23 pre-implemented SOTA RAG algorithms.**
*   **7 Reasoning-based methods** combining reasoning with retrieval.
*   **User-friendly UI**.

<p align="center">
<img src="asset/framework.jpg" alt="FlashRAG Framework">
</p>

[**Visit the original repository for full details.**](https://github.com/RUC-NLPIR/FlashRAG)

## Key Features

*   **Flexible and Extensible Framework:** Assemble complex RAG pipelines with ease using essential components like retrievers, rerankers, generators, and compressors.
*   **Comprehensive Benchmark Datasets:** Evaluate your models using 36 pre-processed RAG datasets, simplifying the validation of your RAG models.
*   **Pre-implemented SOTA Algorithms:** Quickly reproduce and experiment with **23 advanced RAG algorithms**, including **7 reasoning-based methods**, to achieve superior performance on complex tasks.
*   **Reasoning-Based Methods:** Explore novel reasoning-based methods that combine reasoning and retrieval for enhanced performance on multi-hop tasks.
*   **Efficient Preprocessing:** Simplify the RAG workflow with pre-built scripts for corpus processing, index building, and document retrieval.
*   **Optimized Performance:**  Leverage tools like vLLM and FastChat for faster LLM inference and Faiss for efficient vector index management.
*   **User-Friendly UI:**  Easily configure, experiment with, and evaluate RAG baselines via our intuitive FlashRAG-UI.

## Installation <a name="wrench-installation"></a>

[![PyPI - Version](https://img.shields.io/pypi/v/flashrag-dev)](https://pypi.org/project/flashrag-dev/)
[![PyPI - Downloads](https://img.shields.io/pypi/dw/flashrag-dev)](https://pypi.org/project/flashrag-dev/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/flashrag-dev)](https://pypi.org/project/flashrag-dev/)

Install FlashRAG using `pip`:

```bash
pip install flashrag-dev --pre
```

Or, install from source:

```bash
git clone https://github.com/RUC-NLPIR/FlashRAG.git
cd FlashRAG
pip install -e .
```

Install optional dependencies for extra functionality:

```bash
pip install flashrag-dev[full]  # all extras
pip install vllm>=0.4.1           # vLLM for speed
pip install sentence-transformers # sentence-transformers
pip install pyserini             # for bm25
```

For `faiss` installation (CPU and GPU versions):

```bash
conda install -c pytorch faiss-cpu=1.8.0       # CPU-only
conda install -c pytorch -c nvidia faiss-gpu=1.8.0  # GPU (requires CUDA)
```

*   **Note:** Specific `faiss` versions might be required based on your system.

## Quick Start <a name="rocket-quick-start"></a>

### 1. Corpus Construction

Prepare your corpus as a `jsonl` file:

```jsonl
{"id": "0", "contents": "..."}
{"id": "1", "contents": "..."}
```

Refer to the documentation for [Processing Wikipedia](./docs/original_docs/process-wiki.md).

### 2. Index Construction

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

*   Adjust `--pooling_method` based on your model (`mean`, `pooler`, or `cls`).
*   Use `--instruction` for models like E5 and BGE that need query instructions.

For models supporting `sentence-transformers`:

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

#### Sparse Retrieval Methods (BM25)

##### BM25s

```bash
python -m flashrag.retriever.index_builder \
  --retrieval_method bm25 \
  --corpus_path indexes/sample_corpus.jsonl \
  --bm25_backend bm25s \
  --save_dir indexes/
```

##### Pyserini

```bash
python -m flashrag.retriever.index_builder \
  --retrieval_method bm25 \
  --corpus_path indexes/sample_corpus.jsonl \
  --bm25_backend pyserini \
  --save_dir indexes/
```

### 3. Using the Ready-Made Pipeline

1.  Load config files:

    ```python
    from flashrag.config import Config

    config_dict = {'data_dir': 'dataset/'}
    my_config = Config(
        config_file_path = 'my_config.yaml',
        config_dict = config_dict
    )
    ```

    See the [configuration guidance](./docs/original_docs/configuration.md).
2.  Load the dataset and initialize the pipeline:

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
3.  Define your input prompt:

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
4.  Run the pipeline:

    ```python
    output_dataset = pipeline.run(test_data, do_eval=True)
    ```

### 4. Build your own pipeline!

If you need more complex logic, create a custom pipeline by inheriting `BasicPipeline`.

## Components <a name="gear-components"></a>

FlashRAG provides a set of modular components for building RAG systems.

### RAG Components

| Type        | Module            | Description                                                                            |
| ----------- | ----------------- | -------------------------------------------------------------------------------------- |
| Judger      | SKR Judger        | Uses SKR method  ([SKR paper](https://aclanthology.org/2023.findings-emnlp.691.pdf)). |
| Retriever   | Dense Retriever   | Bi-encoder models (dpr, bge, e5) with Faiss search.                                    |
|             | BM25 Retriever    | Sparse retrieval based on Lucene.                                                      |
|             | Bi-Encoder Reranker | Calculates matching scores using bi-encoders.                                       |
|             | Cross-Encoder Reranker | Calculates matching scores using cross-encoders.                                 |
| Refiner     | Extractive Refiner | Extract key content to refine input                                                 |
|             | Abstractive Refiner | Refine input using seq2seq models.                                                      |
|             | LLMLingua Refiner  | prompt compressor ([LLMLingua paper](https://aclanthology.org/2023.emnlp-main.825/)).|
|             | SelectiveContext Refiner | prompt compressor ([SelectiveContext paper](https://arxiv.org/abs/2310.06201)).   |
|             | KG Refiner        |  Using Trace method for knowledge graph ([Trace paper](https://arxiv.org/abs/2406.11460)) |
| Generator   | Encoder-Decoder Generator | Encoder-Decoder model, supporting [FiD](https://arxiv.org/abs/2007.01282).          |
|             | Decoder-only Generator | Native transformer implementation.                                                |
|             | FastChat Generator | Accelerated generation with [FastChat](https://github.com/lm-sys/FastChat).           |
|             | vllm Generator    | Accelerated generation with [vllm](https://github.com/vllm-project/vllm).               |

### Pipelines

RAG pipelines are categorized based on inference paths:

*   **Sequential**: Linear execution (Query -> (Pre-retrieval) -> Retriever -> (Post-retrieval) -> Generator)
*   **Conditional**: Multiple paths based on query types.
*   **Branching**: Parallel execution, merging results.
*   **Loop**: Iterative retrieval and generation.

| Type         | Module            | Description                                                                                                                               |
| ------------ | ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Sequential   | Sequential Pipeline | Linear query execution, supporting refiner, and reranker.                                                                               |
| Conditional  | Conditional Pipeline | Uses a judger module for different query types.                                                                                            |
| Branching    | REPLUG Pipeline   | Integrates probabilities from multiple generation paths ([REPLUG paper](https://arxiv.org/abs/2301.12652)).                                   |
|              | SuRe Pipeline     | Ranks and merges generated results per document ([SuRe paper](https://arxiv.org/abs/2404.13081)).                                         |
| Loop         | Iterative Pipeline | Alternates retrieval and generation.                                                                                                      |
|              | Self-Ask Pipeline  | Decomposes complex problems using [self-ask](https://arxiv.org/abs/2210.03350).                                                               |
|              | Self-RAG Pipeline  | Adaptive retrieval, critique, and generation ([Self-RAG paper](https://arxiv.org/abs/2310.11511)).                                       |
|              | FLARE Pipeline     | Dynamic retrieval during generation ([FLARE paper](https://arxiv.org/abs/2305.06983)).                                                    |
|              | IRCoT Pipeline     | Integrates retrieval with CoT ([IRCoT paper](https://aclanthology.org/2023.acl-long.557.pdf)).                                              |
|              | Reasoning Pipeline | Utilizes reasoning with retrieval ([Reasoning paper]).                                         |

## FlashRAG-UI <a name="art-flashrag-ui"></a>

**FlashRAG-UI** offers a user-friendly, visual interface to configure, experiment, and evaluate RAG methods effectively.

### Features

*   **One-Click Configuration Loading:** Load parameters with simple clicks and inputs.
*   **Quick Method Experience:** Load corpora, experiment with methods, and switch components rapidly.
*   **Efficient Benchmark Reproduction:**  Reproduce baseline methods efficiently.

```bash
cd webui
python interface.py
```

<details>
<summary>Show More UI Images</summary>
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

## Supporting Methods <a name="robot-supporting-methods"></a>

We have implemented **23 works**, using a consistent baseline:

*   **Generator:** LLAMA3-8B-instruct (2048 input length)
*   **Retriever:** e5-base-v2 (5 docs retrieved)
*   **Prompt:** Default prompt.
*   See [method details](./docs/original_docs/baseline_details.md) for specifics.

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

### 🚀 Reasoning-based Methods (NEW!)

| Method                                                                                    | Type        | NQ (EM) | TriviaQA (EM) | PopQA (EM) | Hotpotqa (F1) | 2Wiki (F1) |  Musique (F1) | Bamboogle (F1) | Specific setting                             |
| ----------------------------------------------------------------------------------------- | ----------- | ------- | ------- | ------------- | ------------- | ---------- | ---------- | --------- | ----------------------------------------------- |
| [Search-R1](https://arxiv.org/abs/2503.09516) | Reasoning | 45.2 | 62.2 | 49.2 | 54.5 | 42.6 | 29.2 |  59.9 | SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo |
| [R1-Searcher](https://arxiv.org/pdf/2503.05592) | Reasoning | 36.9 | 61.6 | 42.0 | 49.0 | 49.1 | 24.7 | 57.7 | Qwen-2.5-7B-base-RAG-RL |
| [O2-Searcher](https://arxiv.org/pdf/2505.16582) | Reasoning | 41.4 | 51.4 | 46.8 | 43.4 | 48.6 | 19.0 | 47.6 | O2-Searcher-Qwen2.5-3B-GRPO |
| [AutoRefine](https://www.arxiv.org/pdf/2505.11277) | Reasoning | 43.8 | 59.8 | 32.4 | 54.0 | 50.3 | 23.6 | 46.6 | AutoRefine-Qwen2.5-3B-Base |
| [ReaRAG](https://arxiv.org/abs/2503.21729) | Reasoning | 26.3 | 51.8 | 24.6 | 42.9 | 41.6 | 21.2 | 41.9 | ReaRAG-9B |
| [CoRAG](https://arxiv.org/abs/2503.21729) | Reasoning | 40.9 | 63.1 | 36.0 | 56.6 | 60.7 | 31.9 | 54.1 | CoRAG-Llama3.1-8B-MultihopQA |
| [SimpleDeepSearcher](https://arxiv.org/pdf/2505.16834) | Reasoning | 36.1 | 61.6 | 42.0 | 49.0 | 49.1 | 24.7 | 57.7 | Qwen-7B-SimpleDeepSearcher |

## Supporting Datasets & Document Corpus <a name="notebook-supporting-datasets--document-corpus"></a>

### Datasets

FlashRAG includes 36 pre-processed RAG datasets ([Huggingface Datasets](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets)):

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
| multiple-choice           | MMLU            | -                | 99,842    | 1,531   | 14,042 |
| multiple-choice           | TruthfulQA      | wiki             | /         | 817     | /      |
| multiple-choice           | HellaSWAG       | ActivityNet      | 39,905    | 10,042  | /      |
| multiple-choice           | ARC             | -                | 3,370     | 869     | 3,548  |
| multiple-choice           | OpenBookQA      | -                | 4,957     | 500     | 500    |
| multiple-choice           | QuaRTz          | -                | 2696      | 384     | 784    |
| Fact Verification         | FEVER           | wiki             | 104,966   | 10,444  | /      |
| Dialog Generation         | WOW             | wiki             | 63,734    | 3,054   | /      |
| Entity Linking            | AIDA CoNll-yago | Freebase & wiki  | 18,395    | 4,784   | /      |
| Entity Linking            | WNED            | Wiki             | /         | 8,995   | /      |
| Slot Filling              | T-REx           | DBPedia          | 2,284,168 | 5,000   | /      |
| Slot Filling              | Zero-shot RE    | wiki             | 147,909   | 3,724   | /      |
| In-domain QA              | DomainRAG       | Web pages of RUC | /         | /       | 485    |

### Document Corpus

Use `jsonl` format:

```jsonl
{"id":"0", "contents": "..."}
{"id":"1", "contents": "..."}
```

*   The `contents` key is essential for building indexes.
*   For titles and text, use `{title}\n{text}`.

Available resources:

*   Wikipedia ([Processing Wikipedia](./docs/original_docs/process-wiki.md)).
*   MS MARCO ([Hugging Face](https://huggingface.co/datasets/Tevatron/msmarco-passage-corpus)).
*   Preprocessed index: [FlashRAG_Dataset/retrieval_corpus/wiki18_100w_e5_index.zip](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/file/view/master?id=47985&status=2&fileName=retrieval_corpus%252Fwiki18_100w_e5_index.zip).

## Awesome Work Using FlashRAG <a name="lollipop"></a>

*   [R1-Searcher](https://github.com/SsmallSong/R1-Searcher)
*   [ReSearch](https://github.com/Agent-RL/ReSearch)
*   [AutoCoA](https://github.com/ADaM-BJTU/AutoCoA)

## FAQs <a name="raised_hands-additional-faqs"></a>

*   [Experimental parameter configuration?](./docs/original_docs/configuration.md)
*   [Corpus creation?](./docs/original_docs/process-wiki.md)
*   [Index building?](./docs/original_docs/building-index.md)
*   [Reproducing methods?](./docs/original_docs/reproduce_experiment.md)

## License <a name="bookmark-license"></a>

Licensed under the [MIT License](./LICENSE).

## Citation <a name="star2-citation"></a>

```bibtex
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