# FlashRAG: Your Toolkit for Cutting-Edge RAG Research

**Unlock the power of Retrieval-Augmented Generation (RAG) with FlashRAG!** This Python toolkit provides everything you need to efficiently reproduce, develop, and experiment with state-of-the-art RAG models. From comprehensive benchmark datasets to pre-implemented advanced algorithms, FlashRAG empowers you to push the boundaries of RAG research.

[![](https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv)](https://arxiv.org/abs/2405.13576)
[![](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Datasets-27b3b4.svg)](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/)
[![](https://custom-icon-badges.demolab.com/badge/ModelScope%20Datasets-624aff?style=flat&logo=modelscope&logoColor=white)](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset)
[![](https://devin.ai/assets/deepwiki-badge.png)](https://deepwiki.com/RUC-NLPIR/FlashRAG)
[![](https://img.shields.io/badge/LICENSE-MIT-green)](https://github.com/RUC-NLPIR/FlashRAG/blob/main/LICENSE)
[![](https://img.shields.io/badge/made_with-Python-blue)](https://www.python.org/)

[**Original Repo:**](https://github.com/RUC-NLPIR/FlashRAG)

**Key Features:**

*   **Modular and Extensible Framework:** Easily assemble complex RAG pipelines with customizable components like retrievers, rerankers, generators, and more.
*   **Extensive Benchmark Datasets:** Evaluate your models on **36 pre-processed RAG datasets**, ensuring robust performance assessment.
*   **Advanced RAG Algorithms:** Access **23 pre-implemented, state-of-the-art RAG algorithms** with reported results, streamlining your research.
*   **Reasoning-Powered RAG:** Explore the cutting edge with support for **7 reasoning-based methods** that excel in complex, multi-hop tasks.
*   **Efficient Preprocessing:** Simplify your workflow with tools for corpus processing, index building, and document pre-retrieval.
*   **Optimized Execution:** Leverage libraries like vLLM and FastChat for faster LLM inference, and Faiss for efficient vector indexing.
*   **User-Friendly UI:** Quickly configure, experiment with, and evaluate RAG methods through our intuitive, easy-to-use FlashRAG-UI.

<p align="center">
    <img src="asset/framework.jpg" alt="Framework Overview" />
</p>

## Quick Navigation

*   [🚀 Installation](#wrench-installation)
*   [✨ Features](#sparkles-features)
*   [🧭 Quick Start](#rocket-quick-start)
*   [🛠️ Components](#gear-components)
*   [🖼️ FlashRAG-UI](#art-flashrag-ui)
*   [🧠 Supporting Methods](#robot-supporting-methods)
*   [📚 Supporting Datasets](#notebook-supporting-datasets--document-corpus)
*   [❓ FAQs](#raised_hands-additional-faqs)
*   [📄 License](#bookmark-license)
*   [🌟 Citation](#star2-citation)

## 🚀 Installation

Install FlashRAG using pip:

```bash
pip install flashrag-dev --pre
```

Alternatively, install from source:

```bash
git clone https://github.com/RUC-NLPIR/FlashRAG.git
cd FlashRAG
pip install -e .
```

**Optional Dependencies:**

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

For FAISS installation, use conda:

```bash
# CPU-only version
conda install -c pytorch faiss-cpu=1.8.0

# GPU(+CPU) version
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```

## ✨ Features

*   **Modular Design:** Build custom RAG pipelines with flexible components (retrievers, rerankers, generators, etc.).
*   **Comprehensive Benchmarks:** Access **36 pre-processed RAG benchmark datasets** for thorough evaluation.
*   **State-of-the-Art Implementations:** Experiment with **23 advanced RAG algorithms**, ready to run with reported results.
*   **Reasoning-Enhanced RAG:** Includes **7 reasoning-based methods** for improved performance on challenging tasks.
*   **Simplified Workflow:** Streamline your experiments with efficient preprocessing tools.
*   **Optimized Performance:** Leverage vLLM, FastChat, and Faiss for faster inference and index management.
*   **User-Friendly UI:** Visualize and experiment with RAG methods easily using FlashRAG-UI.

## 🧭 Quick Start

1.  **Corpus Construction:** Prepare your data in `jsonl` format:

    ```jsonl
    {"id": "0", "contents": "..."}
    {"id": "1", "contents": "..."}
    ```

2.  **Index Construction:** Choose your retrieval method (dense, sparse, or SPLADE) and build your index using the provided scripts:

    *   **Dense Retrieval:**

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

    *   **Sparse Retrieval (BM25):**

        ```bash
        python -m flashrag.retriever.index_builder \
            --retrieval_method bm25 \
            --corpus_path indexes/sample_corpus.jsonl \
            --bm25_backend bm25s \
            --save_dir indexes/
        ```
3.  **Using the ready-made pipeline**
    *   Load config file, you can input yaml files or config variables
    *   Load corresponding dataset and initialize the pipeline
    *   Specify input prompt using `PromptTemplete`
    *   execute `pipeline.run` to get the final result

4.  **Build your own pipeline**
    *   Inherit `BasicPipeline`
    *   Initialize the components you need
    *   Complete the `run` function

5.  **Just use components**
    *   Refer to the [<u>basic introduction of the components</u>](./docs/original_docs/basic_usage.md)

## 🛠️ Components

FlashRAG offers a comprehensive suite of RAG components:

**RAG Components:**

| Type        | Module            | Description                                                 |
| ----------- | ----------------- | ----------------------------------------------------------- |
| Judger      | SKR Judger        | Determines retrieval based on the SKR method.               |
| Retriever   | Dense/BM25/Reranker | Various retrieval methods (Bi-Encoder, BM25, Cross-Encoder).  |
| Refiner     | Extractive/Abstractive | Refine inputs using different strategies.                  |
| Generator   | Encoder-Decoder/Decoder-only | Different generator models including FastChat/vllm     |

**Pipelines:**

| Type         | Module              | Description                                                                                                |
| ------------ | ------------------- | ---------------------------------------------------------------------------------------------------------- |
| Sequential   | Sequential Pipeline | Linear execution of RAG process                                                                            |
| Conditional  | Conditional Pipeline| Different paths for different query types                                                                    |
| Branching    | REPLUG/SuRe Pipeline    | Executes multiple paths in parallel                                                                                               |
| Loop         | Iterative/Self-Ask/Self-RAG/FLARE/IRCoT/Reasoning Pipeline | Iterative RAG processes                                                                                                 |

## 🖼️ FlashRAG-UI

<p>Explore and experiment with RAG methods using the intuitive <strong>FlashRAG-UI</strong> interface, which is both user-friendly and visually appealing.
</p>

### FlashRAG-UI Features:
*   **Easy Configuration Loading:** Quickly load RAG method parameters and configuration files.
*   **Rapid Method Exploration:** Load corpora and index files to explore various RAG methods.
*   **Efficient Benchmark Reproduction:** Easily reproduce built-in baseline methods and benchmarks.

To launch the UI:

```bash
cd webui
python interface.py
```

## 🧠 Supporting Methods

We have implemented **23 works** with a consistent setting of:

-   **Generator:** LLAMA3-8B-instruct with input length of 2048
-   **Retriever:** e5-base-v2 as embedding model, retrieve 5 docs per query
-   **Prompt:** A consistent default prompt, template can be found in the [<u>method details</u>](./docs/original_docs/baseline_details.md).

We also provide the result of our supporting methods, please see below table:

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

**🚀 Reasoning-based Methods (NEW!)**

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

## 📚 Supporting Datasets

FlashRAG provides access to a comprehensive collection of datasets.  All datasets are available at [<u>Huggingface datasets</u>](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets).

**Dataset Overview:**

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

*   Supports `jsonl` format:

    ```jsonl
    {"id":"0", "contents": "..."}
    {"id":"1", "contents": "..."}
    ```

*   For Wikipedia, use our [<u>comprehensive script</u>](./docs/original_docs/process-wiki.md) or explore existing processed versions.
*   MS MARCO corpus is available on Hugging Face.
*   [FlashRAG_Dataset/retrieval_corpus/wiki18_100w_e5_index.zip](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/file/view/master?id=47985&status=2&fileName=retrieval_corpus%252Fwiki18_100w_e5_index.zip).

## ❓ FAQs

*   [How to configure experiment parameters?](./docs/original_docs/configuration.md)
*   [How to process and build your own corpus?](./docs/original_docs/process-wiki.md)
*   [How to index my corpus?](./docs/original_docs/building-index.md)
*   [How to reproduce the supporting methods?](./docs/original_docs/reproduce_experiment.md)

## 📄 License

FlashRAG is licensed under the [<u>MIT License</u>](./LICENSE).

## 🌟 Citation

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