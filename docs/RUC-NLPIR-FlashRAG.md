# ⚡ FlashRAG: Your Toolkit for Cutting-Edge Retrieval-Augmented Generation (RAG) Research

**[FlashRAG](https://github.com/RUC-NLPIR/FlashRAG)** is a powerful Python toolkit designed to accelerate your research in Retrieval-Augmented Generation (RAG).  Reproduce state-of-the-art results, build custom RAG pipelines, and explore the forefront of AI with our modular and efficient framework.  

[![arXiv](https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv)](https://arxiv.org/abs/2405.13576)
[![HuggingFace Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Datasets-27b3b4.svg)](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/)
[![ModelScope Datasets](https://custom-icon-badges.demolab.com/badge/ModelScope%20Datasets-624aff?style=flat&logo=modelscope&logoColor=white)](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset)
[![DeepWiki](https://devin.ai/assets/deepwiki-badge.png)](https://deepwiki.com/RUC-NLPIR/FlashRAG)
[![License](https://img.shields.io/badge/LICENSE-MIT-green)](https://github.com/RUC-NLPIR/FlashRAG/blob/main/LICENSE)
[![Made with Python](https://img.shields.io/badge/made_with-Python-blue)](https://www.python.org/)
[![Trendshift](https://trendshift.io/api/badge/repositories/10454)](https://trendshift.io/repositories/10454)

## Key Features

*   **Modular & Flexible Framework:** Build RAG pipelines with customizable components: retrievers, rerankers, generators, and more.
*   **Extensive Benchmark Datasets:** Access 36 pre-processed RAG datasets for comprehensive evaluation.
*   **State-of-the-Art Algorithms:**  Reproduce and experiment with **23 advanced RAG algorithms**.
*   **🚀 Reasoning-Based Methods:** Explore **7 reasoning-based methods** for superior performance on complex tasks.
*   **Efficient Workflow:** Simplify your RAG workflow with tools for corpus processing, index building, and more.
*   **Optimized Execution:** Benefit from vLLM, FastChat, and Faiss for faster LLM inference and vector index management.
*   **User-Friendly UI:** Easily configure and evaluate RAG baselines with our intuitive **FlashRAG-UI** interface.

## Key Benefits

*   **Accelerate Research:** Quickly reproduce existing SOTA works and implement your own RAG processes.
*   **Simplified Experimentation:**  Easily test and validate your RAG models with pre-processed datasets.
*   **Improved Efficiency:** Reduce development time with a streamlined workflow and optimized execution.
*   **Community Driven:** Contribute and collaborate to enhance the toolkit for the benefit of all researchers.

## Sections

*   [Installation](#wrench-installation)
*   [Quick Start](#rocket-quick-start)
*   [Components](#gear-components)
*   [FlashRAG-UI](#art-flashrag-ui)
*   [Supporting Methods](#robot-supporting-methods)
*   [Supporting Datasets & Document Corpus](#notebook-supporting-datasets--document-corpus)
*   [Additional FAQs](#raised_hands-additional-faqs)
*   [License](#bookmark-license)
*   [Citation](#star2-citation)

## :wrench: Installation

Install FlashRAG with pip:

```bash
pip install flashrag-dev --pre
```

For detailed installation instructions and optional dependencies (vLLM, Sentence Transformers, Pyserini, Faiss), see the [Installation](#wrench-installation) section.

## :rocket: Quick Start

Quickly build indexes, use prebuilt pipelines and just use components.

### 1. Corpus Construction

Prepare your data in a `jsonl` format:

```jsonl
{"id": "0", "contents": "..."}
{"id": "1", "contents": "..."}
```

### 2. Index Construction

Build your index using the provided scripts.  Choose between dense (Faiss) and sparse (BM25, SPLADE) retrieval methods.  See the [Quick Start](#rocket-quick-start) section for detailed examples and instructions.

### 3. Using the ready-made pipeline

Learn to set up, configure, and run the RAG workflow using SequentialPipeline and PromptTemplate using config. See the [Quick Start](#rocket-quick-start) section for detailed examples and instructions.

### 4. Build your own pipeline!

Create custom RAG pipelines by inheriting `BasicPipeline` and defining your logic.  See the [Quick Start](#rocket-quick-start) section for detailed examples and instructions.

### 5. Just use components

Learn to use components using the [<u>basic introduction of the components</u>](./docs/original_docs/basic_usage.md) to obtain the input and output formats of each component.

## :gear: Components

FlashRAG provides a comprehensive set of RAG components, including:

**RAG-Components**

| Type        | Module          | Description                                                                    |
| ----------- | --------------- | ------------------------------------------------------------------------------ |
| Judger      | SKR Judger      | Judging whether to retrieve using SKR method                                   |
| Retriever   | Dense Retriever | Bi-encoder models such as dpr, bge, e5, using faiss for search                  |
| Retriever   | BM25 Retriever  | Sparse retrieval method based on Lucene                                        |
| Retriever   | Bi-Encoder Reranker | Calculate matching score using bi-Encoder                                        |
| Retriever   | Cross-Encoder Reranker | Calculate matching score using cross-encoder                                        |
| Refiner   | Extractive Refiner | Refine input by extracting important context                                        |
| Refiner   | Abstractive Refiner | Refine input through seq2seq model                                        |
| Refiner   | LLMLingua Refiner | LLMLingua-series prompt compressor                                                 |
| Refiner   | SelectiveContext Refiner | Selective-Context prompt compressor                                                 |
| Refiner   | KG Refiner | Use Trace method to construct a knowledge graph                                |
| Generator   | Encoder-Decoder Generator | Encoder-Decoder model, supporting Fusion-in-Decoder (FiD)                  |
| Generator   | Decoder-only Generator  | Native transformers implementation                                         |
| Generator   | FastChat Generator    | Accelerate with FastChat                                                     |
| Generator   | vllm Generator        | Accelerate with vllm                                                         |

**Pipelines**

FlashRAG implements various RAG pipelines:

| Type         | Module                 | Description                                                                      |
| ------------ | ---------------------- | -------------------------------------------------------------------------------- |
| Sequential   | Sequential Pipeline    | Linear execution of query, supporting refiner, reranker                        |
| Conditional  | Conditional Pipeline   | With a judger module, distinct execution paths for various query types            |
| Branching    | REPLUG Pipeline        | Generate answer by integrating probabilities in multiple generation paths      |
| Branching    | SuRe Pipeline           | Ranking and merging generated results based on each document                       |
| Loop         | Iterative Pipeline     | Alternating retrieval and generation                                             |
| Loop         | Self-Ask Pipeline      | Decompose complex problems into subproblems using self-ask                         |
| Loop         | Self-RAG Pipeline      | Adaptive retrieval, critique, and generation                                      |
| Loop         | FLARE Pipeline         | Dynamic retrieval during the generation process                                   |
| Loop         | IRCoT Pipeline         | Integrate retrieval process with CoT                                             |
| Loop         | Reasoning Pipeline | Reasoning with retrieval |

## :art: FlashRAG-UI

**FlashRAG-UI** offers an intuitive interface for configuring and experimenting with RAG methods.

### Features

*   **One-Click Configuration Loading:** Load parameters with ease.
*   **Quick Method Experience:** Explore methods through rapid corpus loading.
*   **Efficient Benchmark Reproduction:**  Reproduce built-in baselines easily.

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

FlashRAG supports the following methods:

**Implemented with consistent settings:**

*   Naive Generation
*   Standard RAG
*   AAR-contriever-kilt
*   LongLLMLingua
*   RECOMP-abstractive
*   Selective-Context
*   Trace
*   Spring
*   SuRe
*   REPLUG
*   SKR
*   Adaptive-RAG
*   Ret-Robust
*   Self-RAG
*   FLARE
*   Iter-Retgen, ITRG
*   IRCoT
*   RQRAG

**🚀 Reasoning-based Methods (NEW!)**

*   Search-R1
*   R1-Searcher
*   O2-Searcher
*   AutoRefine
*   ReaRAG
*   CoRAG
*   SimpleDeepSearcher

## :notebook: Supporting Datasets & Document Corpus

### Datasets

FlashRAG includes 36 pre-processed datasets for RAG evaluation:

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

Support for jsonl format corpus.

```jsonl
{"id":"0", "contents": "..."}
{"id":"1", "contents": "..."}
```

See [Supporting Datasets & Document Corpus](#notebook-supporting-datasets--document-corpus) for more details.

### Index

For easier replication, a preprocessed index is available in the ModelScope dataset page: [FlashRAG_Dataset/retrieval_corpus/wiki18_100w_e5_index.zip](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/file/view/master?id=47985&status=2&fileName=retrieval_corpus%252Fwiki18_100w_e5_index.zip).

## :raised_hands: Additional FAQs

*   [How should I set different experimental parameters?](./docs/original_docs/configuration.md)
*   [How to build my own corpus, such as a specific segmented Wikipedia?](./docs/original_docs/process-wiki.md)
*   [How to index my own corpus?](./docs/original_docs/building-index.md)
*   [How to reproduce supporting methods?](./docs/original_docs/reproduce_experiment.md)

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