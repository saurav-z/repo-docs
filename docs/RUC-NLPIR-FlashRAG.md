# ⚡ FlashRAG: The Ultimate Python Toolkit for RAG Research ⚡

> **FlashRAG empowers researchers to effortlessly explore and advance Retrieval-Augmented Generation (RAG) models, offering a comprehensive framework, pre-built components, and intuitive UI to accelerate your RAG journey.** 

[![arXiv](https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv)](https://arxiv.org/abs/2405.13576)
[![HuggingFace Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Datasets-27b3b4.svg)](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/)
[![ModelScope Datasets](https://custom-icon-badges.demolab.com/badge/ModelScope%20Datasets-624aff?style=flat&logo=modelscope&logoColor=white)](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset)
[![DeepWiki Document](https://devin.ai/assets/deepwiki-badge.png)](https://deepwiki.com/RUC-NLPIR/FlashRAG)
[![License](https://img.shields.io/badge/LICENSE-MIT-green)](https://github.com/RUC-NLPIR/FlashRAG/blob/main/LICENSE)
[![Made with Python](https://img.shields.io/badge/made_with-Python-blue)](https://www.python.org/)

**Explore the cutting-edge of RAG research with FlashRAG!** This powerful Python toolkit simplifies the development and reproduction of Retrieval-Augmented Generation (RAG) models.  With FlashRAG, you can easily build, evaluate, and customize RAG pipelines for diverse applications.

[**View Original Repo:**](https://github.com/RUC-NLPIR/FlashRAG)

## Key Features:

*   **Modular Framework:**  Construct complex RAG pipelines with plug-and-play components like retrievers, rerankers, generators, and compressors.
*   **Comprehensive Datasets:**  Access 36 pre-processed benchmark RAG datasets for robust model evaluation.
*   **Advanced Algorithms:**  Reproduce and experiment with **23 state-of-the-art RAG algorithms**, including **7 cutting-edge reasoning-based methods** for superior performance on multi-hop tasks.
*   **Reasoning-Based Methods:**  Explore and leverage **7 new reasoning-based methods** combining retrieval and reasoning for advanced task performance.
*   **Efficient Workflow:** Simplify your RAG workflow with streamlined preprocessing scripts for corpus preparation, indexing, and document retrieval.
*   **Optimized Performance:**  Leverage tools like vLLM and FastChat for LLM inference acceleration, and Faiss for efficient vector index management.
*   **User-Friendly UI:**  Easily configure, experiment with, and evaluate RAG models using the intuitive **FlashRAG-UI**.

## Sections

*   [Installation](#wrench-installation)
*   [Quick Start](#rocket-quick-start)
*   [Components](#gear-components)
*   [FlashRAG-UI](#art-flashrag-ui)
*   [Supporting Methods](#robot-supporting-methods)
*   [Supporting Datasets & Document Corpus](#notebook-supporting-datasets--document-corpus)
*   [FAQs](#raised_hands-additional-faqs)
*   [License](#bookmark-license)
*   [Citation](#star2-citation)
*   [Star History](https://github.com/RUC-NLPIR/FlashRAG#star-history)

## :wrench: Installation

Install FlashRAG with ease using pip:

```bash
pip install flashrag-dev --pre
```

Or install from source:

```bash
git clone https://github.com/RUC-NLPIR/FlashRAG.git
cd FlashRAG
pip install -e .
```

Install optional dependencies:

```bash
pip install flashrag-dev[full]
pip install vllm>=0.4.1  # For faster LLM inference
pip install sentence-transformers
pip install pyserini
```

**Important:** Install Faiss via conda:

```bash
conda install -c pytorch faiss-cpu=1.8.0  # CPU-only
conda install -c pytorch -c nvidia faiss-gpu=1.8.0  # GPU (if available)
```

## :rocket: Quick Start

1.  **Corpus Construction:** Prepare your document data in JSONL format.

    ```jsonl
    {"id": "0", "contents": "Document content..."}
    ```

2.  **Index Construction:** Build your index using the command-line tools provided.

    *   **Dense Retrieval (e.g., E5):**

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
        *   **Sparse Retrieval (e.g., BM25):**

            ```bash
            python -m flashrag.retriever.index_builder \
              --retrieval_method bm25 \
              --corpus_path indexes/sample_corpus.jsonl \
              --bm25_backend bm25s \
              --save_dir indexes/
            ```

3.  **Using the ready-made pipeline**

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

4.  **Build your own pipeline**

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

5.  **Just use components**

    ```python
    # See basic usage doc to implement your code
    ```

## :gear: Components

FlashRAG offers modular RAG components and pre-built pipelines for flexibility.

#### RAG Components

| Type          | Module                | Description                                                     |
| ------------- | --------------------- | --------------------------------------------------------------- |
| Judger        | SKR Judger            | Judger to judge whether to retrieve using SKR method            |
| Retriever     | Dense Retriever       | Bi-encoder models such as dpr, bge, e5, using faiss for search |
|               | BM25 Retriever        | Sparse retrieval method based on Lucene                        |
|               | Bi-Encoder Reranker   | Calculate matching score using bi-Encoder                        |
|               | Cross-Encoder Reranker | Calculate matching score using cross-encoder                      |
| Refiner       | Extractive Refiner    | Refine input by extracting important context                   |
|               | Abstractive Refiner   | Refine input through seq2seq model                              |
|               | LLMLingua Refiner     | LLMLingua-series prompt compressor                               |
|               | SelectiveContext Refiner | Selective-Context prompt compressor                             |
| Generator     | Encoder-Decoder       | Encoder-Decoder model, supporting FiD                          |
|               | Decoder-only          | Native transformers implementation                              |
|               | FastChat Generator    | Accelerate with FastChat                                        |
|               | vllm Generator        | Accelerate with vllm                                            |

#### Pipelines

| Type        | Module                | Description                                                                     |
| ----------- | --------------------- | ------------------------------------------------------------------------------- |
| Sequential  | Sequential Pipeline   | Linear execution of RAG process, supporting refiner, reranker                 |
| Conditional | Conditional Pipeline  | Implements different paths for different types of input queries                  |
| Branching   | REPLUG Pipeline       | Generate answer by integrating probabilities in multiple generation paths   |
|             | SuRe Pipeline      | Ranking and merging generated results based on each document                       |
| Loop        | Iterative Pipeline    | Alternating retrieval and generation                                           |
|             | Self-Ask Pipeline     | Decompose complex problems into subproblems using self-ask                     |
|             | Self-RAG Pipeline     | Adaptive retrieval, critique, and generation                                    |
|             | FLARE Pipeline        | Dynamic retrieval during the generation process                                 |
|             | IRCoT Pipeline        | Integrate retrieval process with CoT                                             |
|             | Reasoning Pipeline    | Reasoning with retrieval                                                        |

## :art: FlashRAG-UI

<p>The **FlashRAG-UI** offers an intuitive visual interface to configure, experiment with, and evaluate RAG models with ease.</p>

### :star2: Features

*   **Effortless Configuration Loading:** Quickly load configurations for various RAG methods.
*   **Rapid Method Experimentation:**  Explore the characteristics of RAG methods by loading datasets and indexes.
*   **Efficient Benchmark Reproduction:**  Reproduce built-in baseline methods efficiently.

```bash
cd webui
python interface.py
```

## :robot: Supporting Methods

The toolkit supports the following methods with results based on the following settings:

*   **Generator:** LLAMA3-8B-instruct (2048 input length)
*   **Retriever:** e5-base-v2 (retrieve 5 docs per query)
*   **Prompt:** Consistent default prompt

| Method                                                                                    | Type        | NQ (EM) | TriviaQA (EM) | Hotpotqa (F1) | 2Wiki (F1) | PopQA (F1) | WebQA(EM) |
| ----------------------------------------------------------------------------------------- | ----------- | ------- | ------------- | ------------- | ---------- | ---------- | --------- |
| Naive Generation                                                                          | Sequential  | 22.6    | 55.7          | 28.4          | 33.9       | 21.7       | 18.8      |
| Standard RAG                                                                              | Sequential  | 35.1    | 58.9          | 35.3          | 21.0       | 36.7       | 15.7      |
| Spring                                                                                  | Sequential  | 37.9    | 64.6          | 42.6          | 37.3       | 54.8       | 27.7      |
| SuRe   | Branching | 37.1    | 53.2          | 33.4          | 20.6       | 48.1       | 24.2      |

#### 🚀 Reasoning-based Methods (NEW!)

| Method                       | Type        | NQ (EM) | TriviaQA (EM) | PopQA (EM) | Hotpotqa (F1) | 2Wiki (F1) | Musique (F1) | Bamboogle (F1) |
| ---------------------------- | ----------- | ------- | ------------- | ---------- | ------------- | ---------- | ---------- | --------- |
| Search-R1                    | Reasoning   | 45.2    | 62.2          | 49.2       | 54.5          | 42.6       | 29.2       | 59.9      |
| R1-Searcher                  | Reasoning   | 36.9    | 61.6          | 42.0       | 49.0          | 49.1       | 24.7       | 57.7      |
|  O2-Searcher   | Reasoning   | 41.4    | 51.4          | 46.8       | 43.4          | 48.6       | 19.0       | 47.6      |
|  AutoRefine    | Reasoning   | 43.8    | 59.8          | 32.4       | 54.0          | 50.3       | 23.6       | 46.6      |

## :notebook: Supporting Datasets & Document Corpus

### Datasets

Access 36 pre-processed datasets via [<u>Huggingface datasets</u>](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets), saved as JSONL:

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
| QA                        | MSMARCO-QA      | web              | 808,731   | 101,093 | /      |
| multi-hop QA              | HotpotQA        | wiki             | 90,447    | 7,405   | /      |
| Open-Domain Summarization | WikiASP         | wiki             | 300,636   | 37,046  | 37,368 |

### Document Corpus

Use JSONL format for retrieval documents:

```jsonl
{"id":"0", "contents": "Document content..."}
```

Process Wikipedia data using the provided script.

### Index

Preprocessed index available at [FlashRAG_Dataset/retrieval_corpus/wiki18_100w_e5_index.zip](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/file/view/master?id=47985&status=2&fileName=retrieval_corpus%252Fwiki18_100w_e5_index.zip).

## :raised_hands: Additional FAQs

*   [Configuration guidance](./docs/original_docs/configuration.md)
*   [Building your own corpus](./docs/original_docs/process-wiki.md)
*   [Indexing your own corpus](./docs/original_docs/building-index.md)
*   [Reproducing supporting methods](./docs/original_docs/reproduce_experiment.md)

## :bookmark: License

FlashRAG is licensed under the [<u>MIT License</u>](./LICENSE).

## :star2: Citation

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