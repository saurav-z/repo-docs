# FlashRAG: The Ultimate Python Toolkit for Efficient Retrieval-Augmented Generation (RAG) Research

> Revolutionize your RAG research with FlashRAG, a powerful Python toolkit offering state-of-the-art algorithms, pre-processed datasets, and an easy-to-use UI. [Explore FlashRAG on GitHub](https://github.com/RUC-NLPIR/FlashRAG).

[![arXiv](https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv)](https://arxiv.org/abs/2405.13576)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Datasets-27b3b4.svg)](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/)
[![ModelScope Datasets](https://custom-icon-badges.demolab.com/badge/ModelScope%20Datasets-624aff?style=flat&logo=modelscope&logoColor=white)](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset)
[![DeepWiki Document](https://devin.ai/assets/deepwiki-badge.png)](https://deepwiki.com/RUC-NLPIR/FlashRAG)
[![License](https://img.shields.io/badge/LICENSE-MIT-green)](https://github.com/RUC-NLPIR/FlashRAG/blob/main/LICENSE)
[![Made with Python](https://img.shields.io/badge/made_with-Python-blue)](https://www.python.org/)
[![Trendshift](https://trendshift.io/api/badge/repositories/10454)](https://trendshift.io/repositories/10454)

FlashRAG is a comprehensive Python toolkit designed to accelerate Retrieval-Augmented Generation (RAG) research. It empowers researchers and developers to efficiently reproduce existing state-of-the-art (SOTA) results, experiment with custom RAG pipelines, and explore cutting-edge advancements in the field.

**Key Features:**

*   **Extensive Framework:** A modular design featuring essential RAG components (retrievers, rerankers, generators, compressors) for flexible pipeline construction.
*   **Rich Datasets:** Includes 36 pre-processed benchmark RAG datasets to facilitate rigorous evaluation and comparison of RAG models.
*   **Advanced Algorithms:**  Offers **23 pre-implemented, cutting-edge RAG algorithms** with reported results, including **7 reasoning-based methods** for superior performance on complex tasks.
*   **Reasoning Capabilities:** Supports **7 Reasoning-based Methods** that combine reasoning ability with retrieval to achieve the state-of-the-art results for the multi-hop tasks.
*   **Efficient Workflow:** Provides streamlined preprocessing scripts for corpus processing, index building, and document retrieval.
*   **Optimized Performance:** Leverages technologies like vLLM, FastChat for LLM inference acceleration, and Faiss for vector index management, enhancing efficiency.
*   **User-Friendly UI:** Integrates FlashRAG-UI for easy configuration, experimentation, and evaluation of implemented RAG baselines.

**Key Benefits:**

*   **Reproduce SOTA results easily.**
*   **Experiment with custom pipelines.**
*   **Accelerate your RAG research.**

**Navigate the Toolkit:**

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

Or clone from GitHub and install:

```bash
git clone https://github.com/RUC-NLPIR/FlashRAG.git
cd FlashRAG
pip install -e .
```

(Optional) Install extra dependencies:

```bash
pip install flashrag-dev[full]
pip install vllm>=0.4.1
pip install sentence-transformers
pip install pyserini
```

## :rocket: Quick Start

### Corpus Construction

Save your corpus as a `jsonl` file.

```jsonl
{"id": "0", "contents": "..."}
{"id": "1", "contents": "..."}
```

### Index Construction

**Dense Retrieval (e.g., E5, BGE):**

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

**Sparse Retrieval (BM25):**

```bash
python -m flashrag.retriever.index_builder \
  --retrieval_method bm25 \
  --corpus_path indexes/sample_corpus.jsonl \
  --bm25_backend bm25s \
  --save_dir indexes/
```

### Use the ready-made pipeline

Load and Configure the Pipeline with a config file:

```python
from flashrag.config import Config
config_dict = {'data_dir': 'dataset/'}
my_config = Config(
    config_file_path = 'my_config.yaml',
    config_dict = config_dict
)
```

Load the Dataset and Run

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

## :gear: Components

FlashRAG offers a wide range of RAG components:

**RAG Components:**

| Type         | Module               | Description                                      |
|--------------|----------------------|--------------------------------------------------|
| Judger       | SKR Judger           | Judge to retrieve using SKR method               |
| Retriever    | Dense/BM25 Retriever | Various retrieval methods, Bi-encoder, BM25      |
| Reranker     | Bi/Cross-Encoder     | Reranking with Bi-Encoder/Cross-Encoder methods  |
| Refiner      | Extractive/Abstractive/LLMLingua/SelectiveContext/KG  | Refine input by extracting/compressing context |
| Generator    | Encoder-Decoder, Decoder-only | Different generation architectures       |

**Pipelines:**

*   **Sequential:** Standard RAG pipeline.
*   **Conditional:**  Pipeline with a judger to select execution paths.
*   **Branching:** REPLUG & SuRe  pipelines.
*   **Loop:** Iterative, Self-Ask, Self-RAG, FLARE, IRCoT, Reasoning pipelines.

## :art: FlashRAG-UI

**FlashRAG-UI** provides a user-friendly interface for configuring and experimenting with RAG methods:

*   **One-Click Configuration:** Load parameters via clicks.
*   **Quick Method Experience:**  Explore RAG methods with ease.
*   **Efficient Benchmark Reproduction:** Reproduce baselines with ease.

Launch the UI:

```bash
cd webui
python interface.py
```

## :robot: Supporting Methods

FlashRAG implements **23 RAG methods**, including **7 Reasoning-based methods**:

**Reasoning-based Methods (NEW!)**

| Method                                                                                    | Type        | NQ (EM) | TriviaQA (EM) | PopQA (EM) | Hotpotqa (F1) | 2Wiki (F1) |  Musique (F1) | Bamboogle (F1) | Specific setting                             |
| ----------------------------------------------------------------------------------------- | ----------- | ------- | ------- | ------------- | ------------- | ---------- | ---------- | --------- | ----------------------------------------------- |
| [Search-R1](https://arxiv.org/abs/2503.09516) | Reasoning | 45.2 | 62.2 | 49.2 | 54.5 | 42.6 | 29.2 |  59.9 | SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo |
| [R1-Searcher](https://arxiv.org/pdf/2503.05592) | Reasoning | 36.9 | 61.6 | 42.0 | 49.0 | 49.1 | 24.7 | 57.7 | Qwen-2.5-7B-base-RAG-RL |
| [O2-Searcher](https://arxiv.org/pdf/2505.16582) | Reasoning | 41.4 | 51.4 | 46.8 | 43.4 | 48.6 | 19.0 | 47.6 | O2-Searcher-Qwen2.5-3B-GRPO |
| [AutoRefine](https://www.arxiv.org/pdf/2505.11277) | Reasoning | 43.8 | 59.8 | 32.4 | 54.0 | 50.3 | 23.6 | 46.6 | AutoRefine-Qwen2.5-3B-Base |
| [ReaRAG](https://arxiv.org/abs/2503.21729) | Reasoning | 26.3 | 51.8 | 24.6 | 42.9 | 41.6 | 21.2 | 41.9 | ReaRAG-9B |
| [CoRAG](https://arxiv.org/abs/2503.21729) | Reasoning | 40.9 | 63.1 | 36.0 | 56.6 | 60.7 | 31.9 | 54.1 | CoRAG-Llama3.1-8B-MultihopQA |
| [SimpleDeepSearcher](https://arxiv.org/pdf/2505.16834) | Reasoning | 36.1 | 61.6 | 42.0 | 49.0 | 49.1 | 24.7 | 57.7 | Qwen-7B-SimpleDeepSearcher |

**General RAG Methods**

| Method | Type | NQ (EM) | TriviaQA (EM) | Hotpotqa (F1) | 2Wiki (F1) | PopQA (F1) | WebQA(EM) | Specific setting |
|---|---|---|---|---|---|---|---|---|
| Standard RAG | Sequential | 35.1 | 58.9 | 35.3 | 21.0 | 36.7 | 15.7 |  |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

(See original README for full table)

## :notebook: Supporting Datasets & Document Corpus

FlashRAG supports 36 RAG datasets:

| Task                      | Dataset Name    | Knowledge Source | # Train   | # Dev   | # Test |
| ------------------------- | --------------- | ---------------- | --------- | ------- | ------ |
| QA                        | NQ              | wiki             | 79,168    | 8,757   | 3,610  |
| ... | ... | ... | ... | ... | ... |

(See original README for full table)

**Corpus Format:** `{"id":"0", "contents": "..."}`

Preprocessed index is available on ModelScope: [FlashRAG\_Dataset/retrieval\_corpus/wiki18\_100w\_e5\_index.zip](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/file/view/master?id=47985&status=2&fileName=retrieval_corpus%252Fwiki18_100w_e5_index.zip).

## :raised_hands: Additional FAQs

*   [Configuration Guidance](./docs/original_docs/configuration.md)
*   [Building Your Own Corpus](./docs/original_docs/process-wiki.md)
*   [Indexing Your Corpus](./docs/original_docs/building-index.md)
*   [Reproducing Methods](./docs/original_docs/reproduce_experiment.md)

## :bookmark: License

This project is licensed under the [MIT License](./LICENSE).

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
```

## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=RUC-NLPIR/FlashRAG&type=Date)](https://star-history.com/#RUC-NLPIR/FlashRAG&Date)