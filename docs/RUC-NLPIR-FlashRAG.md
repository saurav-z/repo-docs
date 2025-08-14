# FlashRAG: The Ultimate Python Toolkit for Efficient RAG Research

**Unleash the power of Retrieval Augmented Generation (RAG) with FlashRAG, a comprehensive Python toolkit designed for researchers and developers.**  This toolkit simplifies RAG research by providing a modular framework, pre-processed datasets, and state-of-the-art algorithms. [**Explore the FlashRAG Repo on GitHub!**](https://github.com/RUC-NLPIR/FlashRAG)

[![arXiv](https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv)](https://arxiv.org/abs/2405.13576)
[![HuggingFace Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Datasets-27b3b4.svg)](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/)
[![ModelScope Datasets](https://custom-icon-badges.demolab.com/badge/ModelScope%20Datasets-624aff?style=flat&logo=modelscope&logoColor=white)](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset)
[![DeepWiki Document](https://devin.ai/assets/deepwiki-badge.png)](https://deepwiki.com/RUC-NLPIR/FlashRAG)
[![License](https://img.shields.io/badge/LICENSE-MIT-green)](https://github.com/RUC-NLPIR/FlashRAG/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/made_with-Python-blue)](https://www.python.org/)

**Key Features:**

*   **Modular & Extensible:** Build complex RAG pipelines with our flexible components (Retriever, Generator, Refiner, etc.)
*   **Rich Benchmark Datasets:** Access 36 pre-processed RAG benchmark datasets for thorough testing.
*   **Pre-implemented RAG Algorithms:** Reproduce and experiment with **23 state-of-the-art RAG algorithms**, including results!
*   🚀 **Reasoning-Based Methods:** Explore cutting-edge advancements with **7 reasoning-based methods** that combine retrieval and reasoning capabilities!
*   **Efficient Preprocessing:** Simplify your workflow with tools for corpus processing, index building, and document retrieval.
*   **Optimized Execution:** Leverage vLLM, FastChat, and Faiss for faster LLM inference and vector index management.
*   **Easy-to-Use UI:** Quickly configure, experiment, and evaluate RAG baselines via the intuitive **FlashRAG-UI**.

**Get Started:**

*   [Installation](#wrench-installation)
*   [Quick Start](#rocket-quick-start)
*   [Explore the FlashRAG-UI](#art-flashrag-ui)

## :wrench: Installation

Install FlashRAG quickly using pip:

```bash
pip install flashrag-dev --pre
```

Or install from source:

```bash
git clone https://github.com/RUC-NLPIR/FlashRAG.git
cd FlashRAG
pip install -e .
```

For optional dependencies like vLLM, sentence-transformers, or pyserini, install them separately using:

```bash
pip install flashrag-dev[full]
pip install vllm>=0.4.1
pip install sentence-transformers
pip install pyserini
```

**Important:** For optimal performance, install `faiss` with conda:

```bash
# CPU-only
conda install -c pytorch faiss-cpu=1.8.0
# GPU (+ CPU)
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```

## :rocket: Quick Start

### Corpus Construction

Prepare your corpus as a `jsonl` file.

```jsonl
{"id": "0", "contents": "..."}
{"id": "1", "contents": "..."}
```

### Index Construction

Choose your retrieval method (Dense, Sparse, or Splade) and use the following commands. See the original README for full examples.

*   **Dense Retrieval (Faiss):**

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

*   **Sparse Neural Retrieval (SPLADE):**

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install pyseismic-lsr # Install Seismic

# Then build the index with Seismic:
python -m flashrag.retriever.index_builder \
        --retrieval_method splade \
        --model_path retriever/splade-v3 \
        --corpus_embedded_path data/ms_marco/ms_marco_embedded_corpus.jsonl
        --corpus_path data/ms_marco/ms_marco_corpus.jsonl
        --save_dir indexes/
        --use_fp16
        --max_length 512
        --batch_size 4
        --n_postings 1000
        --centroid_fraction 0.2
        --min_cluster_size 2
        --summary_energy 0.4
        --batched_indexing 10000000
        --nknn 32
```

### Ready-made Pipeline Usage

Use the `SequentialPipeline` and `Config` classes:

```python
from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline
from flashrag.prompt import PromptTemplate

# Load config and dataset
config_dict = {'data_dir': 'dataset/'}
my_config = Config(config_file_path = 'my_config.yaml', config_dict = config_dict)
all_split = get_dataset(my_config)
test_data = all_split['test']

# Create custom prompt
prompt_templete = PromptTemplate(
    config,
    system_prompt = "Answer the question based on the given document. Only give me the answer and do not output any other words.\nThe following are given documents.\n\n{reference}",
    user_prompt = "Question: {question}\nAnswer:"
)
pipeline = SequentialPipeline(my_config, prompt_template = prompt_templete)

# Run the pipeline
output_dataset = pipeline.run(test_data, do_eval=True)
```

## :gear: Components

FlashRAG provides a comprehensive set of RAG components for flexible pipeline construction.

**RAG Components:**

| Type       | Module          | Description                                                           |
| ---------- | --------------- | --------------------------------------------------------------------- |
| Judger     | SKR Judger      | Judging whether to retrieve using <a href="https://aclanthology.org/2023.findings-emnlp.691.pdf">SKR</a> method            |
| Retriever  | Dense Retriever | Bi-encoder models (dpr, bge, e5, etc.) using faiss                 |
| Retriever  | BM25 Retriever  | Sparse retrieval method based on Lucene                              |
| Reranker   | Bi-Encoder, Cross-Encoder| Calculate matching score using bi-Encoder/cross-encoder |
| Refiner    | Extractive, Abstractive, LLMLingua, SelectiveContext, KG Refiner   | Refine inputs using various methods |
| Generator  | Encoder-Decoder, Decoder-only, FastChat, vllm | Various generator models  |

**Pipelines:**

| Type         | Module              | Description                                                                                              |
| ------------ | ------------------- | -------------------------------------------------------------------------------------------------------- |
| Sequential   | Sequential Pipeline | Linear execution of query, supporting refiner, reranker                                                 |
| Conditional  | Conditional Pipeline| Distinct execution paths for various query types                                                        |
| Branching    | REPLUG Pipeline     | Generate answer by integrating probabilities in multiple generation paths                                |
| Branching | SuRe Pipeline  | Ranking and merging generated results based on each document  |
| Loop         | Iterative Pipeline  | Alternating retrieval and generation                                                                     |
| Loop         | Self-Ask Pipeline   | Decompose complex problems into subproblems using <a href="https://arxiv.org/abs/2210.03350">self-ask</a> |
| Loop         | Self-RAG Pipeline   | Adaptive retrieval, critique, and generation                                                              |
| Loop         | FLARE Pipeline      | Dynamic retrieval during the generation process                                                             |
| Loop         | IRCoT Pipeline      | Integrate retrieval process with CoT                                                                     |
| Loop         | Reasoning Pipeline  | Reasoning with retrieval                                                                                 |

## :art: FlashRAG-UI

**Interact easily!** Leverage **FlashRAG-UI** to configure, experiment with, and evaluate RAG methods through a user-friendly visual interface:

*   **One-Click Configuration Loading:** Easily load and manage parameters.
*   **Quick Method Experience:** Quickly load corpora and indexes.
*   **Efficient Benchmark Reproduction:** Reproduce baseline methods and benchmarks easily.

Run the UI:

```bash
cd webui
python interface.py
```

## :robot: Supporting Methods & Results

FlashRAG supports **23 diverse RAG methods**, including the new **7 reasoning-based methods**, using consistent settings for fair comparison. See the original README for a detailed results table and specific settings.

## :notebook: Supporting Datasets & Document Corpus

FlashRAG supports **36 pre-processed datasets**, including a broad range of QA, multi-hop QA, and summarization tasks.  All datasets are available on [<u>Huggingface datasets</u>](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets).

**Corpus:** Supports jsonl format.  See the Quick Start instructions above.

**Preprocessed Index:** Available on ModelScope: [FlashRAG\_Dataset/retrieval\_corpus/wiki18\_100w\_e5\_index.zip](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/file/view/master?id=47985&status=2&fileName=retrieval_corpus%252Fwiki18_100w_e5_index.zip).

## :raised_hands: Additional FAQs

*   [How should I set different experimental parameters?](./docs/original_docs/configuration.md)
*   [How to build my own corpus, such as a specific segmented Wikipedia?](./docs/original_docs/process-wiki.md)
*   [How to index my own corpus?](./docs/original_docs/building-index.md)
*   [How to reproduce supporting methods?](./docs/original_docs/reproduce_experiment.md)

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
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=RUC-NLPIR/FlashRAG&type=Date)](https://star-history.com/#RUC-NLPIR/FlashRAG&Date)