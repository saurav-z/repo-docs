# FlashRAG: Your Toolkit for Cutting-Edge Retrieval Augmented Generation (RAG) Research

**Tackle complex RAG research efficiently with FlashRAG, a versatile Python toolkit designed to reproduce SOTA works and build custom RAG pipelines.  [Explore the original repository](https://github.com/RUC-NLPIR/FlashRAG)!**

[![arXiv](https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv)](https://arxiv.org/abs/2405.13576)
[![HuggingFace Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Datasets-27b3b4.svg)](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/)
[![ModelScope Datasets](https://custom-icon-badges.demolab.com/badge/ModelScope%20Datasets-624aff?style=flat&logo=modelscope&logoColor=white)](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset)
[![DeepWiki Document](https://devin.ai/assets/deepwiki-badge.png)](https://deepwiki.com/RUC-NLPIR/FlashRAG)
[![License](https://img.shields.io/badge/LICENSE-MIT-green)](https://github.com/RUC-NLPIR/FlashRAG/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/made_with-Python-blue)](https://www.python.org/)
[![Trendshift](https://trendshift.io/api/badge/repositories/10454)](https://trendshift.io/repositories/10454)

<h4 align="center">
<a href="#wrench-installation">Installation</a> |
<a href="#sparkles-features">Features</a> |
<a href="#rocket-quick-start">Quick-Start</a> |
<a href="#gear-components"> Components</a> |
<a href="#art-flashrag-ui"> FlashRAG-UI</a> |
<a href="#robot-supporting-methods"> Supporting Methods</a> |
<a href="#notebook-supporting-datasets--document-corpus"> Supporting Datasets</a> |
<a href="#raised_hands-additional-faqs"> FAQs</a>
</h4>

<p align="center">
<img src="asset/framework.jpg" alt="FlashRAG Framework" width="700">
</p>

## Key Features

*   **Extensive and Customizable Framework**:  Build complex RAG pipelines with flexible components like retrievers, rerankers, generators, and compressors.
*   **Comprehensive Benchmark Datasets**: Evaluate your models on 36 pre-processed RAG benchmark datasets.
*   **State-of-the-Art Algorithms**: Reproduce results using **23 pre-implemented advanced RAG algorithms**.
*   **🚀 Reasoning-based Methods**:  Leverage **7 reasoning-based methods** to combine reasoning and retrieval for superior performance on multi-hop tasks.
*   **Efficient Workflow**: Simplify RAG preparation with tools for corpus processing, index building, and pre-retrieval.
*   **Optimized Execution**: Benefit from vLLM, FastChat, and Faiss for accelerated LLM inference and vector index management.
*   **Easy-to-Use UI**:  Configure and evaluate RAG baselines with a visual interface.

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
For optional dependencies:
```bash
pip install flashrag-dev[full]
pip install vllm>=0.4.1
pip install sentence-transformers
pip install pyserini
```
Install `faiss` and follow instructions to install `seismic`

## Quick Start

*   **Corpus Construction**: Prepare your data in a `jsonl` format.

```jsonl
{"id": "0", "contents": "..."}
{"id": "1", "contents": "..."}
```

*   **Index Construction**: Build indexes for efficient retrieval. Instructions provided for dense, sparse (BM25), and sparse neural retrieval methods (SPLADE)
    *   For **Dense Retrieval Methods**:

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

    *   For **Sparse Retrieval Methods (BM25)**:

```bash
python -m flashrag.retriever.index_builder \
  --retrieval_method bm25 \
  --corpus_path indexes/sample_corpus.jsonl \
  --bm25_backend bm25s \
  --save_dir indexes/ 
```

    *   For **Sparse Neural Retrieval Methods (SPLADE)**:

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
*   **Using the Ready-Made Pipeline**: Use the built-in `SequentialPipeline`

```python
from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline
from flashrag.prompt import PromptTemplate

config_dict = {'data_dir': 'dataset/'}
my_config = Config(
    config_file_path = 'my_config.yaml',
    config_dict = config_dict
)
all_split = get_dataset(my_config)
test_data = all_split['test']

pipeline = SequentialPipeline(my_config)
output_dataset = pipeline.run(test_data, do_eval=True)
```

## Components

FlashRAG offers modular components for building RAG pipelines:

*   **Judger**: SKR Judger
*   **Retriever**: Dense Retriever, BM25 Retriever, Bi-Encoder Reranker, Cross-Encoder Reranker
*   **Refiner**: Extractive Refiner, Abstractive Refiner, LLMLingua Refiner, SelectiveContext Refiner, KG Refiner
*   **Generator**: Encoder-Decoder Generator, Decoder-only Generator, FastChat Generator, vllm Generator

**Pipelines:**

*   **Sequential**:  Query-(pre-retrieval)-retriever-(post-retrieval)-generator
*   **Conditional**:  Different paths based on query type.
*   **Branching**: REPLUG Pipeline, SuRe Pipeline
*   **Loop**: Iterative Pipeline, Self-Ask Pipeline, Self-RAG Pipeline, FLARE Pipeline, IRCoT Pipeline, Reasoning Pipeline

## FlashRAG-UI

<p><strong>FlashRAG-UI</strong> provides a user-friendly visual interface for easy configuration, experimentation, and evaluation of RAG methods.</p>
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

#### Start the UI
```bash
cd webui
python interface.py
```

## Supporting Methods

FlashRAG supports **23 RAG methods**, using LLAMA3-8B-instruct, e5-base-v2, and a consistent prompt for comparison. The results are provided in the [<u>method details</u>](./docs/original_docs/baseline_details.md).

*   **7 Reasoning-based Methods (NEW!)**: Search-R1, R1-Searcher, O2-Searcher, AutoRefine, ReaRAG, CoRAG, SimpleDeepSearcher.

## Supporting Datasets & Document Corpus

### Datasets

36 pre-processed datasets for RAG research, including QA, multi-hop QA, long-form QA, and more.  Datasets available on [<u>Huggingface datasets</u>](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets).

### Document Corpus

Supports `jsonl` format.  Processed Wikipedia and MS MARCO datasets are available.

## Additional FAQs

*   [How should I set different experimental parameters?](./docs/original_docs/configuration.md)
*   [How to build my own corpus, such as a specific segmented Wikipedia?](./docs/original_docs/process-wiki.md)
*   [How to index my own corpus?](./docs/original_docs/building-index.md)
*   [How to reproduce supporting methods?](./docs/original_docs/reproduce_experiment.md)

## License

FlashRAG is licensed under the [<u>MIT License</u>](./LICENSE).

## Citation

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