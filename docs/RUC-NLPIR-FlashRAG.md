# FlashRAG: Supercharge Your RAG Research with This Python Toolkit!

**FlashRAG** is your all-in-one Python toolkit, designed for efficient research and development in Retrieval-Augmented Generation (RAG). [Explore the original repo!](https://github.com/RUC-NLPIR/FlashRAG)

[![arXiv](https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv)](https://arxiv.org/abs/2405.13576)
[![HuggingFace Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Datasets-27b3b4.svg)](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/)
[![ModelScope Datasets](https://custom-icon-badges.demolab.com/badge/ModelScope%20Datasets-624aff?style=flat&logo=modelscope&logoColor=white)](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset)
[![DeepWiki Document](https://devin.ai/assets/deepwiki-badge.png)](https://deepwiki.com/RUC-NLPIR/FlashRAG)
[![License](https://img.shields.io/badge/LICENSE-MIT-green)](https://github.com/RUC-NLPIR/FlashRAG/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/made_with-Python-blue)](https://www.python.org/)

<div align="center">
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
</div>

**FlashRAG empowers you to effortlessly reproduce cutting-edge RAG research and build custom, efficient RAG pipelines.** Our toolkit boasts a comprehensive suite of features and resources.

<p align="center">
<img src="asset/framework.jpg" alt="FlashRAG Framework">
</p>

[![Trendshift](https://trendshift.io/api/badge/repositories/10454)](https://trendshift.io/repositories/10454)

## Key Features

*   ✅ **Modular & Customizable:** Construct intricate RAG pipelines with our core components: retrievers, rerankers, generators, and more.
*   📚 **Extensive Datasets:** Test and validate your RAG models on 36 pre-processed benchmark datasets.
*   🚀 **SOTA Algorithms:** Explore and adapt **23 pre-implemented state-of-the-art RAG algorithms**, including 7 reasoning-based methods.
*   🧠 **Reasoning-Based RAG:** Now supporting **7 reasoning-based methods**, enhancing performance on complex multi-hop tasks.
*   ⚡ **Efficient Workflow:** Streamline your RAG preparation with tools for corpus processing, index building, and document retrieval.
*   🏎️ **Optimized Performance:** Leverage vLLM, FastChat, and Faiss for blazing-fast LLM inference and vector index management.
*   🖼️ **User-Friendly UI:**  Configure and experiment with our RAG baselines easily using FlashRAG-UI.

## 🚀 Quick Start

### Install & Get Started

1.  **Installation:** Install FlashRAG using `pip`:

```bash
pip install flashrag-dev --pre
```

2.  **Index Building:** Use the provided scripts and configuration options to build indexes for dense or sparse retrieval methods.

### Key Code Snippets

#### **For Dense Retrieval Methods**

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

#### For Sparse Retrieval Methods (BM25)

```bash
python -m flashrag.retriever.index_builder \
  --retrieval_method bm25 \
  --corpus_path indexes/sample_corpus.jsonl \
  --bm25_backend bm25s \
  --save_dir indexes/
```

##  🚀 What's New
*   **Reasoning Pipelines:** Enhanced to support 7 reasoning-based methods.
*   **Multimodal RAG:** Enhanced to support multimodal retrieval including LLaVA, Qwen, InternVL, and CLIP-based multimodal retrievers.
*   **23 SOTA RAG algorithms**
*   **FlashRAG-UI:** User-friendly interface for quick experimentation.

## FlashRAG-UI
<p>Effortlessly configure and experiment with RAG methods. With our user-friendly visual interface, easily set up and run experiments, all while evaluating results in real time.</p>
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

#### To launch FlashRAG-UI:
```bash
cd webui
python interface.py
```

## 🛠️ Core Components

FlashRAG provides a modular design with key components for building RAG systems:

*   **Retriever:** Dense and Sparse Retrievers
*   **Refiner:** Extractive, Abstractive, and LLMLingua Refiners, etc.
*   **Generator:** Encoder-Decoder, Decoder-only Generators, FastChat and vLLM support.
*   **Judger**: SKR Judger

## 🤖 Supporting Methods & Results

We've implemented **23 state-of-the-art RAG methods**. Find the detailed implementation details and settings in our [<u>reproduce guidance</u>](./docs/original_docs/reproduce_experiment.md) and [<u>method details</u>](./docs/original_docs/baseline_details.md).

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

## 📚 Supporting Datasets

Access a curated collection of 36 RAG datasets, pre-processed for seamless integration: [Huggingface datasets](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets).

### Preprocessed Corpus:

*   Access a preprocessed index using e5-base-v2 retriever on our uploaded wiki18_100w dataset [FlashRAG_Dataset/retrieval_corpus/wiki18_100w_e5_index.zip](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/file/view/master?id=47985&status=2&fileName=retrieval_corpus%252Fwiki18_100w_e5_index.zip)

## 🙋 FAQs

*   Configuration Guide: [<u>configuration guidance</u>](./docs/original_docs/configuration.md)
*   Building Corpus: [<u>process-wiki</u>](./docs/original_docs/process-wiki.md)
*   Indexing Your Corpus: [<u>building-index</u>](./docs/original_docs/building-index.md)
*   Reproducing Methods: [<u>reproduce_experiment</u>](./docs/original_docs/reproduce_experiment.md)

## ✨ Citation

If FlashRAG aids your research, please cite our paper:

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