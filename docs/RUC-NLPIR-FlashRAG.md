# FlashRAG: Your Toolkit for Cutting-Edge Retrieval Augmented Generation (RAG) Research

**FlashRAG empowers researchers to efficiently explore and advance Retrieval Augmented Generation (RAG) models.**  This comprehensive Python toolkit provides a modular framework, extensive datasets, and pre-implemented state-of-the-art algorithms to accelerate your research.

[![arXiv](https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv)](https://arxiv.org/abs/2405.13576)
[![HuggingFace Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Datasets-27b3b4.svg)](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/)
[![ModelScope Datasets](https://custom-icon-badges.demolab.com/badge/ModelScope%20Datasets-624aff?style=flat&logo=modelscope&logoColor=white)](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset)
[![DeepWiki](https://devin.ai/assets/deepwiki-badge.png)](https://deepwiki.com/RUC-NLPIR/FlashRAG)
[![License](https://img.shields.io/badge/LICENSE-MIT-green)](https://github.com/RUC-NLPIR/FlashRAG/blob/main/LICENSE)
[![Made with Python](https://img.shields.io/badge/made_with-Python-blue)](https://www.python.org/)

<p align="center">
  <a href="#wrench-installation">Installation</a> |
  <a href="#sparkles-features">Features</a> |
  <a href="#rocket-quick-start">Quick Start</a> |
  <a href="#gear-components">Components</a> |
  <a href="#art-flashrag-ui">FlashRAG-UI</a> |
  <a href="#robot-supporting-methods">Supporting Methods</a> |
  <a href="#notebook-supporting-datasets--document-corpus">Datasets</a> |
  <a href="#raised_hands-additional-faqs">FAQs</a>
</p>

FlashRAG offers a streamlined approach to RAG research by providing a comprehensive toolkit. It features 36 pre-processed benchmark datasets and **23 state-of-the-art RAG algorithms**, including **7 reasoning-based methods** that enhance retrieval with reasoning capabilities.

<p align="center">
  <img src="asset/framework.jpg" alt="FlashRAG Framework">
</p>

Reproduce SOTA results, build custom RAG pipelines, and accelerate your research with FlashRAG. Plus, experience our user-friendly interface:

[FlashRAG-UI](https://github.com/user-attachments/assets/8ca00873-5df2-48a7-b853-89e7b18bc6e9)

<p>
<a href="https://trendshift.io/repositories/10454" target="_blank"><img src="https://trendshift.io/api/badge/repositories/10454" alt="RUC-NLPIR%2FFlashRAG | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>


**Key Features:**

*   **Modular and Customizable Framework:** Design complex RAG pipelines with essential components like retrievers, rerankers, and generators.
*   **Comprehensive Benchmark Datasets:** Evaluate your models on 36 pre-processed RAG benchmark datasets.
*   **Pre-implemented SOTA Algorithms:** Quickly experiment with 23 advanced RAG algorithms.
*   **Reasoning-based Methods:** Leverage 7 reasoning-based methods to enhance retrieval for complex tasks.
*   **Efficient Preprocessing:** Simplify your workflow with scripts for corpus processing, index building, and document retrieval.
*   **Optimized Execution:** Utilize vLLM and FastChat for LLM inference acceleration, and Faiss for efficient vector indexing.
*   **User-Friendly UI:**  Easily configure and test RAG models with our intuitive FlashRAG-UI.

**Dive deeper into the project with these sections:**

*   [Installation](#wrench-installation)
*   [Quick Start](#rocket-quick-start)
*   [Components](#gear-components)
*   [FlashRAG-UI](#art-flashrag-ui)
*   [Supporting Methods](#robot-supporting-methods)
*   [Datasets & Document Corpus](#notebook-supporting-datasets--document-corpus)
*   [Additional FAQs](#raised_hands-additional-faqs)
*   [License](#bookmark-license)
*   [Citation](#star2-citation)

[See the original repository here:](https://github.com/RUC-NLPIR/FlashRAG)

---

### Roadmap

*   Support OpenAI models
*   Provide instructions for each component
*   Integrate Sentence Transformers
*   Support multimodal RAG
*   Support reasoning-based methods
*   Inlcude more RAG approaches
*   Enhance code adaptability and readability
*   Add support for api-based retriever (vllm server)

### Changelog

**Recent Updates:**

*   **[08/06/2025]**  Added support for Reasoning Pipeline.
*   **[21/03/2025]**  Expanded toolkit to support 23 state-of-the-art RAG algorithms, including 7 reasoning-based methods.
*   **[24/02/2025]**  Added support for multimodal RAG.
*   **[21/01/2025]**  Technical paper accepted to ACM Web Conference (WWW 2025).
*   **[12/01/2025]** Introduced FlashRAG-UI.
*   **[11/01/2025]** Added support for the RQRAG method.
*   **[07/01/2025]** Support the aggregation of multiple retrievers.
*   **[07/01/2025]** Integrated a very flexible and lightweight corpus chunking library [**Chunkie**](https://github.com/chonkie-ai/chonkie?tab=readme-ov-file#usage), which supports various custom chunking methods (tokens, sentences, semantic, etc.).
*   **[21/10/2024]** Released a version based on the Paddle framework that supports Chinese hardware platforms.
*   **[13/10/2024]** Added a new in-domain dataset - [DomainRAG](https://arxiv.org/pdf/2406.05654).
*   **[24/09/2024]** Released a version based on the MindSpore framework that supports Chinese hardware platforms.
*   **[18/09/2024]** Introduced a lightweight `BM25s` package as an alternative (faster and easier to use).
*   **[09/09/2024]** Add support for a new method [<u>Adaptive-RAG</u>](https://aclanthology.org/2024.naacl-long.389.pdf), which can automatically select the RAG process to execute based on the type of query.
*   **[02/08/2024]** Added support for a new method [<u>Spring</u>](https://arxiv.org/abs/2405.19670).
*   **[17/07/2024]** Updated HuggingFace dataset link.
*   **[06/07/2024]** Added support for a new method: [<u>Trace</u>](https://arxiv.org/abs/2406.11460).
*   **[19/06/2024]** Added support for a new method: [<u>IRCoT</u>](https://arxiv.org/abs/2212.10509), and update the [<u>result table</u>](#robot-supporting-methods).
*   **[15/06/2024]** Provided a [<u>demo</u>](./examples/quick_start/demo_en.py) to perform the RAG process using our toolkit.
*   **[11/06/2024]** Integrated `sentence transformers` in the retriever module.
*   **[05/06/2024]** Provided detailed document for reproducing existing methods.
*   **[02/06/2024]** Provided an introduction of FlashRAG for beginners.
*   **[31/05/2024]** Supported Openai-series models as generator.

### Installation

```bash
pip install flashrag-dev --pre
```

*   [Installation instructions](https://github.com/RUC-NLPIR/FlashRAG#wrench-installation)

### Quick Start

*   [Quick Start Guide](https://github.com/RUC-NLPIR/FlashRAG#rocket-quick-start)

### Components

*   [Components Overview](https://github.com/RUC-NLPIR/FlashRAG#gear-components)

### FlashRAG-UI

*   [UI features and instructions](https://github.com/RUC-NLPIR/FlashRAG#art-flashrag-ui)

### Supporting Methods

*   [Method Details & Results](https://github.com/RUC-NLPIR/FlashRAG#robot-supporting-methods)

### Datasets & Document Corpus

*   [Dataset List and Structure](https://github.com/RUC-NLPIR/FlashRAG#notebook-supporting-datasets--document-corpus)
*   [Document Corpus Details](https://github.com/RUC-NLPIR/FlashRAG#notebook-supporting-datasets--document-corpus)

### Additional FAQs

*   [Frequently Asked Questions](https://github.com/RUC-NLPIR/FlashRAG#raised_hands-additional-faqs)

---

### License

This project is licensed under the [MIT License](./LICENSE).

### Citation

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

### Star History

[![Star History Chart](https://api.star-history.com/svg?repos=RUC-NLPIR/FlashRAG&type=Date)](https://star-history.com/#RUC-NLPIR/FlashRAG&Date)