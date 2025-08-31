# MonkeyOCR: Extracting Insights from Documents with AI

**Tired of tedious manual document processing?** MonkeyOCR utilizes a Structure-Recognition-Relation (SRR) triplet paradigm to offer a powerful and efficient solution for document parsing, providing accurate extraction of text, tables, formulas, and more.  [Explore the original repo](https://github.com/Yuliang-Liu/MonkeyOCR) for more details.

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

## Key Features:

*   **Advanced Document Parsing:**  Accurately extracts text, tables, formulas, and logical order from a variety of document types.
*   **Efficient SRR Paradigm:**  Simplifies the document parsing pipeline for faster and more efficient processing compared to complex multimodal models.
*   **High Performance:** MonkeyOCR-pro-1.2B outperforms previous versions and competitive models.
*   **Multi-Language Support:** Performs well on both English and Chinese documents.
*   **Optimized Speed:** Offers significant speed improvements on various GPUs.
*   **Easy Deployment:** Ready to use with local installation, Gradio demo, and Docker support, and Fast API service.
*   **Quantization Support:** Quantization is available to improve deployment speed.

## What's New:

*   **2025.07.10:** 🚀  Released [MonkeyOCR-pro-1.2B](https://huggingface.co/echo840/MonkeyOCR-pro-1.2B) – a faster and more accurate version.
*   **2025.06.12:** 🚀 Trending on [Hugging Face](https://huggingface.co/models?sort=trending).
*   **2025.06.05:** 🚀  Released [MonkeyOCR](https://huggingface.co/echo840/MonkeyOCR), for parsing both English and Chinese documents.

## Quick Start:

### 1. Installation
    *   Follow the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) to set up your environment.

### 2. Download Model Weights
    *   Use Hugging Face:

    ```python
    pip install huggingface_hub
    python tools/download_model.py -n MonkeyOCR-pro-3B  # or MonkeyOCR
    ```
    *   Or, use ModelScope:
    ```python
    pip install modelscope
    python tools/download_model.py -t modelscope -n MonkeyOCR-pro-3B  # or MonkeyOCR
    ```

### 3. Inference
    *   Parse a file or directory:
    ```bash
    python parse.py input_path
    ```
    *   Explore more advanced usage options in the original [README](https://github.com/Yuliang-Liu/MonkeyOCR).

### 4. Gradio Demo

*   Run the Gradio demo:
    ```bash
    python demo/demo_gradio.py
    ```
    *   Access the demo at http://localhost:7860

### 5. Fast API

*   Run the FastAPI service with:
    ```bash
    uvicorn api.main:app --port 8000
    ```
    *   Access the API documentation at http://localhost:8000/docs

## Docker Deployment
  * Follow the [docker deployment guide](https://github.com/Yuliang-Liu/MonkeyOCR#docker-deployment).

## Windows Support
  * Follow the [windows support guide](docs/windows_support.md).

## Quantization
  * Follow the [quantization guide](docs/Quantization.md).

## Benchmark Results:

*(Detailed benchmark results are available in the original README - consider including key highlights, like the models compared, what metrics were used, and a few standout results for each category.  Use tables if space allows.  For instance, highlight MonkeyOCR's performance on key benchmarks.)*

## Visualization Demo

Try the demo: http://vlrlabmonkey.xyz:7685

*(Include images and brief descriptions from the original README.)*

## Citing MonkeyOCR
```BibTeX
@misc{li2025monkeyocrdocumentparsingstructurerecognitionrelation,
      title={MonkeyOCR: Document Parsing with a Structure-Recognition-Relation Triplet Paradigm}, 
      author={Zhang Li and Yuliang Liu and Qiang Liu and Zhiyin Ma and Ziyang Zhang and Shuo Zhang and Zidun Guo and Jiarui Zhang and Xinyu Wang and Xiang Bai},
      year={2025},
      eprint={2506.05218},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.05218}, 
}
```
## Acknowledgments
*(Keep acknowledgements. Make sure the included links are working.)*
## Limitation
*(Keep limitations.)*

## Copyright
*(Keep the copyright and contact information.)*
```