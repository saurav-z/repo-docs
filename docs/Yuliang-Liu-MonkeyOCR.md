# MonkeyOCR: Extract, Understand, and Structure Your Documents with AI

**Tired of manually extracting data from documents? MonkeyOCR uses a revolutionary Structure-Recognition-Relation (SRR) triplet paradigm to parse documents accurately and efficiently.** ([Original Repository](https://github.com/Yuliang-Liu/MonkeyOCR))

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR-pro-3B)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

**Key Features:**

*   **Superior Accuracy:** MonkeyOCR-pro-1.2B outperforms many state-of-the-art models on both English and Chinese document parsing.
*   **Blazing-Fast Performance:** Experience significant speed improvements, with up to a 36% speed boost compared to previous versions.
*   **SRR Paradigm:** Our innovative Structure-Recognition-Relation triplet paradigm simplifies document processing, avoiding the need for complex, multi-tool pipelines.
*   **Supports Various Document Types:**  Handles books, slides, financial reports, textbooks, and more.
*   **Ease of Use:**  Simple installation and inference steps, with a user-friendly Gradio demo and FastAPI.

**Performance Highlights:**

*   **MonkeyOCR-pro-1.2B excels:** Surpasses MonkeyOCR-3B by 7.4% on Chinese documents.
*   **Speed and Accuracy:**  Offers a ~36% speed increase with only a small (1.6%) performance drop.
*   **Competitive in Benchmarks:**  Outperforms Nanonets-OCR-3B on olmOCR-Bench.
*   **Top-Tier Results on OmniDocBench:** MonkeyOCR-pro-3B achieves the best overall performance, even surpassing closed-source and extra-large open-source VLMs like Gemini and GPT-4o.

**[See the demo](http://vlrlabmonkey.xyz:7685/) for a quick hands-on experience!** (The latest model is available for selection)

## Quick Start

### 1. Installation

   ```bash
   # Install MonkeyOCR
   See the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) to set up your environment.
   ```

### 2. Download Model Weights

   ```bash
   pip install huggingface_hub

   # Download MonkeyOCR-pro-3B (or MonkeyOCR)
   python tools/download_model.py -n MonkeyOCR-pro-3B
   ```

   Or from ModelScope:

   ```bash
   pip install modelscope

   # Download MonkeyOCR-pro-3B (or MonkeyOCR)
   python tools/download_model.py -t modelscope -n MonkeyOCR-pro-3B
   ```

### 3. Inference

   ```bash
   # Replace input_path with the path to your document (PDF, image, or directory)
   python parse.py input_path
   ```
   See details in the original README.

### 4. Gradio Demo

   ```bash
   python demo/demo_gradio.py
   ```

### 5. Fast API

   ```bash
   uvicorn api.main:app --port 8000
   ```

## Inference Speed

**Inference speed on different GPUs, showing performance across various page counts.**

*(Tables showcasing inference speeds for MonkeyOCR-pro-3B and MonkeyOCR-pro-1.2B on different GPUs)*

## Benchmark Results

**(Comprehensive evaluation results on OmniDocBench and olmOCR-bench, comparing MonkeyOCR to other methods and models.  The end-to-end evaluation results of different tasks.)**

## Supported Hardware
MonkeyOCR is tested on a variety of GPUs (3090, 4090, A6000, H800, etc.). Details in the original README.

## News
*   **[2025.07.10]** 🚀 MonkeyOCR-pro-1.2B released!
*   **[2025.06.12]** 🚀 Trending on Hugging Face!
*   **[2025.06.05]** 🚀 MonkeyOCR released.

## Docker Deployment

**(Instructions for building and running MonkeyOCR using Docker, including support for Gradio and FastAPI)**

## Windows Support

**(Link to Windows support guide.)**

## Quantization

**(Information on quantizing the model using AWQ.)**

## Citing MonkeyOCR

**(BibTeX entry for citing the work.)**

## Acknowledgments

**(Thank you notes)**

## Limitation
**(Known limitations of the current model.)**

## Copyright
**(Licensing and contact information.)**