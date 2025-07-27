# MonkeyOCR: Revolutionizing Document Parsing with a Structure-Recognition-Relation Triplet Paradigm

**Tired of clunky, multi-tool document processing pipelines? MonkeyOCR simplifies document parsing, offering unparalleled accuracy and speed.**

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

👉 **[Explore the MonkeyOCR Repo on GitHub](https://github.com/Yuliang-Liu/MonkeyOCR)**

## Key Features

*   **SRR Paradigm:** Employs a Structure-Recognition-Relation (SRR) triplet paradigm, streamlining document parsing.
*   **Superior Performance:** Achieves state-of-the-art results, outperforming both closed-source and open-source models.
*   **Speed & Efficiency:** Offers significant speed improvements with minimal performance trade-off.
*   **Model Variants:** Choose from MonkeyOCR-pro-1.2B and MonkeyOCR-pro-3B, tailored for different needs.
*   **Diverse Hardware Support:** Runs on various GPUs, including 3090, 4090, A6000, H800, A100 and more (and community contributions for even broader compatibility).
*   **Easy to Use:** Simple installation and inference with command-line tools and a Gradio demo.
*   **FastAPI Integration:** Provides a service using FastAPI for easy integration.
*   **Docker Support:** Seamless deployment with Docker for easy setup and use.

## Performance Highlights

*   **MonkeyOCR-pro-1.2B** surpasses MonkeyOCR-3B by 7.4% on Chinese documents.
*   **MonkeyOCR-pro-1.2B** is approximately 36% faster than MonkeyOCR-pro-3B, with only ~1.6% performance drop.
*   On the **olmOCR-Bench**, MonkeyOCR-pro-1.2B outperforms Nanonets-OCR-3B by 7.3%.
*   **MonkeyOCR-pro-3B** achieves the best overall performance on both English and Chinese documents on **OmniDocBench**, outperforming the likes of Gemini 2.0-Flash, Gemini 2.5-Pro, Qwen2.5-VL-72B, GPT-4o, and InternVL3-78B.

## Benchmark Results

**(See detailed tables below)**

### Comparing MonkeyOCR with closed-source and extra large open-source VLMs.
<a href="https://zimgs.com/i/EKhkhY"><img src="https://v1.ax1x.com/2025/07/15/EKhkhY.png" alt="EKhkhY.png" border="0" /></a>

## Inference Speed (Pages/s) on Different GPUs

**(See tables for detailed performance across various GPU configurations and page counts.)**

## Quick Start

### 1. Installation

1.  Follow the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) to set up your environment.

### 2. Download Model Weights

Download the model from Hugging Face:

```bash
pip install huggingface_hub
python tools/download_model.py -n MonkeyOCR  # or MonkeyOCR-pro-1.2B
```

Alternatively, download from ModelScope:

```bash
pip install modelscope
python tools/download_model.py -t modelscope -n MonkeyOCR  # or MonkeyOCR-pro-1.2B
```

### 3. Inference

Parse PDFs or images using:

```bash
# Replace input_path with the path to a PDF or image or directory

# End-to-end parsing
python parse.py input_path

# Parse files in a dir with specific group page num
python parse.py input_path -g 20

# Single-task recognition (outputs markdown only)
python parse.py input_path -t text/formula/table

# Parse PDFs in input_path and split results by pages
python parse.py input_path -s

# Specify output directory and model config file
python parse.py input_path -o ./output -c config.yaml
```

**(See more usage examples and output file details in the original README.)**

### 4. Gradio Demo

Run the interactive demo:

```bash
python demo/demo_gradio.py
```

Access the demo at `http://localhost:7860`.

### 5. FastAPI Service

Start the FastAPI service:

```bash
uvicorn api.main:app --port 8000
```

Access the API documentation at `http://localhost:8000/docs`.

**(See the original README for Docker Deployment, Windows Support, Quantization, Benchmark Results, Visualization Demo, and further details.)**