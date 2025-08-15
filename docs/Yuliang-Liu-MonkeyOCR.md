# MonkeyOCR: Revolutionizing Document Parsing with a Triplet Paradigm

**Effortlessly extract and understand information from complex documents with MonkeyOCR, a cutting-edge solution leveraging the Structure-Recognition-Relation (SRR) triplet paradigm.**

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

> **Explore the power of MonkeyOCR at its original repository: [https://github.com/Yuliang-Liu/MonkeyOCR](https://github.com/Yuliang-Liu/MonkeyOCR)**

## Key Features:

*   **SRR Paradigm:** Simplifies document parsing with a novel Structure-Recognition-Relation triplet approach.
*   **Superior Performance:** MonkeyOCR-pro-1.2B surpasses MonkeyOCR-3B by 7.4% on Chinese documents and outperforms other models.
*   **Exceptional Speed:** Achieve up to 36% speed improvements over previous models.
*   **Benchmarked Excellence:** Outperforms leading VLM models on benchmarks like OmniDocBench and olmOCR-Bench.
*   **Comprehensive Output:** Generates Markdown files, layout results (PDFs), and detailed intermediate block results (JSON).
*   **User-Friendly:** Includes a Gradio demo and FastAPI service for easy access and deployment.
*   **Hardware Flexibility:** Supports various GPUs including 3090, 4090, A6000, H800, and 4060.
*   **Windows Support:** Supports Windows systems with detailed installation and usage guides.
*   **Quantization:** Quantization support for efficient resource usage.

## Introduction

MonkeyOCR employs a Structure-Recognition-Relation (SRR) triplet paradigm to parse documents effectively. This approach streamlines multi-tool pipelines.

### Key Highlights:

*   MonkeyOCR-pro-1.2B outperforms MonkeyOCR-3B.
*   Speed improvements achieved with MonkeyOCR-pro-1.2B.
*   MonkeyOCR-pro-1.2B surpasses Nanonets-OCR-3B on olmOCR-Bench.
*   MonkeyOCR-pro-3B achieves top performance on OmniDocBench, even against closed-source models.

### Model Comparison:

[Include the image from the original README here.]

## Inference Speed

Detailed inference speed benchmarks are provided for various GPU configurations and PDF page counts.

[Include the tables from the original README here.]

## Supported Hardware

MonkeyOCR has been tested on a wide range of GPUs.

[Include the text about hardware support from the original README here.]

## Quick Start:

### 1. Local Installation

*   Install MonkeyOCR: See the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support)
*   Download Model Weights:
    *   From Huggingface:
        ```bash
        pip install huggingface_hub
        python tools/download_model.py -n MonkeyOCR  # or MonkeyOCR-pro-1.2B
        ```
    *   From ModelScope:
        ```bash
        pip install modelscope
        python tools/download_model.py -t modelscope -n MonkeyOCR  # or MonkeyOCR-pro-1.2B
        ```

### 2. Inference

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

[Include the "More Usage Examples" and "Output Results" details sections from the original README here, formatted as expandable details.]

### 3. Gradio Demo

```bash
python demo/demo_gradio.py
```

### 4. Fast API

```bash
uvicorn api.main:app --port 8000
```

## Docker Deployment

[Include the Docker deployment instructions from the original README here.]

## Windows Support

[Include the Windows Support link from the original README here.]

## Quantization

[Include the Quantization information from the original README here.]

## Benchmark Results

[Include the benchmark tables from the original README here.]

## Visualization Demo

[Include the "Get a Quick Hands-On Experience" text and demo link from the original README here.]

[Include the image examples from the original README here.]

## Citing MonkeyOCR

[Include the citation BibTeX from the original README here.]

## Acknowledgments

[Include the acknowledgments from the original README here.]

## Limitation

[Include the limitation section from the original README here.]

## Copyright

[Include the copyright text from the original README here.]