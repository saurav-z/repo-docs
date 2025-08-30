# MonkeyOCR: Effortless Document Parsing with Intelligent Structure Recognition and Relationship Analysis

**Unlock the power of automated document understanding with MonkeyOCR, a cutting-edge solution for parsing documents using an innovative Structure-Recognition-Relation (SRR) triplet paradigm.** ([Original Repo](https://github.com/Yuliang-Liu/MonkeyOCR))

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace Weights](https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

## Key Features:

*   **SRR Paradigm:** MonkeyOCR employs a Structure-Recognition-Relation (SRR) triplet paradigm, offering a streamlined approach to document parsing.
*   **Superior Performance:** MonkeyOCR-pro-1.2B outperforms MonkeyOCR-3B in accuracy, speed, and efficiency, outperforming even closed-source VLMs.
*   **High Speed:**  Experience faster document processing; achieve 36% speed improvements compared to MonkeyOCR-pro-3B with minimal performance loss.
*   **Versatile:**  Supports parsing of various document types, including text, formulas, and tables in both English and Chinese.
*   **Model Variety:** Offers multiple model options, including MonkeyOCR-pro-3B and MonkeyOCR-pro-1.2B, allowing you to select the best fit for your needs.
*   **Multiple Deployment Options:** Deploy using local installation, Gradio demo, FastAPI, or Docker.
*   **Broad Hardware Support:** Runs on a range of GPUs, including 3090, 4090, A6000, H800, and more.
*   **Quantization Support:**  Quantize your model with AWQ for optimized performance.

## What's New

*   🚀 **2024.07.10**:  MonkeyOCR-pro-1.2B is released, offering faster processing and better accuracy, speed, and efficiency than the previous 3B version.
*   🚀 **2024.06.12**: Model trending on Hugging Face, reflecting high user interest and adoption.
*   🚀 **2024.06.05**:  Release of MonkeyOCR for English and Chinese document parsing.

## Quick Start Guide

### 1. Local Installation

*   **Prerequisites:**  See the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) to set up your environment.
*   **Download Model Weights:**
    *   Using Hugging Face:

        ```bash
        pip install huggingface_hub
        python tools/download_model.py -n MonkeyOCR-pro-3B  # or MonkeyOCR
        ```

    *   Using ModelScope:

        ```bash
        pip install modelscope
        python tools/download_model.py -t modelscope -n MonkeyOCR-pro-3B  # or MonkeyOCR
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

For additional usage examples, see the [detailed examples](https://github.com/Yuliang-Liu/MonkeyOCR#quick-start) in the original repository.

### 3. Gradio Demo

```bash
python demo/demo_gradio.py
```
Access the demo at http://localhost:7860.

### 4. FastAPI

```bash
uvicorn api.main:app --port 8000
```
Access API documentation at http://localhost:8000/docs.

## Docker Deployment

1.  Navigate to the `docker` directory:
    ```bash
    cd docker
    ```

2.  **Prerequisite:** Ensure NVIDIA GPU support is available in Docker (via `nvidia-docker2`).
    If GPU support is not enabled, run the following to set up the environment:
    ```bash
    bash env.sh
    ```

3.  Build the Docker image:
    ```bash
    docker compose build monkeyocr
    ```
    > [!IMPORTANT]
    >
    > If your GPU is from the 20/30/40-series, V100, L20/L40 or similar, please build the patched Docker image for LMDeploy compatibility:
    >
    > ```bash
    > docker compose build monkeyocr-fix
    > ```
    >
    > Otherwise, you may encounter the following error: `triton.runtime.errors.OutOfResources: out of resource: shared memory`

4.  Run the container with the Gradio demo:
    ```bash
    docker compose up monkeyocr-demo
    ```
    Alternatively, start an interactive development environment:
    ```bash
    docker compose run --rm monkeyocr-dev
    ```

5.  Run the FastAPI service:
    ```bash
    docker compose up monkeyocr-api
    ```
    Access API documentation at http://localhost:7861/docs.

## Windows Support

See the [windows support guide](docs/windows_support.md) for details.

## Quantization

This model can be quantized using AWQ. Follow the instructions in the [quantization guide](docs/Quantization.md).

## Benchmark Results

### 1. End-to-End Evaluation

For comprehensive performance comparisons across different models, consult the tables within the original README.

### 2. Text Recognition Performance

[See Tables in the original README]

### 3. OLMOCR-Bench Results

MonkeyOCR-pro-3B achieves strong performance, surpassing the open-source and other model in omlOCR-bench.

[See Tables in the original README]

## Demo and Visualization

**Experience MonkeyOCR's capabilities firsthand**: http://vlrlabmonkey.xyz:7685

## Citing MonkeyOCR

```bibtex
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

[See the original README for a list of acknowledgments]

## Limitations

*   Limited support for photographed text, handwritten content, and Traditional Chinese characters.
*   Demo performance can be affected by traffic.
*   Processing time includes overhead.

## Copyright

The model is intended for academic research and non-commercial use only. Contact xbai@hust.edu.cn or ylliu@hust.edu.cn for information on faster or stronger models.