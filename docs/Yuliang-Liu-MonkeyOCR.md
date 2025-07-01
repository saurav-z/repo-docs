# MonkeyOCR: Revolutionizing Document Parsing with Structure-Recognition-Relation Triplet Paradigm

**MonkeyOCR** offers a cutting-edge approach to document parsing, providing superior performance and efficiency.  [Explore the code on GitHub](https://github.com/Yuliang-Liu/MonkeyOCR) to revolutionize your document processing workflows!

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

*   **Key Features:**

    *   **Superior Performance:** MonkeyOCR achieves state-of-the-art results on both English and Chinese document parsing, outperforming existing pipeline-based and end-to-end models.
    *   **Structure-Recognition-Relation (SRR) Paradigm:**  Employs a novel SRR triplet paradigm for efficient and accurate document understanding.
    *   **Faster Processing:**  Processes multi-page documents at impressive speeds, significantly exceeding the performance of other leading solutions.
    *   **Comprehensive Output:** Generates parsed documents in Markdown format, layout results, and detailed intermediate block results for flexibility.
    *   **Ease of Use:**  Provides simple installation and easy-to-use command-line tools and a Gradio demo for quick experimentation.
    *   **Open Source & Accessible:** Download pre-trained weights and leverage the open-source code base for customization and research.

## Introduction

MonkeyOCR utilizes a Structure-Recognition-Relation (SRR) triplet paradigm for document parsing. This innovative approach simplifies the modular multi-tool pipelines of traditional methods while avoiding the computational overhead of large multimodal models, resulting in superior performance, especially on English and Chinese documents. MonkeyOCR is designed to streamline your document processing workflows with improved speed and accuracy.

### Key Advantages:

*   **Improved Accuracy:** MonkeyOCR demonstrates significant gains in parsing accuracy across various document types, outperforming pipeline-based methods. For instance, it achieves improvements of 5.1% on average across nine types of Chinese and English documents.
*   **High Efficiency:** MonkeyOCR boasts processing speeds that surpass existing solutions. It reaches a processing speed of 0.84 pages per second for multi-page documents.
*   **High Performance:** The 3B-parameter model achieves the best average performance on English documents, outperforming models such as Gemini 2.5 Pro and Qwen2.5 VL-72B.

### Performance Highlights:

*   **Overall edit↓: 0.140 EN, 0.277 ZH** - MonkeyOCR-3B is state of the art in Overall Edit
*   **Text Edit↓: 0.058 EN, 0.134 ZH** - MonkeyOCR-3B provides better text recognition compared to the other models.
*   **Formula Edit↓: 0.238 EN, 0.529 ZH** - The model is particularly effective at formula recognition, with strong performance on both English and Chinese documents.
*   **Table TEDS↑: 80.2 EN, 76.2 ZH** - MonkeyOCR-3B excels in table structure analysis and extraction.

## Quick Start

Get started with MonkeyOCR in just a few steps:

### 1. Installation

Follow the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda.md#install-with-cuda-support) to set up your environment.

### 2. Download Model Weights

Download the model weights from Hugging Face or ModelScope:

*   **Hugging Face:**

    ```bash
    pip install huggingface_hub
    python tools/download_model.py
    ```

*   **ModelScope:**

    ```bash
    pip install modelscope
    python tools/download_model.py -t modelscope
    ```

### 3. Inference

Use the following commands to parse documents:

```bash
# Replace input_path with your PDF or image path
python parse.py input_path

# Single-task recognition (outputs markdown only)
python parse.py input_path -t text/formula/table

# Specify output directory and model config file
python parse.py input_path -o ./output -c config.yaml

# Parse images in input_path(a dir) in groups with specific group size
python parse.py input_path -g 20

# Parse a PDF and split results by pages
python parse.py your.pdf -s
```

### 4. Gradio Demo

Run the Gradio demo for a hands-on experience:

```bash
python demo/demo_gradio.py
```

### 5. FastAPI

Start the FastAPI service:

```bash
uvicorn api.main:app --port 8000
```

## Docker Deployment

*   Navigate to the `docker` directory:

    ```bash
    cd docker
    ```

*   Ensure NVIDIA GPU support in Docker (via `nvidia-docker2`):

    ```bash
    bash env.sh
    ```

*   Build the Docker image:

    ```bash
    docker compose build monkeyocr
    ```
    **Note:** If your GPU is from the 30/40-series, V100, or similar, build the patched Docker image: `docker compose build monkeyocr-fix`

*   Run with Gradio demo:

    ```bash
    docker compose up monkeyocr-demo
    ```

*   Run the FastAPI service:

    ```bash
    docker compose up monkeyocr-api
    ```

## Windows Support

For Windows deployment, utilize WSL and Docker Desktop.  See the [Windows Support](docs/windows_support.md) guide.

## Quantization

Quantize the model using AWQ. Refer to the [Quantization guide](docs/Quantization.md) for instructions.

## Benchmark Results

*   **OmniDocBench Evaluation:** Comprehensive results are provided, comparing MonkeyOCR with various methods across different tasks and document types.  The results are broken down by task (Text, Formula, Table) and document type (Book, Slides, Financial Report, etc.).
*   **Table 1:** Presents the end-to-end evaluation results, including Overall Edit, Text Edit, Formula Edit, Formula CDM, Table TEDS, Table Edit, and Read Order Edit, for both English and Chinese documents.
*   **Table 2:**  Shows the end-to-end text recognition performance across 9 PDF page types (Book, Slides, Financial Report, etc.).
*   **Performance Comparisons:** Visual comparisons with closed-source and large open-source VLMs highlight MonkeyOCR's superior performance.

## Visualization Demo

Experience MonkeyOCR firsthand!  Visit our interactive demo:  http://vlrlabmonkey.xyz:7685

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

Special thanks to the contributors, open-source resources, and datasets that supported the development of MonkeyOCR, including MinerU, DocLayout-YOLO, PyMuPDF, Qwen2.5-VL, LMDeploy, and many others.

## Alternative Models

Explore alternatives like PP-StructureV3 and MinerU 2.0 if MonkeyOCR doesn't fully meet your needs.

## Copyright

MonkeyOCR is intended for non-commercial use. For larger models or commercial inquiries, please contact the authors at xbai@hust.edu.cn or ylliu@hust.edu.cn.