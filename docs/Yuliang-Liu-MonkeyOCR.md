html
<div align="center">
  <h1>MonkeyOCR: Unlock Advanced Document Parsing with AI</h1>
</div>

<p>
  <strong>MonkeyOCR</strong> offers cutting-edge document parsing by leveraging a unique Structure-Recognition-Relation (SRR) triplet paradigm, providing superior accuracy and speed.  Explore the project on <a href="https://github.com/Yuliang-Liu/MonkeyOCR">GitHub</a>.
</p>

<div align="center">
  <!-- Shields -->
  <a href="https://arxiv.org/abs/2506.05218">
    <img src="https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv" alt="arXiv"/>
  </a>
  <a href="https://huggingface.co/echo840/MonkeyOCR">
    <img src="https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace" alt="HuggingFace Weights"/>
  </a>
  <a href="https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue">
    <img src="https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues" alt="GitHub issues"/>
  </a>
  <a href="https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed">
    <img src="https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues" alt="GitHub closed issues"/>
  </a>
  <a href="https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt">
    <img src="https://img.shields.io/badge/License-Apache%202.0-yellow" alt="License"/>
  </a>
  <a href="https://github.com/Yuliang-Liu/MonkeyOCR">
    <img src="https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views" alt="GitHub views"/>
  </a>
</div>

## Key Features

*   **Superior Accuracy:** MonkeyOCR achieves state-of-the-art results on various document types.
*   **Fast Processing:**  Efficient multi-page document parsing at 0.84 pages per second, surpassing competitors.
*   **SRR Paradigm:** Employs a novel Structure-Recognition-Relation triplet for streamlined document processing.
*   **Chinese and English Support:** Robustly parses both Chinese and English documents.
*   **Easy to Use:** Simple installation and intuitive API for quick integration.
*   **Flexible Output:** Generates Markdown, layout results (PDF), and detailed block information (JSON).
*   **Demo Available:**  Explore MonkeyOCR's capabilities through a user-friendly online demo.

## Introduction

MonkeyOCR revolutionizes document parsing by utilizing a Structure-Recognition-Relation (SRR) triplet paradigm. This innovative approach simplifies the multi-tool pipeline of traditional methods while avoiding the inefficiencies of large multimodal models for full-page document processing.  MonkeyOCR offers notable improvements in both speed and accuracy across diverse document types.

### Performance Highlights:

*   **Enhanced Accuracy:** MonkeyOCR achieves an average improvement of 5.1% across various Chinese and English documents compared to pipeline-based methods like MinerU.
*   **Competitive Edge:**  Outperforms models such as Gemini 2.5 Pro and Qwen2.5 VL-72B on English documents with a 3B-parameter model.
*   **High Speed:**  Processes multi-page documents at 0.84 pages per second, exceeding the speeds of MinerU and Qwen2.5 VL-7B.

## Quick Start

Follow these steps to get started with MonkeyOCR.

### 1. Installation
See the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda.md#install-with-cuda-support) to set up your environment.

### 2. Download Model Weights

Download the model weights from Hugging Face or ModelScope.

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

Parse documents using the provided command-line tools.

```bash
# Replace input_path with the path to a PDF, image, or directory
python parse.py input_path
```

**More Usage Examples:**

```bash
# Single file processing
python parse.py input.pdf                           # Parse single PDF file
python parse.py input.pdf -o ./output               # Parse with custom output dir
python parse.py input.pdf -s                        # Parse PDF with page splitting
python parse.py image.jpg                           # Parse single image file

# Single task recognition
python parse.py image.jpg -t text                   # Text recognition from image
python parse.py image.jpg -t formula                # Formula recognition from image
python parse.py image.jpg -t table                  # Table recognition from image
python parse.py document.pdf -t text                # Text recognition from all PDF pages

# Folder processing (all files individually)
python parse.py /path/to/folder                     # Parse all files in folder
python parse.py /path/to/folder -s                  # Parse with page splitting
python parse.py /path/to/folder -t text             # Single task recognition for all files

# Multi-file grouping (batch processing by page count)
python parse.py /path/to/folder -g 5                # Group files with max 5 total pages
python parse.py /path/to/folder -g 10 -s            # Group files with page splitting
python parse.py /path/to/folder -g 8 -t text        # Group files for single task recognition

# Advanced configurations
python parse.py input.pdf -c model_configs.yaml     # Custom model configuration
python parse.py /path/to/folder -g 15 -s -o ./out   # Group files, split pages, custom output
```

#### Output Results

MonkeyOCR generates three types of output files:

1.  **Processed Markdown File** (`your.md`):  The final parsed document content in markdown format, containing text, formulas, tables, and other structured elements.
2.  **Layout Results** (`your_layout.pdf`): The layout results drawn on the original PDF.
3.  **Intermediate Block Results** (`your_middle.json`): A JSON file containing detailed information about detected blocks, including coordinates, content, type, and relationships.

> [!TIP]
>
> For improved Chinese document parsing, consider using the layout\_zh.pt model or the PP-DocLayout\_plus-L model. Details in the original README.

### 4. Gradio Demo

Start the interactive Gradio demo.

```bash
python demo/demo_gradio.py
```

### 5. Fast API

Run the FastAPI service.

```bash
uvicorn api.main:app --port 8000
```

API documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

## Docker Deployment

See the original README for Docker deployment instructions.

## Windows Support

See the original README for Windows support instructions.

## Quantization

See the original README for Quantization instructions.

## Benchmark Results

### 1. The end-to-end evaluation results of different tasks.

(Table from Original README)

### 2. The end-to-end text recognition performance across 9 PDF page types.

(Table from Original README)

### 3. Comparing MonkeyOCR with closed-source and extra large open-source VLMs.

(Image from Original README)

## Visualization Demo

Experience MonkeyOCR's capabilities first-hand!  Visit the demo at: [http://vlrlabmonkey.xyz:7685](http://vlrlabmonkey.xyz:7685)

> **Demo Instructions:**
>
> 1.  Upload a PDF or image.
> 2.  Click "Parse (解析)" for structure detection and content recognition.
> 3.  Select a prompt and click "Test by prompt" for targeted recognition.

(Image Examples from Original README)

## Citing MonkeyOCR

(BibTeX entry from Original README)

## Acknowledgments

(Acknowledgments from Original README)

## Alternative Models to Explore

(Alternative Models from Original README)

## Copyright

(Copyright from Original README)