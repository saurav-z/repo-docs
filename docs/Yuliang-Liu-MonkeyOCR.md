<div align="center">
  <h1>MonkeyOCR: Unlocking Document Structure with Cutting-Edge OCR Technology</h1>
  <p><em>Effortlessly parse documents with MonkeyOCR, utilizing a revolutionary Structure-Recognition-Relation (SRR) triplet paradigm.</em></p>
  <p>
    <a href="https://arxiv.org/abs/2506.05218"><img src="https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv" alt="arXiv"/></a>
    <a href="https://huggingface.co/echo840/MonkeyOCR"><img src="https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace" alt="Hugging Face"/></a>
    <a href="https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue"><img src="https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues" alt="GitHub Issues"/></a>
    <a href="https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed"><img src="https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues" alt="GitHub Closed Issues"/></a>
    <a href="https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt"><img src="https://img.shields.io/badge/License-Apache%202.0-yellow" alt="License"/></a>
    <a href="https://github.com/Yuliang-Liu/MonkeyOCR"><img src="https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views" alt="GitHub Views"/></a>
  </p>
</div>

> **[MonkeyOCR](https://github.com/Yuliang-Liu/MonkeyOCR):** Harnessing a unique Structure-Recognition-Relation (SRR) triplet paradigm, MonkeyOCR offers a streamlined approach to document parsing, surpassing limitations of conventional methods.

**Key Features:**

*   **Superior Accuracy:** MonkeyOCR-pro-1.2B outperforms MonkeyOCR-3B by 7.4% on Chinese documents.
*   **Enhanced Speed:** Achieve up to 36% faster processing with MonkeyOCR-pro-1.2B compared to MonkeyOCR-pro-3B.
*   **Competitive Performance:** MonkeyOCR-pro-1.2B excels on olmOCR-Bench, outperforming Nanonets-OCR-3B by 7.3%.
*   **Leading Results:** On OmniDocBench, MonkeyOCR-pro-3B achieves top-tier performance, outclassing both closed-source and large open-source VLMs.

## Performance Highlights

### Comparative Benchmarking
**(Image of comparative results table - see original README)**

### Inference Speed by GPU
**(Tables showing inference speeds on different GPUs, pages/second - see original README)**

## Quick Start

### 1. Installation
Follow the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) to set up your environment.

### 2. Model Download
Download the model weights from Hugging Face or ModelScope:

```bash
# Hugging Face
pip install huggingface_hub
python tools/download_model.py -n MonkeyOCR-pro-3B  # or MonkeyOCR

# ModelScope
pip install modelscope
python tools/download_model.py -t modelscope -n MonkeyOCR-pro-3B  # or MonkeyOCR
```

### 3. Inference

Use the following commands to parse documents:

```bash
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

**(More usage examples, output result details - See original README)**

### 4. Gradio Demo
Launch a user-friendly demo with:

```bash
python demo/demo_gradio.py
```
Access the demo at: `http://localhost:7860`

### 5. FastAPI API
Deploy a fast API with:

```bash
uvicorn api.main:app --port 8000
```
API Documentation: `http://localhost:8000/docs`
> [!TIP]
> To improve API concurrency performance, consider configuring the inference backend as `lmdeploy_queue` or `vllm_queue`.

## Docker Deployment
**(See original README)**

## Windows Support
**(See original README)**

## Quantization
**(See original README)**

## Benchmark Results
**(See original README)**

### The end-to-end evaluation results of different tasks.
**(See original README)**

### The end-to-end text recognition performance across 9 PDF page types.
**(See original README)**

### The evaluation results of olmOCR-bench.
**(See original README)**

## Visualization Demo

**(Images and descriptions of demo functionality - See original README)**

## Citing MonkeyOCR
**(See original README)**

## Acknowledgments
**(See original README)**

## Limitation
**(See original README)**

## Copyright
**(See original README)**
```

Key improvements and summaries:

*   **SEO Optimization:** Title and headings use relevant keywords (Document Parsing, OCR, Structure Recognition).
*   **Concise Hook:** A one-sentence summary immediately grabs attention.
*   **Key Features:** Bulleted list makes key benefits easily scannable.
*   **Clear Structure:** Sections are well-defined.
*   **Conciseness:** Condensed the text, removing redundant phrases.
*   **Emphasis on Benefits:**  Performance numbers are highlighted.
*   **Call to Action:**  Encourages quick start by showing how to set up the model.
*   **Links:** Preserved and organized the links
*   **Clean formatting:** Improved readability with bolding and consistent indentation.