# MonkeyOCR: Effortlessly Parse Documents with Unmatched Accuracy and Speed (🚀 New MonkeyOCR-pro-1.2B!)

**MonkeyOCR** empowers you to unlock the structure within your documents using a cutting-edge Structure-Recognition-Relation (SRR) triplet paradigm, surpassing the capabilities of traditional, multi-tool approaches and large multimodal models.  [Explore the original repository on GitHub](https://github.com/Yuliang-Liu/MonkeyOCR).

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

## Key Features

*   **SRR Paradigm:** Streamlines document parsing by recognizing document structure, content, and relationships.
*   **Superior Performance:** MonkeyOCR-pro-1.2B surpasses the accuracy of MonkeyOCR-3B while offering enhanced speed.
*   **Blazing Fast:** Achieve up to a 36% speed improvement with MonkeyOCR-pro-1.2B.
*   **Top-Tier Results:** Outperforms other OCR models in accuracy and speed.
*   **Optimized for Diverse Documents:** Effectively parses English and Chinese documents, from financial reports to academic papers.
*   **Easy Deployment:**  Quickly get started with local installation, Docker, and FastAPI.

## What's New

*   **[2025.07.10]** 🚀 MonkeyOCR-pro-1.2B released. Faster, leaner, and more accurate than previous versions!
*   **[2025.06.12]** 🚀  Trending on Hugging Face!
*   **[2025.06.05]** 🚀  MonkeyOCR English and Chinese documents parsing model released.

## Performance Highlights

MonkeyOCR demonstrates remarkable performance, surpassing various other tools and models:

*   **MonkeyOCR-pro-1.2B** outshines MonkeyOCR-3B by 7.4% in Chinese document parsing.
*   MonkeyOCR-pro-1.2B provides about a 36% speed increase over MonkeyOCR-pro-3B while maintaining a drop of just about 1.6% in performance.
*   On the olmOCR-Bench, MonkeyOCR-pro-1.2B surpasses Nanonets-OCR-3B by 7.3%.
*   Achieves the best overall performance on both English and Chinese documents on OmniDocBench, surpassing even closed-source and extra-large open-source VLMs, such as Gemini 2.0-Flash, Gemini 2.5-Pro, Qwen2.5-VL-72B, GPT-4o, and InternVL3-78B.

### Performance Comparison

[Insert the Image Here (Replace with relevant image from original README, e.g., comparing MonkeyOCR with closed-source and extra large open-source VLMs.)]

## Inference Speed

Performance varies depending on the GPU used. The following tables show the inference speed for different models and GPU configurations.

### Inference Speed (Pages/s) on Different GPUs

**Inference Speed**
| Model          | GPU   | 50 Pages | 100 Pages | 300 Pages | 500 Pages | 1000 Pages |
| -------------- | ----- | -------- | --------- | --------- | --------- | ---------- |
| MonkeyOCR-pro-3B | 3090  | 0.492    | 0.484     | 0.497     | 0.492     | 0.496      |
|                | A6000 | 0.585    | 0.587     | 0.609     | 0.598     | 0.608      |
|                | H800  | 0.923    | 0.768     | 0.897     | 0.930     | 0.891      |
|                | 4090  | 0.972    | 0.969     | 1.006     | 0.986     | 1.006      |
| MonkeyOCR-pro-1.2B | 3090  | 0.615    | 0.660     | 0.677     | 0.687     | 0.683      |
|                | A6000 | 0.709    | 0.786     | 0.825     | 0.829     | 0.825      |
|                | H800  | 0.965    | 1.082     | 1.101     | 1.145     | 1.015      |
|                | 4090  | 1.194    | 1.314     | 1.436     | 1.442     | 1.434      |

### VLM OCR Speed (Pages/s) on Different GPUs

**VLM OCR Speed**
| Model          | GPU   | 50 Pages | 100 Pages | 300 Pages | 500 Pages | 1000 Pages |
| -------------- | ----- | -------- | --------- | --------- | --------- | ---------- |
| MonkeyOCR-pro-3B | 3090  | 0.705    | 0.680     | 0.711     | 0.700     | 0.724      |
|                | A6000 | 0.885    | 0.860     | 0.915     | 0.892     | 0.934      |
|                | H800  | 1.371    | 1.135     | 1.339     | 1.433     | 1.509      |
|                | 4090  | 1.321    | 1.300     | 1.384     | 1.343     | 1.410      |
| MonkeyOCR-pro-1.2B | 3090  | 0.919    | 1.086     | 1.166     | 1.182     | 1.199      |
|                | A6000 | 1.177    | 1.361     | 1.506     | 1.525     | 1.569      |
|                | H800  | 1.466    | 1.719     | 1.763     | 1.875     | 1.650      |
|                | 4090  | 1.759    | 1.987     | 2.260     | 2.345     | 2.415      |

## Supported Hardware

MonkeyOCR has been successfully tested on a variety of GPUs, including the 3090, 4090, A6000, H800, and A100.  It also supports the 4060 (8GB of VRAM) for quantized 3B and 1.2B models.

## Quick Start

### 1. Install MonkeyOCR

Follow the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support).

### 2. Download Model Weights

```bash
pip install huggingface_hub

python tools/download_model.py -n MonkeyOCR-pro-3B  # or MonkeyOCR
```
Alternatively, download from ModelScope:

```bash
pip install modelscope

python tools/download_model.py -t modelscope -n MonkeyOCR-pro-3B  # or MonkeyOCR
```

### 3. Inference

Use the following commands to parse files:

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

<details>
<summary><b>More usage examples</b></summary>

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
python parse.py input.pdf --pred-abandon            # Enable predicting abandon elements
  python parse.py /path/to/folder -g 10 -m            # Group files and merge text blocks in output
```

</details>

<details>
<summary><b>Output Results</b></summary>

MonkeyOCR produces these outputs:

1.  **Processed Markdown File** (`your.md`): Parsed document content in Markdown.
2.  **Layout Results** (`your_layout.pdf`): Results drawn on original PDF.
3.  **Intermediate Block Results** (`your_middle.json`): Detailed information about all detected blocks.

</details>

### 4. Gradio Demo

```bash
python demo/demo_gradio.py
```
Access the demo at:  http://localhost:7860

### 5. Fast API

```bash
uvicorn api.main:app --port 8000
```
Access API docs at: http://localhost:8000/docs

>   [!TIP]
>   Configure the inference backend as `lmdeploy_queue` or `vllm_queue` to improve API concurrency performance.

## Docker Deployment

1.  `cd docker`
2.  `bash env.sh` (If GPU support not enabled)
3.  `docker compose build monkeyocr`
    >   [!IMPORTANT]
    >   If your GPU is from the 20/30/40-series, V100, L20/L40 or similar, build the patched Docker image with: `docker compose build monkeyocr-fix`
4.  `docker compose up monkeyocr-demo`  (Gradio demo, port 7860)
    or `docker compose run --rm monkeyocr-dev` (interactive development)
5.  `docker compose up monkeyocr-api` (FastAPI, port 7861)

## Windows Support

See the [windows support guide](docs/windows_support.md)

## Quantization

Quantize with AWQ following the [quantization guide](docs/Quantization.md).

## Benchmark Results

### End-to-End Evaluation Results

[Insert the table here:  "1. The end-to-end evaluation results of different tasks."]

### End-to-end text recognition performance across 9 PDF page types

[Insert the table here:  "2. The end-to-end text recognition performance across 9 PDF page types."]

### The evaluation results of olmOCR-bench

[Insert the table here: "3. The evaluation results of olmOCR-bench."]

## Demo

[Provide the link to the online demo:  http://vlrlabmonkey.xyz:7685]

[Insert images (Example for formula document, table document, newspaper, financial report.)]

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

[List of acknowledgments]

## Limitations

*   Limited support for photographed text, handwritten content, Traditional Chinese characters, or multilingual text (planned for future releases).
*   Demo deployment on a single GPU may cause "application is busy" issues during high traffic.
*   Demo processing time includes overhead.
*   Inference speeds for MonkeyOCR, MinerU, and Qwen2.5 VL-7B measured on an H800 GPU.

## Copyright

This model is intended for academic research and non-commercial use only.  For inquiries about faster (smaller) or stronger models, contact xbai@hust.edu.cn or ylliu@hust.edu.cn.