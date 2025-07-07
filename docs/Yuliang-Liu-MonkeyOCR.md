# MonkeyOCR: Revolutionizing Document Parsing with SRR Triplet Paradigm

**MonkeyOCR** is a cutting-edge document parsing model that leverages a Structure-Recognition-Relation (SRR) triplet paradigm for superior performance.  [Access the original repository here](https://github.com/Yuliang-Liu/MonkeyOCR).

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)


> **MonkeyOCR: Document Parsing with a Structure-Recognition-Relation Triplet Paradigm**<br>
> Zhang Li, Yuliang Liu, Qiang Liu, Zhiyin Ma, Ziyang Zhang, Shuo Zhang, Zidun Guo, Jiarui Zhang, Xinyu Wang, Xiang Bai <br>
[![arXiv](https://img.shields.io/badge/Arxiv-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218) 
[![Source_code](https://img.shields.io/badge/Code-Available-white)](README.md)
[![Model Weight](https://img.shields.io/badge/HuggingFace-gray)](https://huggingface.co/echo840/MonkeyOCR)
[![Model Weight](https://img.shields.io/badge/ModelScope-green)](https://modelscope.cn/models/l1731396519/MonkeyOCR)
[![Public Courses](https://img.shields.io/badge/Openbayes-yellow)](https://openbayes.com/console/public/tutorials/91ESrGvEvBq)
[![Demo](https://img.shields.io/badge/Demo-blue)](http://vlrlabmonkey.xyz:7685/)

## Key Features:

*   **SRR Paradigm:** Simplifies document parsing by recognizing structure, content, and relationships in a unified approach.
*   **Superior Performance:** Achieves significant improvements over pipeline-based methods and end-to-end models.
    *   Up to 15% gain on formulas and 8.6% on tables.
    *   Outperforms models like Gemini 2.5 Pro and Qwen2.5 VL-72B on English documents (with 3B-parameter model).
*   **Fast Processing:** Processes multi-page documents at speeds comparable to or better than leading alternatives.
*   **Supports diverse Chinese and English PDF types**: Including books, slides, financial reports, textbooks, and more.
*   **Open Source:** Available for use and contribution.
*   **Easy to Use:** Includes a Gradio demo and Fast API deployment options.

## Quick Start

### 1. Installation

Follow the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda.md#install-with-cuda-support) to set up your environment.

### 2. Download Model Weights

Download the model weights from Hugging Face or ModelScope:

```python
# Hugging Face
pip install huggingface_hub
python tools/download_model.py

# ModelScope
pip install modelscope
python tools/download_model.py -t modelscope
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
```

</details>

#### Output Results

MonkeyOCR generates:

1.  **Markdown File** (`your.md`): Parsed document content.
2.  **Layout Results** (`your_layout.pdf`): Layout results overlaid on original PDF.
3.  **Intermediate Block Results** (`your_middle.json`): Detailed block information in JSON format.

### 4. Gradio Demo

Run the demo locally:

```bash
python demo/demo_gradio.py
```

### 5. Fast API

Start the API service:

```bash
uvicorn api.main:app --port 8000
```

Access the API documentation at http://localhost:8000/docs.

## Docker Deployment

Refer to the instructions in the original README for Docker deployment.

## Windows Support

Refer to the [Windows Support](docs/windows_support.md) Guide for details.

## Quantization

This model can be quantized using AWQ. Follow the instructions in the [Quantization guide](docs/Quantization.md).

## Benchmark Results

### 1. The end-to-end evaluation results of different tasks.
(Table formatting remains as in the original README)

### 2. The end-to-end text recognition performance across 9 PDF page types.
(Table formatting remains as in the original README)

### 3. Comparing MonkeyOCR with closed-source and extra large open-source VLMs.
(Image formatting remains as in the original README)

## Visualization Demo

Experience MonkeyOCR's capabilities: http://vlrlabmonkey.xyz:7685

### Example Documents
(Images and descriptions of sample documents remain as in the original README)

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

(Acknowledgments remain as in the original README)

## Alternative Models to Explore

(Alternative models remain as in the original README)

## Copyright

(Copyright notice remains as in the original README)