# MonkeyOCR: Revolutionizing Document Parsing with Advanced Structure-Recognition-Relation

**Tired of clunky multi-tool pipelines?** MonkeyOCR provides a streamlined, efficient solution for document parsing, leveraging a cutting-edge Structure-Recognition-Relation (SRR) triplet paradigm. Explore the original repo [here](https://github.com/Yuliang-Liu/MonkeyOCR).

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

## Key Features:

*   **SRR Paradigm:** Simplifies document processing compared to modular approaches.
*   **Superior Performance:** MonkeyOCR-pro-1.2B outperforms Nanonets-OCR-3B and excels on both Chinese and English documents.
*   **Enhanced Speed:**  MonkeyOCR-pro-1.2B offers significant speed improvements over its 3B counterpart.
*   **State-of-the-Art Results:**  MonkeyOCR-pro-3B achieves the best overall performance, surpassing even closed-source and extra-large open-source VLMs.
*   **Model Weights Available:**  Downloadable weights on Hugging Face and ModelScope.
*   **Flexible Deployment:** Supports local installation, Gradio demo, and FastAPI service via Docker.

## Key Improvements (vs Original README)

*   **SEO Optimization:**  Keyword-rich headings and a concise introductory sentence.
*   **Clear Key Feature Bullets:**  Highlights key benefits.
*   **Concise Summarization:**  Focuses on the most important information.
*   **Improved Readability:**  Better formatting and organization.

## Performance Highlights

MonkeyOCR consistently demonstrates impressive results across various benchmarks.

### Comparing MonkeyOCR with closed-source and extra large open-source VLMs.
<a href="https://zimgs.com/i/EKhkhY"><img src="https://v1.ax1x.com/2025/07/15/EKhkhY.png" alt="EKhkhY.png" border="0" /></a>

### Inference Speed on Different GPUs

Detailed performance metrics are available.

## Quick Start

Easily get started with MonkeyOCR:

### 1. Install Dependencies
Follow the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support).

### 2. Download Model Weights

```bash
pip install huggingface_hub

python tools/download_model.py -n MonkeyOCR-pro-3B  # or MonkeyOCR
```

or, download from ModelScope:

```bash
pip install modelscope

python tools/download_model.py -t modelscope -n MonkeyOCR-pro-3B  # or MonkeyOCR
```

### 3. Run Inference

*   **Single File:** `python parse.py input.pdf`
*   **Batch Processing:** `python parse.py /path/to/folder`
*   **Task-Specific Parsing:** `python parse.py image.jpg -t text` (and other tasks)
*   **Split PDFs by page**: `python parse.py input.pdf -s`

See more examples, and details about output files below.

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

MonkeyOCR mainly generates three types of output files:

1.  **Processed Markdown File** (`your.md`): The final parsed document content in markdown format, containing text, formulas, tables, and other structured elements.
2.  **Layout Results** (`your_layout.pdf`): The layout results drawed on origin PDF.
3.  **Intermediate Block Results** (`your_middle.json`): A JSON file containing detailed information about all detected blocks, including:

    *   Block coordinates and positions
    *   Block content and type information
    *   Relationship information between blocks

These files provide both the final formatted output and detailed intermediate results for further analysis or processing.

</details>

### 4. Run the Gradio Demo
```bash
python demo/demo_gradio.py
```
Access the demo at http://localhost:7860.

### 5. Run the FastAPI Service
```bash
uvicorn api.main:app --port 8000
```
Access the API documentation at http://localhost:8000/docs.

## Docker Deployment

Streamline your workflow with Docker:

1.  `cd docker`
2.  `bash env.sh` (if necessary)
3.  `docker compose build monkeyocr`
    *   Use `docker compose build monkeyocr-fix` for 20/30/40-series and similar GPUs
4.  `docker compose up monkeyocr-demo` (Gradio)
5.  `docker compose run --rm monkeyocr-dev` (Development)
6.  `docker compose up monkeyocr-api` (FastAPI)

## Windows Support

See the [windows support guide](docs/windows_support.md) for details.

## Quantization

This model can be quantized using AWQ. Follow the instructions in the [quantization guide](docs/Quantization.md).

## Benchmark Results

View the comprehensive evaluation results:

### 1. OmniDocBench Results

[Tables of benchmark results are available here](https://github.com/Yuliang-Liu/MonkeyOCR#1-the-end-to-end-evaluation-results-of-different-tasks).

### 2. End-to-end text recognition across 9 PDF page types:

[Tables of text recognition performance are available here](https://github.com/Yuliang-Liu/MonkeyOCR#2-the-end-to-end-text-recognition-performance-across-9-pdf-page-types).

### 3. The evaluation results of olmOCR-bench:

[Tables of olmOCR-bench results are available here](https://github.com/Yuliang-Liu/MonkeyOCR#3-the-evaluation-results-of-olmocr-bench).

## Visualization Demo

Experience MonkeyOCR's capabilities firsthand: http://vlrlabmonkey.xyz:7685

### Diverse Chinese and English PDF Type Support

<p align="center">
  <img src="asserts/Visualization.GIF?raw=true" width="600"/>
</p>

### Formula Example

<img src="https://v1.ax1x.com/2025/06/10/7jVLgB.jpg" alt="7jVLgB.jpg" border="0" />

### Table Example

<img src="https://v1.ax1x.com/2025/06/11/7jcOaa.png" alt="7jcOaa.png" border="0" />

### Newspaper Example

<img src="https://v1.ax1x.com/2025/06/11/7jcP5V.png" alt="7jcP5V.png" border="0" />

### Financial Report Example

<img src="https://v1.ax1x.com/2025/06/11/7jc10I.png" alt="7jc10I.png" border="0" />
<img src="https://v1.ax1x.com/2025/06/11/7jcRCL.png" alt="7jcRCL.png" border="0" />

## Citation

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

We extend our gratitude to the listed open-source projects and contributors.

## Limitations

Currently, MonkeyOCR has limitations in the handling of photographed text, handwritten content, Traditional Chinese characters, and multilingual text. These limitations are planned to be addressed in future releases. Demo performance may be impacted by high traffic.

## Copyright

Our model is intended for academic research and non-commercial use only. Contact xbai@hust.edu.cn or ylliu@hust.edu.cn for inquiries.