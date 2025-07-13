# MonkeyOCR: Extract Insight from Documents with AI (Document Parsing)

MonkeyOCR is a cutting-edge document parsing system that unlocks structured information from your documents using an innovative Structure-Recognition-Relation (SRR) triplet paradigm.  Explore the original repository: [Yuliang-Liu/MonkeyOCR](https://github.com/Yuliang-Liu/MonkeyOCR).

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

## Key Features

*   **SRR Paradigm:**  Simplifies document parsing by recognizing structure, extracting content, and understanding relationships between elements.
*   **Superior Performance:** MonkeyOCR-pro-1.2B outperforms previous versions and competing models on various benchmarks.
*   **Speed and Efficiency:** Optimized for faster processing speeds, with significant improvements in inference time.
*   **Versatile:** Supports both English and Chinese documents, handling text, formulas, and tables.
*   **Easy to Use:** Provides a simple command-line interface, Gradio demo, and FastAPI service for quick deployment.
*   **Multiple Output Formats:** Generates Markdown files, layout results (PDF), and detailed JSON data for comprehensive results.
*   **Quantization Support:** Supports AWQ quantization for efficient inference.

## Performance Highlights

*   **MonkeyOCR-pro-1.2B:**
    *   Outperforms MonkeyOCR-3B by 7.4% on Chinese documents.
    *   Delivers ~36% speed improvement over MonkeyOCR-pro-3B with only ~1.6% performance drop.
    *   Outperforms Nanonets-OCR-3B by 7.3% on olmOCR-Bench.
*   **MonkeyOCR-pro-3B:** Achieves state-of-the-art performance on OmniDocBench, even surpassing closed-source and extra-large open-source VLMs like Gemini and GPT-4o.

## Inference Speed

The table below shows pages/second (Pages/s) processing speeds on different GPUs.

| Model              | GPU   | 50 Pages | 100 Pages | 300 Pages | 500 Pages | 1000 Pages |
| ------------------ | ----- | -------- | --------- | --------- | --------- | ---------- |
| MonkeyOCR-pro-3B   | 3090  | 0.492    | 0.484     | 0.497     | 0.492     | 0.496      |
| MonkeyOCR-pro-3B   | A6000 | 0.585    | 0.587     | 0.609     | 0.598     | 0.608      |
| MonkeyOCR-pro-3B   | 4090  | 0.972    | 0.969     | 1.006     | 0.986     | 1.006      |
| MonkeyOCR-pro-1.2B | 3090  | 0.615    | 0.660     | 0.677     | 0.687     | 0.683      |
| MonkeyOCR-pro-1.2B | A6000 | 0.709    | 0.786     | 0.825     | 0.829     | 0.825      |
| MonkeyOCR-pro-1.2B | 4090  | 1.194    | 1.314     | 1.436     | 1.442     | 1.434      |

## Quick Start

### 1. Installation

Follow the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda.md#install-with-cuda-support) to set up your environment.

### 2. Download Model Weights

Download the model from Hugging Face:

```bash
pip install huggingface_hub
python tools/download_model.py -n MonkeyOCR-pro-1.2B  # MonkeyOCR
```

or from ModelScope:

```bash
pip install modelscope
python tools/download_model.py -t modelscope -n MonkeyOCR-pro-1.2B   # MonkeyOCR
```

### 3. Inference

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

For more detailed usage examples and configurations, see the original README.

### 4. Gradio Demo

Run the interactive demo:

```bash
python demo/demo_gradio.py
```

### 5. Fast API

Start the FastAPI service:

```bash
uvicorn api.main:app --port 8000
```

## Docker Deployment

See the original README for detailed Docker deployment instructions.

## Windows Support

See the [Windows Support](docs/windows_support.md) Guide for details.

## Quantization

See the [Quantization guide](docs/Quantization.md) for quantization instructions.

## Benchmark Results

See the original README for benchmark results.

## Visualization Demo

Experience MonkeyOCR firsthand: [http://vlrlabmonkey.xyz:7685](http://vlrlabmonkey.xyz:7685) (The latest model is available for selection)

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

(See the original README for acknowledgments.)

## Limitations

(See the original README for limitations.)

## Copyright

(See the original README for copyright information.)