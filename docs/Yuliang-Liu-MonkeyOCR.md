# MonkeyOCR: Revolutionizing Document Parsing with SRR Triplet Paradigm

**Tired of slow, clunky document processing pipelines?** MonkeyOCR offers a streamlined, high-performance solution for parsing complex documents using a Structure-Recognition-Relation (SRR) triplet paradigm. Explore the original repo [here](https://github.com/Yuliang-Liu/MonkeyOCR).

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

## Key Features

*   **SRR Paradigm:** Simplifies document parsing compared to multi-tool pipelines, and avoids the inefficiency of large multimodal models.
*   **Superior Performance:** Achieves state-of-the-art results on various benchmarks, outperforming both closed-source and open-source VLMs.
    *   **MonkeyOCR-pro-1.2B** outperforms MonkeyOCR-3B by 7.4% on Chinese documents.
    *   **MonkeyOCR-pro-1.2B** delivers approximately a 36% speed improvement over MonkeyOCR-pro-3B, with approximately 1.6% drop in performance.
    *   Outperforms Nanonets-OCR-3B by 7.3% on olmOCR-Bench.
    *   MonkeyOCR-pro-3B achieves the best overall performance on OmniDocBench on both English and Chinese documents, outperforming even closed-source and extra-large open-source VLMs such as Gemini 2.0-Flash, Gemini 2.5-Pro, Qwen2.5-VL-72B, GPT-4o, and InternVL3-78B.
*   **Fast Inference:** Optimized for speed, with detailed inference speed benchmarks across different GPUs.
*   **Multiple Model Sizes:** Offers both 3B and 1.2B parameter models for flexible deployment based on your needs.
*   **Easy to Use:**  Provides a simple command-line interface, Gradio demo, and FastAPI service for easy integration and testing.
*   **Flexible Deployment:** Supports various hardware, including 3090, 4090, A6000, H800, and 4060 GPUs, and provides Docker support.
*   **Quantization:** Offers support for AWQ quantization to optimize models for resource-constrained environments.

## Benchmarks & Performance

MonkeyOCR demonstrates impressive performance across several benchmarks, including:

### [OmniDocBench Results - Performance Summary](https://v1.ax1x.com/2025/07/15/EKhkhY.png)

### Inference Speed

Achieved high inference speed. See full table in the original README.

## Quick Start

### 1. Installation

Follow the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) to set up your environment.

### 2. Download Model Weights

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

Use the `parse.py` script to process documents:

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

### 4. Gradio Demo

```bash
python demo/demo_gradio.py
```

Access the demo at http://localhost:7860.

### 5. Fast API

```bash
uvicorn api.main:app --port 8000
```

Access the API documentation at http://localhost:8000/docs.

## Docker Deployment

Instructions are available in the original README, or follow the steps below:
1.  `cd docker`
2.  Ensure NVIDIA GPU support. (`bash env.sh`)
3.  Build: `docker compose build monkeyocr`
4.  Run Gradio: `docker compose up monkeyocr-demo`
5.  Run FastAPI: `docker compose up monkeyocr-api`

**Note:** For 20/30/40-series, V100, or similar GPUs, use: `docker compose build monkeyocr-fix`

## Windows Support

See the [windows support guide](docs/windows_support.md) for details.

## Quantization

Follow the instructions in the [quantization guide](docs/Quantization.md).

## Further Reading

### Evaluation Results
*   End-to-end evaluation results of different tasks (See original README for the table).
*   End-to-end text recognition performance across 9 PDF page types (See original README for the table).
*   The evaluation results of olmOCR-bench (See original README for the table).

### Demo
Get a Quick Hands-On Experience with Our Demo:  http://vlrlabmonkey.xyz:7685

### Example Output
*   Support diverse Chinese and English PDF types (GIF Image in original README)
*   Example for formula document (image in original README)
*   Example for table document (image in original README)
*   Example for newspaper (image in original README)
*   Example for financial report (image in original README)

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

(See original README)

## Limitations

(See original README)

## Copyright

(See original README)