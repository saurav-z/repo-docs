# MonkeyOCR: Effortless Document Parsing with AI (and lightning-fast speeds!)

**MonkeyOCR** is a powerful document parsing solution that utilizes a Structure-Recognition-Relation (SRR) triplet paradigm for accurate and efficient document understanding.  It simplifies the document processing pipeline and achieves impressive performance compared to other approaches.  Check out the original repo for more details: [Yuliang-Liu/MonkeyOCR](https://github.com/Yuliang-Liu/MonkeyOCR)

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

Key Features:

*   **SRR Paradigm:** Utilizes a Structure-Recognition-Relation triplet for efficient document processing.
*   **High Accuracy:** Achieves state-of-the-art results on various document parsing tasks.
*   **Blazing Speed:** Offers impressive inference speeds, especially with the MonkeyOCR-pro-1.2B model.
*   **Versatile:** Supports parsing of various document types, including PDFs and images.
*   **Easy to Use:** Provides a simple command-line interface and a user-friendly Gradio demo.
*   **Hardware Compatibility:** Optimized for a range of GPUs, including 3090, 4090, A6000, H800, and others.
*   **Quantization Support:** Can be quantized using AWQ for even greater efficiency.

## Key Improvements in MonkeyOCR-pro-1.2B

*   **Superior Performance:** Outperforms the original MonkeyOCR-3B on Chinese documents.
*   **Faster Inference:** Offers a significant speed boost compared to MonkeyOCR-pro-3B, with minimal performance drop.
*   **Competitive Benchmarking:** Outperforms Nanonets-OCR-3B on olmOCR-Bench.
*   **Robust on OmniDocBench:** MonkeyOCR-pro-3B achieves the best overall performance on both English and Chinese documents.

## Performance Benchmarks

*   **OmniDocBench Results:** MonkeyOCR exhibits excellent performance compared to other pipeline tools and expert VLMs, including GPT4o, Qwen2.5-VL, and InternVL3-8B, in tasks such as Overall, Text, Formula, Table, and Read Order evaluations. See the detailed benchmarks for more data.

    *   **MonkeyOCR-pro-3B**
        *   English: 0.138
        *   Chinese: 0.206

    *   **MonkeyOCR-pro-1.2B**
        *   English: 0.153
        *   Chinese: 0.223
*   **Text Recognition Performance:**  MonkeyOCR achieves state-of-the-art results across nine different PDF page types.
*   **olmOCR-Bench Results:** MonkeyOCR-pro-3B achieves top overall scores, showing excellent performance in tasks such as ArXiv, Old Scans Math, Tables, and more.

## Inference Speed

MonkeyOCR models offer impressive inference speeds on various GPUs.

*   **Inference Speed (Pages/s):**
    *   Test results are available for both MonkeyOCR-pro-3B and MonkeyOCR-pro-1.2B models on different GPUs (3090, A6000, H800, 4090) and various page counts (50, 100, 300, 500, 1000 pages).
*   **VLM OCR Speed (Pages/s):**
    *   Test results are available for both MonkeyOCR-pro-3B and MonkeyOCR-pro-1.2B models on different GPUs (3090, A6000, H800, 4090) and various page counts (50, 100, 300, 500, 1000 pages).

## Quick Start

Get started with MonkeyOCR in a few simple steps:

### 1. Installation
See the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) to set up your environment.

### 2. Download Model Weights
```bash
pip install huggingface_hub
python tools/download_model.py -n MonkeyOCR  # or MonkeyOCR-pro-1.2B
```
You can also download our model from ModelScope.
```bash
pip install modelscope
python tools/download_model.py -t modelscope -n MonkeyOCR  # or MonkeyOCR-pro-1.2B
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
See additional examples in the original README for more detailed usage.

### 4. Gradio Demo
```bash
python demo/demo_gradio.py
```
Access the demo at http://localhost:7860.

### 5. Fast API
```bash
uvicorn api.main:app --port 8000
```
Access the API documentation at http://localhost:8000/docs

## Docker Deployment

Follow the instructions in the original README for Docker setup.

## Windows Support

See the [windows support guide](docs/windows_support.md) for details.

## Quantization

This model can be quantized using AWQ. Follow the instructions in the [quantization guide](docs/Quantization.md).

## Visualization Demo

Try out the interactive demo at: http://vlrlabmonkey.xyz:7685

*   Upload PDFs or Images
*   Parse with structure detection, content recognition, and relationship prediction
*   Select prompts for testing
*   Get results in Markdown format

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

[List of Limitations]

## Copyright

[Copyright Notice]