# MonkeyOCR: Intelligent Document Parsing with Structure-Recognition-Relation

**Unlock the power of structured document understanding with MonkeyOCR, a cutting-edge model leveraging a Structure-Recognition-Relation (SRR) triplet paradigm.  [Explore the original repo](https://github.com/Yuliang-Liu/MonkeyOCR)!**

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

MonkeyOCR streamlines document processing by effectively recognizing and understanding document structure. This approach simplifies the multi-step pipelines of other modular systems.

**Key Features:**

*   **Superior Performance:** MonkeyOCR-pro-1.2B surpasses MonkeyOCR-3B by 7.4% on Chinese documents, excelling against leading open and closed-source models.
*   **Enhanced Speed & Efficiency:** MonkeyOCR-pro-1.2B offers approximately a 36% speed boost over MonkeyOCR-pro-3B, with a minimal performance trade-off.
*   **State-of-the-Art Results:** On OmniDocBench, MonkeyOCR-pro-3B achieves top-tier results on both English and Chinese documents, outperforming even cutting-edge VLMs.
*   **Versatile Hardware Support:**  Compatible with a wide range of GPUs including 3090, 4090, A6000, H800, and others, along with community support for various hardware.
*   **Easy to Use:**  Includes a Gradio demo and Fast API, making experimentation and integration straightforward.

## Quick Start

### 1.  Installation
Follow the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) to set up your environment.

### 2.  Model Download
Download the model weights using either Hugging Face or ModelScope:
```bash
pip install huggingface_hub
python tools/download_model.py -n MonkeyOCR  # or MonkeyOCR-pro-1.2B

pip install modelscope
python tools/download_model.py -t modelscope -n MonkeyOCR  # or MonkeyOCR-pro-1.2B
```
### 3.  Inference
Run inference on your documents using the following commands:
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
For more usage examples please see the original README

### 4.  Gradio Demo
Launch the interactive demo:
```bash
python demo/demo_gradio.py
```
Access the demo at `http://localhost:7860`.

### 5.  FastAPI
Start the FastAPI service:
```bash
uvicorn api.main:app --port 8000
```
Explore the API documentation at `http://localhost:8000/docs`.

## Docker Deployment

Detailed instructions are available in the original README.  See the original README for details.

## Quantization

See the [quantization guide](docs/Quantization.md) for information on quantizing the model.

## Benchmark Results

See the original README for the benchmark results.

## Visualization Demo
(See the original README for details.)

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
(See the original README for details.)

## Limitation
(See the original README for details.)

## Copyright
(See the original README for details.)