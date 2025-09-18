# MonkeyOCR: Effortlessly Parse Documents with Cutting-Edge AI

**Unlock the power of intelligent document processing with MonkeyOCR, a state-of-the-art AI solution for accurately extracting and understanding text, tables, formulas, and more from your documents.**  [View the original repository](https://github.com/Yuliang-Liu/MonkeyOCR)

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR-pro-3B)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

## Key Features

*   **SRR Triplet Paradigm:** MonkeyOCR employs a Structure-Recognition-Relation (SRR) triplet paradigm for efficient and accurate document parsing.
*   **Superior Performance:**  MonkeyOCR-pro-1.2B outperforms previous models in accuracy, speed, and efficiency, achieving state-of-the-art results on various benchmarks.  Notably, MonkeyOCR-pro-3B achieves the best performance, outperforming closed-source and extra-large open-source VLMs like Gemini and GPT-4o on OmniDocBench.
*   **Multi-Platform Support:** Tested and optimized for various GPU hardware, with community support extending compatibility to various GPUs.
*   **Easy Deployment:**  Supports local installation, Hugging Face and ModelScope model downloads, Gradio demo, and FastAPI deployment via Docker for flexible usage.
*   **Comprehensive Output:**  Provides processed Markdown files, layout results, and detailed intermediate block results for in-depth analysis.

## News

*   **July 10, 2025:** 🚀 Released [MonkeyOCR-pro-1.2B](https://huggingface.co/echo840/MonkeyOCR-pro-1.2B), offering improved performance and speed.
*   **June 12, 2025:** 🚀 Trending on [Hugging Face](https://huggingface.co/models?sort=trending).
*   **June 5, 2025:** 🚀 Released [MonkeyOCR](https://huggingface.co/echo840/MonkeyOCR), an English and Chinese document parsing model.

## Quick Start

### 1.  Installation
   *   Follow the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) to set up your environment.

### 2.  Model Download
   *   Download the model from Hugging Face:
      ```bash
      pip install huggingface_hub
      python tools/download_model.py -n MonkeyOCR-pro-3B  # or MonkeyOCR
      ```
   *   Alternatively, download from ModelScope:
      ```bash
      pip install modelscope
      python tools/download_model.py -t modelscope -n MonkeyOCR-pro-3B  # or MonkeyOCR
      ```

### 3.  Inference
   *   Use the `parse.py` script to process documents:
      ```bash
      python parse.py input_path
      ```
     *   See the original repository for more detailed usage examples.

### 4.  Gradio Demo
   *   Run the Gradio demo for a user-friendly experience:
      ```bash
      python demo/demo_gradio.py
      ```
      *   Access the demo at http://localhost:7860.

### 5.  FastAPI Deployment
   *   Start the FastAPI service:
      ```bash
      uvicorn api.main:app --port 8000
      ```
      *   Access API documentation at http://localhost:8000/docs.

## Docker Deployment

Follow the instructions in the original README to deploy with Docker, including instructions for GPU support and LMDeploy compatibility if required.

## Windows Support

Refer to the [windows support guide](docs/windows_support.md) for Windows-specific instructions.

## Quantization

Follow the [quantization guide](docs/Quantization.md) to quantize the model using AWQ.

## Benchmark Results

Detailed performance benchmarks are provided in the original README, including end-to-end evaluation results and the evaluation results of olmOCR-bench, highlighting the strengths of MonkeyOCR.  Key findings include:

*   MonkeyOCR-pro-3B and 1.2B achieve state-of-the-art results compared to closed and open-source baselines.

## Visualization Demo

Experience the power of MonkeyOCR with our interactive demo:  http://vlrlabmonkey.xyz:7685. The demo supports diverse Chinese and English PDF types.

## Citing MonkeyOCR

If you use MonkeyOCR in your research, please cite it using the provided BibTeX entry in the original README.

## Acknowledgments

The project team expresses gratitude to the contributors and resources mentioned in the original README.

## Limitations

The current version does not fully support photographed text, handwritten content, Traditional Chinese characters, or multilingual text.  The demo processing time may vary during high traffic. The inference speeds of MonkeyOCR, MinerU, and Qwen2.5 VL-7B were measured on an H800 GPU.

## Copyright

MonkeyOCR is intended for academic research and non-commercial use. Contact the authors for inquiries.