# MonkeyOCR: Revolutionizing Document Parsing with the SRR Triplet Paradigm

**MonkeyOCR** simplifies document parsing, offering superior performance and efficiency. Find the original repository [here](https://github.com/Yuliang-Liu/MonkeyOCR).

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

> MonkeyOCR employs a Structure-Recognition-Relation (SRR) triplet paradigm, outperforming even closed-source and extra-large open-source VLMs in document processing.

## Key Features

*   **Superior Performance:** MonkeyOCR-pro-1.2B outperforms the 3B version on Chinese documents by 7.4%.
*   **Enhanced Speed:** Enjoy a 36% speed boost with MonkeyOCR-pro-1.2B over MonkeyOCR-pro-3B.
*   **State-of-the-Art Results:** MonkeyOCR-pro-1.2B excels on olmOCR-Bench, and MonkeyOCR-pro-3B leads on OmniDocBench.
*   **Flexible Deployment:** Supports a wide range of hardware, including 3090, 4090, A6000, H800, A100, and 4060.
*   **Easy to Use:** Offers a simple API and Gradio demo for quick experimentation.
*   **Docker Support:** Easily deploy MonkeyOCR using Docker for convenient use.

## News
* ```2025.07.10 ``` 🚀 We release [MonkeyOCR-pro-1.2B](https://huggingface.co/echo840/MonkeyOCR-pro-1.2B), — a leaner and faster version model that outperforms our previous 3B version in accuracy, speed, and efficiency.
* ```2025.06.12 ``` 🚀 The model’s trending on [Hugging Face](https://huggingface.co/models?sort=trending). Thanks for the love!
* ```2025.06.05 ``` 🚀 We release [MonkeyOCR](https://huggingface.co/echo840/MonkeyOCR), an English and Chinese documents parsing model.

## Getting Started

### Quick Installation

1.  **Install:** Follow the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support).
2.  **Download Weights:** Get your model from Hugging Face or ModelScope.
    ```bash
    pip install huggingface_hub
    python tools/download_model.py -n MonkeyOCR  # or MonkeyOCR-pro-1.2B
    ```
    or
    ```bash
    pip install modelscope
    python tools/download_model.py -t modelscope -n MonkeyOCR  # or MonkeyOCR-pro-1.2B
    ```
3.  **Inference:** Use the provided command-line tools.
    ```bash
    python parse.py input_path
    ```

### Available Models and Results

*   **MonkeyOCR-pro-3B (Demo Available):**  Offers excellent performance.
*   **MonkeyOCR-pro-1.2B:** A faster and leaner version of the model.
*   See detailed benchmark results in the original README.

## Usage Examples
See the original README for more comprehensive examples.

## Deployment Options

*   **Gradio Demo:**  Run the demo with `python demo/demo_gradio.py` and access it at http://localhost:7860.
*   **FastAPI:** Deploy using `uvicorn api.main:app --port 8000`.  Access the API documentation at http://localhost:8000/docs.
*   **Docker:** Utilize Docker Compose to build and run containers.  See the original README for detailed instructions.

## Windows Support
See the [windows support guide](docs/windows_support.md) for details.

## Quantization
This model can be quantized using AWQ. Follow the instructions in the [quantization guide](docs/Quantization.md).

## Acknowledgments
(Same as original README)

## Limitation
(Same as original README)

## Copyright
(Same as original README)