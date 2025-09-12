# MonkeyOCR: Unlock the Power of Document Parsing with State-of-the-Art Accuracy 

**MonkeyOCR** is a cutting-edge document parsing tool, using a Structure-Recognition-Relation (SRR) triplet paradigm to extract and organize information, offering superior performance compared to modular approaches and large multimodal models. Check out the original repo [here](https://github.com/Yuliang-Liu/MonkeyOCR).

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR-pro-3B)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

## Key Features

*   **Superior Accuracy:** MonkeyOCR-pro-1.2B surpasses MonkeyOCR-3B by 7.4% on Chinese documents.
*   **Enhanced Speed:** Achieve approximately a 36% speed improvement with MonkeyOCR-pro-1.2B.
*   **Leading Performance:** MonkeyOCR-pro-1.2B outperforms Nanonets-OCR-3B by 7.3% on olmOCR-Bench.
*   **Top-Tier Results:** MonkeyOCR-pro-3B achieves the best overall performance on both English and Chinese documents on OmniDocBench, exceeding even closed-source and extra-large open-source VLMs.
*   **SRR Triplet Paradigm:** Simplifies document parsing and avoids the inefficiency of large multimodal models.
*   **Comprehensive Output:** Generates markdown files, layout results, and intermediate block results.
*   **Easy Deployment:** Supports local installation, Gradio demo, FastAPI service, and Docker deployment.
*   **Quantization Support:** Compatible with AWQ for model quantization.
*   **Extensive Hardware Support:** Optimized for a wide range of GPUs, including 3090, 4090, A6000, H800, A100, 4060, and more.

## Performance Highlights

### Benchmark Results on OmniDocBench

[Image of Benchmark results - Replace with the original image link]

### Benchmark Results on OLMOCR-Bench

[Table with OLMOCR Bench results]

## Quick Start Guide

### 1.  Installation

   *   Follow the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) to set up your environment.

### 2.  Download Model Weights

   *   **Hugging Face:**
        ```bash
        pip install huggingface_hub
        python tools/download_model.py -n MonkeyOCR-pro-3B  # or MonkeyOCR
        ```
    *   **ModelScope:**
        ```bash
        pip install modelscope
        python tools/download_model.py -t modelscope -n MonkeyOCR-pro-3B  # or MonkeyOCR
        ```

### 3.  Inference

   *   Use the `parse.py` script to process PDFs or images:

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

   *   For more examples, refer to the "More usage examples" in the original README.

### 4.  Gradio Demo

   *   Run the Gradio demo:
       ```bash
       python demo/demo_gradio.py
       ```
       Access the demo at http://localhost:7860.

### 5.  FastAPI

   *   Start the FastAPI service:
       ```bash
       uvicorn api.main:app --port 8000
       ```
       Access the API documentation at http://localhost:8000/docs.

## Docker Deployment

1.  Navigate to the `docker` directory:

    ```bash
    cd docker
    ```

2.  Ensure NVIDIA GPU support is available in Docker (via `nvidia-docker2`). Run `bash env.sh` if it isn't.

3.  Build the Docker image:

    ```bash
    docker compose build monkeyocr
    ```

4.  For 20/30/40-series GPUs, build the patched image:

    ```bash
    docker compose build monkeyocr-fix
    ```

5.  Run the container with the Gradio demo (accessible on port 7860):

    ```bash
    docker compose up monkeyocr-demo
    ```

6.  Run the FastAPI service (accessible on port 7861):

    ```bash
    docker compose up monkeyocr-api
    ```

## Windows Support

Refer to the [windows support guide](docs/windows_support.md).

## Quantization

Follow the [quantization guide](docs/Quantization.md) for AWQ quantization.

## Visualization Demo

Experience MonkeyOCR's capabilities via a live demo:  http://vlrlabmonkey.xyz:7685

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

(Keep the original acknowledgments)

## Limitation

(Keep the original limitation)

## Copyright

(Keep the original copyright)