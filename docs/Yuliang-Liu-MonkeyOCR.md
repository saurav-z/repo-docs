# MonkeyOCR: Effortless Document Parsing with Triplet Paradigm

**Unlock the power of intelligent document understanding with MonkeyOCR, a cutting-edge solution for parsing complex documents.** ([Original Repo](https://github.com/Yuliang-Liu/MonkeyOCR))

[<img src="https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv" alt="arXiv"/>](https://arxiv.org/abs/2506.05218)
[<img src="https://img.shields.io/badge/HuggingFace-black.svg?logo=HuggingFace" alt="HuggingFace"/>](https://huggingface.co/echo840/MonkeyOCR-pro-3B)
[<img src="https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues" alt="GitHub issues"/>](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[<img src="https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues" alt="GitHub closed issues"/>](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[<img src="https://img.shields.io/badge/License-Apache%202.0-yellow" alt="License"/>](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[<img src="https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views" alt="GitHub views"/>](https://github.com/Yuliang-Liu/MonkeyOCR)

## Key Features

*   **Structure-Recognition-Relation (SRR) Triplet Paradigm:**  A novel approach for efficient and accurate document parsing.
*   **Superior Performance:** MonkeyOCR-pro-3B outperforms other models in various benchmarks, excelling in both English and Chinese document processing.
*   **Faster Inference:**  MonkeyOCR-pro-1.2B offers significant speed improvements while maintaining strong accuracy.
*   **Versatile:**  Supports a wide range of document types, including PDFs, images, and various layouts.
*   **Easy to Use:**  Provides clear instructions and a Gradio demo for quick start and experimentation.

## What's New

*   **2025.07.10:** 🚀 [MonkeyOCR-pro-1.2B](https://huggingface.co/echo840/MonkeyOCR-pro-1.2B) released, a faster and leaner version.
*   **2025.06.12:** 🚀 Trending on [Hugging Face](https://huggingface.co/models?sort=trending).
*   **2025.06.05:** 🚀 Initial release of [MonkeyOCR](https://huggingface.co/echo840/MonkeyOCR).

## Quickstart

### 1. Installation

Follow the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) for environment setup.

### 2. Model Download

Download models from Hugging Face or ModelScope:

```bash
# Hugging Face
pip install huggingface_hub
python tools/download_model.py -n MonkeyOCR-pro-3B  # or MonkeyOCR

# ModelScope
pip install modelscope
python tools/download_model.py -t modelscope -n MonkeyOCR-pro-3B  # or MonkeyOCR
```

### 3. Inference

Use the `parse.py` script:

```bash
# Basic parsing
python parse.py input.pdf

# Advanced options (see details below)
python parse.py input.pdf -o ./output -c config.yaml
```

**More Usage Examples:**

*   Single file processing
    ```bash
    python parse.py input.pdf                           # Parse single PDF file
    python parse.py input.pdf -o ./output               # Parse with custom output dir
    python parse.py input.pdf -s                        # Parse PDF with page splitting
    python parse.py image.jpg                           # Parse single image file
    ```
*   Single task recognition
    ```bash
    python parse.py image.jpg -t text                   # Text recognition from image
    python parse.py image.jpg -t formula                # Formula recognition from image
    python parse.py image.jpg -t table                  # Table recognition from image
    python parse.py document.pdf -t text                # Text recognition from all PDF pages
    ```
*   Folder processing (all files individually)
    ```bash
    python parse.py /path/to/folder                     # Parse all files in folder
    python parse.py /path/to/folder -s                  # Parse with page splitting
    python parse.py /path/to/folder -t text             # Single task recognition for all files
    ```
*   Multi-file grouping (batch processing by page count)
    ```bash
    python parse.py /path/to/folder -g 5                # Group files with max 5 total pages
    python parse.py /path/to/folder -g 10 -s            # Group files with page splitting
    python parse.py /path/to/folder -g 8 -t text        # Group files for single task recognition
    ```
*   Advanced configurations
    ```bash
    python parse.py input.pdf -c model_configs.yaml     # Custom model configuration
    python parse.py /path/to/folder -g 15 -s -o ./out   # Group files, split pages, custom output
    python parse.py input.pdf --pred-abandon            # Enable predicting abandon elements
    python parse.py /path/to/folder -g 10 -m            # Group files and merge text blocks in output
    ```

**Output Results:**

MonkeyOCR generates:

1.  Processed Markdown File (`your.md`)
2.  Layout Results (`your_layout.pdf`)
3.  Intermediate Block Results (`your_middle.json`)

### 4. Gradio Demo

Run the demo:

```bash
python demo/demo_gradio.py
```

Access the demo at http://localhost:7860.

### 5. FastAPI

Start the API service:

```bash
uvicorn api.main:app --port 8000
```

API documentation is available at http://localhost:8000/docs.  Consider `vllm_async` for improved concurrency.

## Docker Deployment

1.  `cd docker`
2.  Run `bash env.sh` (if NVIDIA GPU support not already enabled)
3.  Build the Docker image: `docker compose build monkeyocr`

    *   **Important:**  For 20/30/40-series, V100, L20/L40 GPUs, build the patched image: `docker compose build monkeyocr-fix`.
4.  Run the Gradio demo: `docker compose up monkeyocr-demo`
5.  Run the FastAPI service: `docker compose up monkeyocr-api`

## Windows Support

See the [windows support guide](docs/windows_support.md).

## Quantization

Quantize the model using AWQ. Follow the [quantization guide](docs/Quantization.md).

## Benchmarks

### End-to-End Evaluation Results

*(Tables from the original README, showing performance comparisons.)*

### End-to-end text recognition performance across 9 PDF page types.

*(Tables from the original README, showing performance comparisons.)*

### The evaluation results of olmOCR-bench.

*(Tables from the original README, showing performance comparisons.)*

## Visualization Demo

**Experience MonkeyOCR in action: http://vlrlabmonkey.xyz:7685**

*(Images from the original README, showing different document types.)*

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

*(List of acknowledgments from the original README.)*

## Limitations

*(List of limitations from the original README.)*

## Copyright

*(Copyright information from the original README.)*