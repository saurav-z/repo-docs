# MonkeyOCR: Unleash the Power of Document Parsing with Advanced AI 

**Effortlessly extract and understand information from documents with MonkeyOCR, a state-of-the-art document parsing solution.** Explore the project on [GitHub](https://github.com/Yuliang-Liu/MonkeyOCR).

<div align="center">
    <a href="https://arxiv.org/abs/2506.05218">
        <img src="https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv" alt="arXiv"/>
    </a>
    <a href="https://huggingface.co/echo840/MonkeyOCR">
        <img src="https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace" alt="Hugging Face Weights"/>
    </a>
    <a href="https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue">
        <img src="https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues" alt="GitHub Issues"/>
    </a>
    <a href="https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed">
        <img src="https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues" alt="GitHub Closed Issues"/>
    </a>
    <a href="https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt">
        <img src="https://img.shields.io/badge/License-Apache%202.0-yellow" alt="License"/>
    </a>
    <a href="https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views">
        <img src="https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views" alt="GitHub Views"/>
    </a>
</div>

*   **Structure-Recognition-Relation (SRR) Paradigm:** MonkeyOCR employs a novel SRR triplet paradigm, simplifying the document parsing pipeline.
*   **Superior Performance:** Outperforms leading closed-source and open-source solutions, excelling on both English and Chinese documents.
*   **Fast & Efficient:** Offers significant speed improvements with minimal performance trade-off, especially with MonkeyOCR-pro-1.2B.
*   **Versatile Hardware Support:** Supports various GPUs, including 3090, 4090, A6000, H800, and 4060 (quantized).
*   **Easy to Use:** Includes straightforward installation, demo applications, and API integration.

## Key Features

*   **Advanced Document Parsing:** Accurately extracts text, tables, formulas, and more from documents.
*   **Multi-Format Support:** Processes PDFs, images, and other document formats.
*   **State-of-the-Art Accuracy:** Achieves top performance on multiple benchmarks, surpassing existing solutions.
*   **Fast Inference Speeds:** Offers high-speed document processing on various GPUs.
*   **Flexible Deployment:** Provides local installation, Docker deployment, and API access.
*   **Quantization Support:** Offers options for model quantization (AWQ) for optimized resource usage.
*   **Gradio Demo:** Includes an interactive demo for quick hands-on experience, showcasing diverse Chinese and English PDF types.
*   **FastAPI Integration:** MonkeyOCR features a service based on FastAPI, enabling seamless integration and accessibility.

## Quick Start

### 1. Installation

Follow the detailed [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) to set up your environment.

### 2. Model Download

Download the model weights from Hugging Face using:

```bash
pip install huggingface_hub
python tools/download_model.py -n MonkeyOCR  # or MonkeyOCR-pro-1.2B
```

Alternatively, download the model from ModelScope:

```bash
pip install modelscope
python tools/download_model.py -t modelscope -n MonkeyOCR  # or MonkeyOCR-pro-1.2B
```

### 3. Inference

Run the parsing using the following commands:

```bash
# Replace input_path with the path to a PDF, image, or directory

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
Access the demo at http://localhost:7860 to start using MonkeyOCR.

### 5. Fast API
Start the service with the following:
```bash
uvicorn api.main:app --port 8000
```
Then check the API documentation at http://localhost:8000/docs.

## Docker Deployment

1.  Navigate to the `docker` directory:

    ```bash
    cd docker
    ```

2.  Build the Docker image (ensure NVIDIA GPU support is set up).

    ```bash
    docker compose build monkeyocr
    ```
    For 20/30/40-series GPUs or V100, use this instead:
    ```bash
    docker compose build monkeyocr-fix
    ```

3.  Run Gradio Demo:

    ```bash
    docker compose up monkeyocr-demo
    ```

    Or start an interactive development environment:

    ```bash
    docker compose run --rm monkeyocr-dev
    ```

4.  Run FastAPI service:
    ```bash
    docker compose up monkeyocr-api
    ```
    Check API documentation at http://localhost:7861/docs.

## Quantization

Follow the [quantization guide](docs/Quantization.md) to quantize the model using AWQ.

## Benchmark Results

MonkeyOCR demonstrates competitive performance on standard document parsing benchmarks such as OmniDocBench and OLMOCR-Bench. The model outperforms other state of the art solutions, which you can examine in the benchmark tables above.

### 1. OmniDocBench results
See the detailed results of MonkeyOCR models with pipeline tools, expert VLMs, general VLMs, and more.

### 2. Text recognition results
See the text recognition performance of multiple models, categorized by document type.

### 3. OLMOCR-bench results
See the performance results comparing MonkeyOCR to alternative solutions.

## Visualization Demo

Experience the capabilities of MonkeyOCR with our interactive demo: [http://vlrlabmonkey.xyz:7685](http://vlrlabmonkey.xyz:7685)

**Key features:**

*   Upload PDFs or images.
*   Click "Parse" to extract structure and content.
*   View results in markdown format.
*   Select prompts for specific recognition tasks.

### Example
<img src="asserts/Visualization.GIF?raw=true" width="600"/>

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

We thank the contributors and open-source projects such as MinerU, DocLayout-YOLO, PyMuPDF, Qwen2.5-VL, LMDeploy, and PP-StructureV3. We also thank the providers of datasets and other valuable contributions.

## Limitations

MonkeyOCR currently has limitations, including non-support for handwritten content, Traditional Chinese characters, and multilingual text.

## Copyright

Our model is for academic research and non-commercial use. Contact xbai@hust.edu.cn or ylliu@hust.edu.cn for faster or stronger models.