# MonkeyOCR: Unleash the Power of Document Parsing with SRR Triplet Paradigm

[MonkeyOCR](https://github.com/Yuliang-Liu/MonkeyOCR) is a cutting-edge document parsing system that leverages a Structure-Recognition-Relation (SRR) triplet paradigm to efficiently and accurately extract information from various document types.

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

Key features:

*   **Superior Performance:** MonkeyOCR-pro-1.2B surpasses MonkeyOCR-3B by 7.4% on Chinese documents and outperforms Nanonets-OCR-3B by 7.3% on olmOCR-Bench.
*   **High Speed & Efficiency:** MonkeyOCR-pro-1.2B delivers approximately a 36% speed improvement over MonkeyOCR-pro-3B, with approximately 1.6% drop in performance.
*   **State-of-the-Art Results:** MonkeyOCR-pro-3B achieves the best overall performance on both English and Chinese documents on OmniDocBench, surpassing even closed-source and large open-source VLMs.
*   **Flexible Deployment:** Supports various hardware, including 3090, 4090, A6000, H800, and more, with easy-to-use Docker and API deployment options.
*   **Comprehensive Output:** Generates Markdown files, layout results, and intermediate JSON files for detailed analysis.

## Key Results & Benchmarks

### OmniDocBench Performance

MonkeyOCR excels in various document parsing tasks, consistently outperforming existing pipeline tools and even leading VLM models.

**Overall End-to-End Performance:**

| Model                         | Overall (EN) | Overall (ZH) |
| ----------------------------- | ------------ | ------------ |
| **MonkeyOCR-pro-3B (Demo)**   | **0.138**    | **0.206**    |
| **MonkeyOCR-pro-1.2B**        | 0.153        | 0.223        |

**Text Recognition Performance:**

| Model                       | Overall |
| --------------------------- | ------- |
| **MonkeyOCR-pro-3B (Demo)** | **0.100** |
| **MonkeyOCR-pro-1.2B**      | 0.112   |

### omlOCR-bench

MonkeyOCR achieves leading results in omlOCR-bench, showcasing its ability to excel across various document parsing challenges.

**Overall omlOCR-bench Performance:**

| Model                         | Overall         |
| ----------------------------- | --------------- |
| **MonkeyOCR-pro-3B (Demo)**   | **75.8 ± 1.0**  |
| **MonkeyOCR-pro-1.2B**        | 71.8 ± 1.1    |

*For more comprehensive benchmark results, refer to the detailed tables in the original README.*

### Inference Speed (Pages/s)

*Refer to the original README for GPU-specific inference speed data.*

## Quick Start Guide

### 1. Install MonkeyOCR
Follow the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) to set up your environment.

### 2. Download Model Weights
Download our model from Huggingface.

```python
pip install huggingface_hub

python tools/download_model.py -n MonkeyOCR  # or MonkeyOCR-pro-1.2B
```
You can also download our model from ModelScope.

```python
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

### 4. Gradio Demo
```bash
python demo/demo_gradio.py
```
Once the demo is running, you can access it at http://localhost:7860.

### 5. Fast API
```bash
uvicorn api.main:app --port 8000
```
Once the API service is running, you can access the API documentation at http://localhost:8000/docs to explore available endpoints.

## Docker Deployment

1.  Navigate to the `docker` directory:

    ```bash
    cd docker
    ```

2.  Build the Docker image:

    ```bash
    docker compose build monkeyocr
    ```

3.  Run the container with the Gradio demo:

    ```bash
    docker compose up monkeyocr-demo
    ```

4.  Run the FastAPI service:

    ```bash
    docker compose up monkeyocr-api
    ```

## Windows Support

See the [windows support guide](docs/windows_support.md) for details.

## Quantization

This model can be quantized using AWQ. Follow the instructions in the [quantization guide](docs/Quantization.md).

## Demo

Experience the power of MonkeyOCR firsthand! Visit our interactive demo at: [http://vlrlabmonkey.xyz:7685](http://vlrlabmonkey.xyz:7685)

## Citing MonkeyOCR

If you are referencing the baseline results from this repository, please use the following BibTeX entry:
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

*Refer to the original README for full acknowledgments.*

## Limitations

*   *Refer to the original README for full limitations.*

## Copyright

*Refer to the original README for copyright information.*