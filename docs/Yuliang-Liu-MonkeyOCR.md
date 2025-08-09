# MonkeyOCR: Effortless Document Parsing with SRR Paradigm

**Unlock the power of structured document understanding with MonkeyOCR, a cutting-edge solution for intelligent document parsing, available on [GitHub](https://github.com/Yuliang-Liu/MonkeyOCR).**

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

**MonkeyOCR** employs a Structure-Recognition-Relation (SRR) triplet paradigm to provide an efficient and accurate document parsing solution. It simplifies the complexities of traditional modular approaches and large multimodal models for processing documents, enabling superior performance across a variety of document types.

## Key Features

*   **State-of-the-Art Performance:** MonkeyOCR-pro-3B achieves the best overall performance, even outperforming closed-source and extra-large open-source VLMs.
*   **Improved Efficiency & Speed:** MonkeyOCR-pro-1.2B surpasses MonkeyOCR-3B in speed, accuracy, and efficiency.
*   **Versatile Support:** Supports diverse Chinese and English document types, including books, slides, financial reports, and more.
*   **Ease of Use:** Simple installation and straightforward inference, including a Gradio demo and FastAPI service.
*   **Hardware Flexibility:** Works on a wide range of GPUs, from 3090 to H800, and even quantized models on 4060 with 8GB of VRAM.
*   **Comprehensive Output:** Generates Markdown output along with layout and intermediate results files for detailed analysis.

## Performance Highlights

*   **MonkeyOCR-pro-1.2B** outperforms MonkeyOCR-3B by 7.4% on Chinese documents.
*   **MonkeyOCR-pro-1.2B** delivers approximately a 36% speed improvement over MonkeyOCR-pro-3B.
*   **MonkeyOCR-pro-1.2B** outperforms Nanonets-OCR-3B by 7.3% on olmOCR-Bench.
*   **MonkeyOCR-pro-3B** achieves the best overall performance on OmniDocBench.

### Comparison to Other Models

| Model                       | Overall (OmniDocBench) |
| --------------------------- | ----------------------- |
| MonkeyOCR-pro-3B          | **0.138**              |
| MonkeyOCR-pro-1.2B          | 0.153              |
| **Pipeline Tools**           | 0.145 - 0.646     |
| **Expert VLMs**           | 0.139 - 0.493    |
| **General VLMs**       | 0.233 - 0.314              |
|   |   |

### Inference Speed on Different GPUs

| Model                 | GPU   | Pages/s (50 Pages) |
| --------------------- | ----- | ------------------ |
| MonkeyOCR-pro-3B      | 3090  | 0.705              |
| MonkeyOCR-pro-1.2B    | 3090  | 0.919              |
| MonkeyOCR-pro-3B      | 4090  | 1.321              |
| MonkeyOCR-pro-1.2B    | 4090  | 1.759              |
| **More in original Readme** |      |                     |

## Quick Start

### 1. Installation

See the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) to set up your environment.

### 2. Model Download

Download model weights from Hugging Face or ModelScope:

```bash
# From Hugging Face
pip install huggingface_hub
python tools/download_model.py -n MonkeyOCR  # or MonkeyOCR-pro-1.2B

# From ModelScope
pip install modelscope
python tools/download_model.py -t modelscope -n MonkeyOCR  # or MonkeyOCR-pro-1.2B
```

### 3. Inference

Use the `parse.py` script to parse documents:

```bash
python parse.py input_path  # Parses a file or a directory
```

**Explore the original [README](https://github.com/Yuliang-Liu/MonkeyOCR) for detailed usage instructions, including single-task recognition, grouping options, and output details.**

### 4. Gradio Demo

Run a user-friendly demo:

```bash
python demo/demo_gradio.py
```

Access the demo at `http://localhost:7860`.

### 5. FastAPI Service

Deploy MonkeyOCR as a RESTful API:

```bash
uvicorn api.main:app --port 8000
```

Access API documentation at `http://localhost:8000/docs`.

## Docker Deployment

See the [original readme](https://github.com/Yuliang-Liu/MonkeyOCR) for complete Docker setup instructions.

## Windows Support & Quantization

*   **Windows Support:** See the [windows support guide](docs/windows_support.md).
*   **Quantization:** See the [quantization guide](docs/Quantization.md).

## Benchmark Results

For detailed benchmark results, including evaluations on OmniDocBench and olmOCR-Bench, consult the [original README](https://github.com/Yuliang-Liu/MonkeyOCR).

## Visualization Demo

Experience MonkeyOCR in action!  http://vlrlabmonkey.xyz:7685

## Example outputs

See the example outputs in the original Readme.

## Citation

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

The authors acknowledge the contributions of various open-source projects and datasets, as detailed in the [original README](https://github.com/Yuliang-Liu/MonkeyOCR).

## Limitations

*   MonkeyOCR's current limitations are detailed in the [original README](https://github.com/Yuliang-Liu/MonkeyOCR).
*   Currently, MonkeyOCR do not yet fully support for photographed text, handwritten content, Traditional Chinese characters, or multilingual text.

## Copyright

This model is intended for academic research and non-commercial use only. Contact the authors at xbai@hust.edu.cn or ylliu@hust.edu.cn for commercial inquiries.