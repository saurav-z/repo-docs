# MonkeyOCR: Unleash the Power of Accurate Document Parsing with AI

**MonkeyOCR is a cutting-edge AI model revolutionizing document parsing by using a Structure-Recognition-Relation (SRR) triplet paradigm to efficiently extract information from documents.**  [View the original repository](https://github.com/Yuliang-Liu/MonkeyOCR).

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

MonkeyOCR streamlines document processing, delivering superior accuracy and speed compared to modular approaches, particularly with large multimodal models. Key features include:

*   **Enhanced Accuracy:**  MonkeyOCR-pro-1.2B outperforms MonkeyOCR-3B by 7.4% on Chinese documents, and also excels on the olmOCR-Bench and OmniDocBench.
*   **Superior Speed:** MonkeyOCR-pro-1.2B offers a 36% speed improvement over MonkeyOCR-pro-3B, with only a marginal performance decrease.
*   **State-of-the-Art Performance:** MonkeyOCR-pro-3B achieves top-tier results on OmniDocBench, surpassing even closed-source and extra-large open-source VLMs such as Gemini 2.0-Flash, Gemini 2.5-Pro, Qwen2.5-VL-72B, GPT-4o, and InternVL3-78B.
*   **Easy Deployment:** Supports various hardware configurations, including 3090, 4090, A6000, H800, and more, and offers a convenient Docker deployment option.
*   **Comprehensive Results:** Generates Markdown files, layout results (PDFs), and detailed JSON files with block information for thorough analysis and customization.

## Key Features

*   **Advanced Document Structure Recognition:** Accurately identifies and extracts text, formulas, tables, and other elements.
*   **Efficient Information Extraction:** SRR paradigm simplifies pipelines and avoids inefficiency of large multimodal models for full-page document processing.
*   **Multiple Model Options:** Offers MonkeyOCR-pro-3B and MonkeyOCR-pro-1.2B models, allowing flexibility in performance and speed.
*   **Versatile Usage:** Supports single-task recognition, batch processing, and page splitting for diverse use cases.
*   **User-Friendly Demo:** Interactive Gradio demo for easy document parsing and testing, including a simple API.
*   **Quantization Support:** Utilizes AWQ for model quantization.
*   **FastAPI Integration:** Includes FastAPI service for API documentation and access.

## Benchmark Results

**[Table of End-to-End Results]**

**[Table of End-to-End Text Recognition Performance]**

**[Table of olmOCR-bench Results]**

## Quick Start

### 1. Installation

See the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) to set up your environment.

### 2. Download Model Weights

```python
pip install huggingface_hub

python tools/download_model.py -n MonkeyOCR-pro-3B  # or MonkeyOCR
```

You can also download our model from ModelScope.

```python
pip install modelscope

python tools/download_model.py -t modelscope -n MonkeyOCR-pro-3B  # or MonkeyOCR
```

### 3. Inference

```bash
# Single file processing
python parse.py input.pdf
python parse.py image.jpg

# Single task recognition
python parse.py image.jpg -t text
python parse.py document.pdf -t text

# Folder processing
python parse.py /path/to/folder
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
Access the API documentation at http://localhost:8000/docs

## Docker Deployment

Follow the instructions in the [Docker Deployment section] to set up and use the Docker image.

## Windows Support

See the [windows support guide](docs/windows_support.md) for details.

## Quantization

This model can be quantized using AWQ. Follow the instructions in the [quantization guide](docs/Quantization.md).

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

[Acknowledgments section, include links as appropriate]

## Limitations

*   [Limitations section, including lack of support for photographed text, handwritten content, etc.]
*   [Single GPU limitation]
*   [Processing time details for demo use]

## Copyright

[Copyright and Contact information]