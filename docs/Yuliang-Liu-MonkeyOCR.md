# MonkeyOCR: The Ultimate Document Parsing Solution

**Unlock the power of intelligent document processing with MonkeyOCR, a cutting-edge system leveraging a Structure-Recognition-Relation triplet paradigm for superior accuracy and efficiency.** [Explore the original repo](https://github.com/Yuliang-Liu/MonkeyOCR).

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

## Key Features

*   **Superior Accuracy:** Achieve state-of-the-art results, surpassing even leading closed-source and open-source models on the OmniDocBench benchmark.
*   **High Efficiency:**  Experience significant speed improvements with MonkeyOCR-pro-1.2B, offering a 36% speed boost over the 3B version, with minimal performance trade-off.
*   **Structure-Recognition-Relation (SRR) Paradigm:**  Benefit from a streamlined architecture that simplifies document processing compared to modular approaches.
*   **Versatile Deployment:** Easily deploy MonkeyOCR via local installation, Docker, or FastAPI.  Quantization is also supported for optimized performance.
*   **Comprehensive Output:**  Generate markdown files, layout results, and intermediate block results, providing flexibility for various applications.

## Performance Highlights

*   **MonkeyOCR-pro-1.2B** outperforms MonkeyOCR-3B by 7.4% on Chinese documents.
*   **MonkeyOCR-pro-1.2B** outperforms Nanonets-OCR-3B by 7.3% on olmOCR-Bench.
*   **MonkeyOCR-pro-3B** achieves the best overall performance on both English and Chinese documents on OmniDocBench, outperforming competitors like Gemini 2.0-Flash, Gemini 2.5-Pro, Qwen2.5-VL-72B, GPT-4o, and InternVL3-78B.

### Benchmark Results

Comprehensive benchmark results showcasing MonkeyOCR's performance across various metrics and datasets are available below in the original README.

## Quick Start

### 1.  Installation

Follow the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) to set up your environment.

### 2. Download Model Weights

Choose your preferred method:

**Hugging Face:**

```bash
pip install huggingface_hub
python tools/download_model.py -n MonkeyOCR-pro-3B  # or MonkeyOCR
```

**ModelScope:**

```bash
pip install modelscope
python tools/download_model.py -t modelscope -n MonkeyOCR-pro-3B  # or MonkeyOCR
```

### 3. Inference

Use the `parse.py` script for document parsing.

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

For detailed usage examples and output details, see the original README.

### 4. Gradio Demo

Run the interactive demo:

```bash
python demo/demo_gradio.py
```

Access the demo at `http://localhost:7860`.

### 5. FastAPI

Start the API service:

```bash
uvicorn api.main:app --port 8000
```

Access API documentation at `http://localhost:8000/docs`.

## Docker Deployment

Detailed instructions for deploying MonkeyOCR with Docker are available in the original README.

## Quantization

Quantization instructions are provided in the [quantization guide](docs/Quantization.md) in the original repo.

## Citing MonkeyOCR

To cite MonkeyOCR, please use the following BibTeX entry:

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

We would like to thank [MinerU](https://github.com/opendatalab/MinerU), [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO), [PyMuPDF](https://github.com/pymupdf/PyMuPDF), [layoutreader](https://github.com/ppaanngggg/layoutreader), [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [LMDeploy](https://github.com/InternLM/lmdeploy), [PP-StructureV3](https://github.com/PaddlePaddle/PaddleOCR), [PP-DocLayout_plus-L](https://huggingface.co/PaddlePaddle/PP-DocLayout_plus-L), and [InternVL3](https://github.com/OpenGVLab/InternVL) for providing base code and models, as well as their contributions to this field. We also thank [M6Doc](https://github.com/HCIILAB/M6Doc), [DocLayNet](https://github.com/DS4SD/DocLayNet), [CDLA](https://github.com/buptlihang/CDLA), [D4LA](https://github.com/AlibabaResearch/AdvancedLiterateMachinery), [DocGenome](https://github.com/Alpha-Innovator/DocGenome), [PubTabNet](https://github.com/ibm-aur-nlp/PubTabNet), and [UniMER-1M](https://github.com/opendatalab/UniMERNet) for providing valuable datasets. We also thank everyone who contributed to this open-source effort.

## Limitations

*   Limited support for photographed text, handwritten content, Traditional Chinese characters, or multilingual text in the current release.
*   Demo server may experience delays during high traffic.
*   Processing time shown on the demo page includes overhead.

## Copyright

This model is intended for academic research and non-commercial use only. For inquiries about faster (smaller) or stronger models, please contact xbai@hust.edu.cn or ylliu@hust.edu.cn.