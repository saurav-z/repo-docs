# MonkeyOCR: Revolutionizing Document Parsing with AI

**MonkeyOCR is a cutting-edge document parsing system that utilizes a Structure-Recognition-Relation (SRR) triplet paradigm to accurately extract and structure information from complex documents.**  [Access the original repository here](https://github.com/Yuliang-Liu/MonkeyOCR).

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

## Key Features

*   **SRR Paradigm:** Simplifies multi-tool pipelines, avoiding the inefficiencies of large multimodal models for full-page document processing.
*   **Superior Accuracy:** Outperforms pipeline-based and end-to-end models, especially on complex documents.
*   **High Speed:** Processes multi-page documents rapidly, exceeding the speed of comparable models.
*   **Chinese and English Support:** Optimized for both Chinese and English document parsing.
*   **Open Source and Accessible:** Provides model weights, demos, and detailed usage instructions.

## Performance Highlights

*   **Improved Accuracy:** MonkeyOCR achieves an average improvement of 5.1% across nine document types compared to MinerU, including a 15.0% gain on formulas and an 8.6% gain on tables.
*   **State-of-the-Art Results:** The 3B-parameter model achieves the best average performance on English documents, outperforming models such as Gemini 2.5 Pro and Qwen2.5 VL-72B.
*   **Exceptional Speed:** Processes multi-page documents at 0.84 pages per second, surpassing MinerU (0.65) and Qwen2.5 VL-7B (0.12).

## Quick Start

### 1. Installation

*   Follow the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda.md#install-with-cuda-support) to set up your environment.

### 2. Model Weights Download

*   Download the model from Hugging Face:

    ```bash
    pip install huggingface_hub
    python tools/download_model.py
    ```

*   Alternatively, download the model from ModelScope:

    ```bash
    pip install modelscope
    python tools/download_model.py -t modelscope
    ```

### 3. Inference

*   Parse documents using the following commands:

    ```bash
    # Parse a single file
    python parse.py input.pdf
    ```

    ```bash
    # Parse a directory
    python parse.py /path/to/your/documents/
    ```

    ```bash
    # See more examples
    python parse.py -h
    ```

### 4. Gradio Demo

*   Launch the interactive demo:

    ```bash
    python demo/demo_gradio.py
    ```

### 5. Fast API

*   Start the FastAPI service:

    ```bash
    uvicorn api.main:app --port 8000
    ```

## Docker Deployment

Detailed instructions for deploying MonkeyOCR with Docker can be found in the [Docker deployment guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docker/README.md).

## Quantization

*   This model can be quantized using AWQ. Follow the instructions in the [Quantization guide](docs/Quantization.md).

## Benchmark Results

Detailed benchmark results can be found [here](https://github.com/Yuliang-Liu/MonkeyOCR#benchmark-results).

## Visualization Demo

Experience MonkeyOCR in action through our interactive demo:  http://vlrlabmonkey.xyz:7685

## Citing MonkeyOCR

If you use MonkeyOCR in your research, please cite us:

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

We are grateful to the following projects for their contributions: [MinerU](https://github.com/opendatalab/MinerU), [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO), [PyMuPDF](https://github.com/pymupdf/PyMuPDF), [layoutreader](https://github.com/ppaanngggg/layoutreader), [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [LMDeploy](https://github.com/InternLM/lmdeploy), [PP-StructureV3](https://github.com/PaddlePaddle/PaddleOCR), [PP-DocLayout_plus-L](https://huggingface.co/PaddlePaddle/PP-DocLayout_plus-L) and [InternVL3](https://github.com/OpenGVLab/InternVL), [M6Doc](https://github.com/HCIILAB/M6Doc), [DocLayNet](https://github.com/DS4SD/DocLayNet), [CDLA](https://github.com/buptlihang/CDLA), [D4LA](https://github.com/AlibabaResearch/AdvancedLiterateMachinery), [DocGenome](https://github.com/Alpha-Innovator/DocGenome), [PubTabNet](https://github.com/ibm-aur-nlp/PubTabNet), and [UniMER-1M](https://github.com/opendatalab/UniMERNet).

## Alternative Models to Explore

*   [PP-StructureV3](https://github.com/PaddlePaddle/PaddleOCR)
*   [MinerU 2.0](https://github.com/opendatalab/mineru)

## Copyright

This model is intended for non-commercial use. For commercial inquiries, please contact xbai@hust.edu.cn or ylliu@hust.edu.cn.