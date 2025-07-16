<div align="center">
  <h1>MonkeyOCR: Unleash the Power of AI for Document Parsing</h1>
  <p><em>Effortlessly extract information from documents with our innovative Structure-Recognition-Relation (SRR) triplet paradigm.</em></p>

  [![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
  [![HuggingFace Weights](https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR)
  [![GitHub Issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
  [![GitHub Closed Issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
  [![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
  [![GitHub Views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)
</div>

> **MonkeyOCR** offers a revolutionary approach to document parsing, leveraging a Structure-Recognition-Relation (SRR) triplet paradigm for superior performance and efficiency. 

## Key Features

*   **Enhanced Accuracy**: Outperforms leading solutions on both Chinese and English documents.
*   **Blazing Fast**: Significantly faster inference speeds compared to modular approaches and large multimodal models.
*   **Versatile**: Handles a wide range of document types, including PDFs, images, formulas, tables, and more.
*   **Efficient**: MonkeyOCR-pro-1.2B demonstrates a 36% speed improvement over MonkeyOCR-pro-3B.
*   **State-of-the-Art Performance**: Outperforms closed-source and extra-large open-source VLMs.

## Benchmarks & Performance Highlights

MonkeyOCR excels in various benchmarks, demonstrating its prowess in document parsing.  Here's a snapshot of the results:

*   **OmniDocBench**: MonkeyOCR-pro-3B achieves top overall performance on both English and Chinese documents.
*   **olmOCR-Bench**: MonkeyOCR-pro-1.2B outperforms Nanonets-OCR-3B by 7.3%.
*   **Text Recognition**: MonkeyOCR delivers excellent text recognition across diverse document types, including financial reports, textbooks, and newspapers.

**See the detailed benchmark results in the [original README](https://github.com/Yuliang-Liu/MonkeyOCR) for a comprehensive comparison.**

## Inference Speed

The following tables show the inference speed (pages per second) for different models on various GPUs:

### Inference Speed (Pages/s) on Different GPUs (Full Parsing)

| Model             | GPU    | 50 Pages | 100 Pages | 300 Pages | 500 Pages | 1000 Pages |
| ----------------- | ------ | -------- | --------- | --------- | --------- | ---------- |
| MonkeyOCR-pro-3B  | 3090   | 0.492    | 0.484     | 0.497     | 0.492     | 0.496      |
| MonkeyOCR-pro-3B  | A6000  | 0.585    | 0.587     | 0.609     | 0.598     | 0.608      |
| MonkeyOCR-pro-3B  | H800   | 0.923    | 0.768     | 0.897     | 0.930     | 0.891      |
| MonkeyOCR-pro-3B  | 4090   | 0.972    | 0.969     | 1.006     | 0.986     | 1.006      |
| MonkeyOCR-pro-1.2B | 3090   | 0.615    | 0.660     | 0.677     | 0.687     | 0.683      |
| MonkeyOCR-pro-1.2B | A6000  | 0.709    | 0.786     | 0.825     | 0.829     | 0.825      |
| MonkeyOCR-pro-1.2B | H800   | 0.965    | 1.082     | 1.101     | 1.145     | 1.015      |
| MonkeyOCR-pro-1.2B | 4090   | 1.194    | 1.314     | 1.436     | 1.442     | 1.434      |

### VLM OCR Speed (Pages/s) on Different GPUs (Text Recognition)

| Model             | GPU    | 50 Pages | 100 Pages | 300 Pages | 500 Pages | 1000 Pages |
| ----------------- | ------ | -------- | --------- | --------- | --------- | ---------- |
| MonkeyOCR-pro-3B  | 3090   | 0.705    | 0.680     | 0.711     | 0.700     | 0.724      |
| MonkeyOCR-pro-3B  | A6000  | 0.885    | 0.860     | 0.915     | 0.892     | 0.934      |
| MonkeyOCR-pro-3B  | H800   | 1.371    | 1.135     | 1.339     | 1.433     | 1.509      |
| MonkeyOCR-pro-3B  | 4090   | 1.321    | 1.300     | 1.384     | 1.343     | 1.410      |
| MonkeyOCR-pro-1.2B | 3090   | 0.919    | 1.086     | 1.166     | 1.182     | 1.199      |
| MonkeyOCR-pro-1.2B | A6000  | 1.177    | 1.361     | 1.506     | 1.525     | 1.569      |
| MonkeyOCR-pro-1.2B | H800   | 1.466    | 1.719     | 1.763     | 1.875     | 1.650      |
| MonkeyOCR-pro-1.2B | 4090   | 1.759    | 1.987     | 2.260     | 2.345     | 2.415      |

## Quick Start

Get up and running quickly with MonkeyOCR.

### 1. Installation
Follow the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) to set up your environment.

### 2. Download Model Weights
Download the model from Hugging Face or ModelScope.

```bash
pip install huggingface_hub
python tools/download_model.py -n MonkeyOCR  # or MonkeyOCR-pro-1.2B
```
or
```bash
pip install modelscope
python tools/download_model.py -t modelscope -n MonkeyOCR  # or MonkeyOCR-pro-1.2B
```

### 3. Inference
Parse documents with the following command:

```bash
python parse.py <input_path>
```
Refer to the [original README](https://github.com/Yuliang-Liu/MonkeyOCR) for detailed usage instructions and options.

### 4. Gradio Demo
Launch the interactive demo:
```bash
python demo/demo_gradio.py
```
Access the demo at http://localhost:7860.

### 5. Fast API

```bash
uvicorn api.main:app --port 8000
```
API documentation: http://localhost:8000/docs

## Docker Deployment

Run MonkeyOCR using Docker.  See the [original README](https://github.com/Yuliang-Liu/MonkeyOCR) for detailed instructions.

## Windows Support

Windows users, please refer to the [windows support guide](docs/windows_support.md).

## Quantization

Optimize model size and performance with AWQ quantization. See the [quantization guide](docs/Quantization.md).

## Demo

Experience MonkeyOCR in action with our interactive demo:  http://vlrlabmonkey.xyz:7685

## Citation

Cite MonkeyOCR using the following BibTeX entry:

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

We extend our gratitude to the contributors of the tools and datasets used in MonkeyOCR.

## Limitations

*   Limited support for photographed text, handwritten content, Traditional Chinese characters, and multilingual text.
*   Demo may experience high traffic issues.

## Copyright

MonkeyOCR is available for academic research and non-commercial use. For commercial inquiries, please contact the authors.