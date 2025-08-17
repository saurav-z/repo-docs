# MonkeyOCR: Effortless Document Parsing with Advanced Structure Recognition

**Unleash the power of MonkeyOCR to accurately parse complex documents with unparalleled speed and efficiency, using a Structure-Recognition-Relation (SRR) triplet paradigm. 🐒** ([Original Repo](https://github.com/Yuliang-Liu/MonkeyOCR))

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

MonkeyOCR leverages a novel SRR paradigm to streamline document parsing, offering significant improvements over traditional modular approaches.

**Key Features:**

*   **Superior Performance:** MonkeyOCR-pro-1.2B achieves state-of-the-art results, surpassing even closed-source and large open-source VLM models on key benchmarks.
*   **Blazing Fast Speed:** Experience up to 36% speed improvements with MonkeyOCR-pro-1.2B compared to MonkeyOCR-pro-3B, while maintaining exceptional accuracy.
*   **Multi-Language Support:**  Effectively parses both English and Chinese documents.
*   **Versatile Use Cases:** Ideal for extracting text, formulas, tables, and document structure from PDFs and images.
*   **Easy to Use:** Simple installation and straightforward commands for quick deployment.
*   **Comprehensive Results:** Outputs processed markdown, layout results, and detailed intermediate block results.
*   **Flexible Deployment:** Supports local installation, Gradio demo, FastAPI service, and Docker deployment.
*   **Hardware Compatibility:** Tested on a wide range of GPUs (3090, 4090, A6000, H800, etc.) and supported by community contributions on additional hardware.

## Performance Highlights

MonkeyOCR consistently outperforms leading models on the OmniDocBench and olmOCR-Bench, showcasing its robust document understanding capabilities.

### OmniDocBench Performance

*   MonkeyOCR-pro-3B achieves the best overall performance on both English and Chinese documents, outperforming even closed-source and extra-large open-source VLMs such as Gemini 2.0-Flash, Gemini 2.5-Pro, Qwen2.5-VL-72B, GPT-4o, and InternVL3-78B.

### End-to-end Evaluation Results Summary (OmniDocBench)

The following image presents a summarized view of the performance of MonkeyOCR compared to other models. For comprehensive results, refer to the detailed tables in the original README.
<a href="https://zimgs.com/i/EKhkhY"><img src="https://v1.ax1x.com/2025/07/15/EKhkhY.png" alt="EKhkhY.png" border="0" /></a>


## Inference Speed (Pages/s)

### Detailed speed benchmarks across various GPU configurations.
> Comprehensive benchmarks and speed comparisons are available. See original README for full details. Example follows.

**Example Table for Inference Speed (Pages/s)**

| Model              | GPU   | 50 Pages | 100 Pages | 300 Pages | 500 Pages | 1000 Pages |
| ------------------ | ----- | -------- | --------- | --------- | --------- | ---------- |
| MonkeyOCR-pro-3B   | 3090  | 0.492    | 0.484     | 0.497     | 0.492     | 0.496      |
| MonkeyOCR-pro-1.2B | 3090  | 0.615    | 0.660     | 0.677     | 0.687     | 0.683      |

## Quick Start Guide

### 1. Local Installation
1.  **Install MonkeyOCR:** Follow the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) to set up your environment.
2.  **Download Model Weights:** Choose your desired model from Hugging Face or ModelScope.

    ```bash
    pip install huggingface_hub
    python tools/download_model.py -n MonkeyOCR  # or MonkeyOCR-pro-1.2B
    ```
    or
    ```bash
    pip install modelscope
    python tools/download_model.py -t modelscope -n MonkeyOCR  # or MonkeyOCR-pro-1.2B
    ```

3.  **Inference:** Use the `parse.py` script with various options for processing:
    ```bash
    # Parse a file or a directory
    python parse.py input_path
    # For more options, see the original README
    ```

### 2. Gradio Demo

Launch a user-friendly demo to experiment with MonkeyOCR:
```bash
python demo/demo_gradio.py
```
Access the demo at: http://localhost:7860

### 3. Fast API

Start an API service for flexible integration:
```bash
uvicorn api.main:app --port 8000
```
Access API documentation at: http://localhost:8000/docs

### 4. Docker Deployment

Deploy MonkeyOCR effortlessly using Docker:
```bash
#build image (requires NVIDIA GPU support)
cd docker
docker compose build monkeyocr

# Run Gradio demo 
docker compose up monkeyocr-demo

# Run FastAPI
docker compose up monkeyocr-api
```

## Quantization

Optimize model performance using AWQ quantization: Refer to the [quantization guide](docs/Quantization.md).

## Benchmark Results & Evaluation

For comprehensive performance metrics and detailed comparisons, see the tables in the original README.  Key results are summarized above.

## Visualization Demo

Experience MonkeyOCR's capabilities firsthand with our interactive demo: [http://vlrlabmonkey.xyz:7685](http://vlrlabmonkey.xyz:7685)

**Supported PDF types:**

*   Diverse Chinese and English documents.
*   Example:
    <img src="https://v1.ax1x.com/2025/06/10/7jVLgB.jpg" alt="7jVLgB.jpg" border="0" />
*   Tables
    <img src="https://v1.ax1x.com/2025/06/11/7jcOaa.png" alt="7jcOaa.png" border="0" />
*   Newspapers
    <img src="https://v1.ax1x.com/2025/06/11/7jcP5V.png" alt="7jcP5V.png" border="0" />
*   Financial Reports
    <img src="https://v1.ax1x.com/2025/06/11/7jc10I.png" alt="7jc10I.png" border="0" />
    <img src="https://v1.ax1x.com/2025/06/11/7jcRCL.png" alt="7jcRCL.png" border="0" />

## Citing MonkeyOCR

If you use MonkeyOCR in your research, please cite it using the following BibTeX entry:

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

We extend our gratitude to the following projects and contributors for their valuable resources:

*   [MinerU](https://github.com/opendatalab/MinerU), [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO), [PyMuPDF](https://github.com/pymupdf/PyMuPDF), [layoutreader](https://github.com/ppaanngggg/layoutreader), [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [LMDeploy](https://github.com/InternLM/lmdeploy), [PP-StructureV3](https://github.com/PaddlePaddle/PaddleOCR), [PP-DocLayout_plus-L](https://huggingface.co/PaddlePaddle/PP-DocLayout_plus-L), and [InternVL3](https://github.com/OpenGVLab/InternVL).

*   [M6Doc](https://github.com/HCIILAB/M6Doc), [DocLayNet](https://github.com/DS4SD/DocLayNet), [CDLA](https://github.com/buptlihang/CDLA), [D4LA](https://github.com/AlibabaResearch/AdvancedLiterateMachinery), [DocGenome](https://github.com/Alpha-Innovator/DocGenome), [PubTabNet](https://github.com/ibm-aur-nlp/PubTabNet), and [UniMER-1M](https://github.com/opendatalab/UniMERNet).

*   Everyone who contributed to this open-source effort.

## Limitations

*   MonkeyOCR does not yet fully support photographed text, handwritten content, Traditional Chinese characters, or multilingual text.
*   Single GPU deployment can cause demo slowdowns during high traffic.
*   Processing time includes overhead beyond computation time.

## Copyright

This model is intended for academic research and non-commercial use only.  For inquiries regarding faster (smaller) or stronger models, please contact xbai@hust.edu.cn or ylliu@hust.edu.cn.