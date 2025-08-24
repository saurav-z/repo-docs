# MonkeyOCR: Effortlessly Parse Documents with Cutting-Edge AI

**MonkeyOCR revolutionizes document parsing with its Structure-Recognition-Relation (SRR) triplet paradigm, offering a streamlined and efficient approach to processing your documents.**

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

**Key Features:**

*   **SRR Paradigm:** Simplifies document parsing, surpassing modular approaches.
*   **Superior Performance:** MonkeyOCR-pro-1.2B outperforms previous versions and competes with state-of-the-art models.
*   **Optimized Speed:** Achieve significant speed improvements across various hardware configurations.
*   **Comprehensive Support:** Supports diverse document types, including PDFs, images, and a wide range of hardware.
*   **Easy to Use:** Includes a Gradio demo, FastAPI service, and straightforward installation.

**Key Achievements**

*   MonkeyOCR-pro-1.2B surpasses MonkeyOCR-3B by 7.4% on Chinese documents.
*   MonkeyOCR-pro-1.2B delivers approximately a 36% speed improvement over MonkeyOCR-pro-3B, with approximately 1.6% drop in performance.
*   On olmOCR-Bench, MonkeyOCR-pro-1.2B outperforms Nanonets-OCR-3B by 7.3%.
*   On OmniDocBench, MonkeyOCR-pro-3B achieves the best overall performance on both English and Chinese documents, outperforming even closed-source and extra-large open-source VLMs such as Gemini 2.0-Flash, Gemini 2.5-Pro, Qwen2.5-VL-72B, GPT-4o, and InternVL3-78B.

**Compare with Closed-Source and Large Open-Source Models:**
[Image comparing the model with closed-source and extra large open-source VLMs, as shown in the original README.md.  For a complete comparison, review the image located within the [Original Repository](https://github.com/Yuliang-Liu/MonkeyOCR)]

## Inference Speed

*   **See the original README for GPU Speed and PDF counts.**

**[Full details about performance benchmarks, including OmniDocBench and olmOCR-bench, can be found in the original [MonkeyOCR Repository](https://github.com/Yuliang-Liu/MonkeyOCR).]**

## Getting Started

### Installation

*   Follow the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) to set up your environment.

### Download Model Weights

*   Download from Hugging Face:
    ```bash
    pip install huggingface_hub
    python tools/download_model.py -n MonkeyOCR-pro-3B  # or MonkeyOCR
    ```

*   Download from ModelScope:
    ```bash
    pip install modelscope
    python tools/download_model.py -t modelscope -n MonkeyOCR-pro-3B  # or MonkeyOCR
    ```

### Inference

*   Run end-to-end parsing:
    ```bash
    python parse.py input_path
    ```
*   Parse with grouping
    ```bash
    python parse.py input_path -g 20
    ```
*   Single-task recognition
    ```bash
    python parse.py input_path -t text/formula/table
    ```

### Gradio Demo

*   Run the demo:
    ```bash
    python demo/demo_gradio.py
    ```
    Access the demo at http://localhost:7860.

### FastAPI

*   Run FastAPI:
    ```bash
    uvicorn api.main:app --port 8000
    ```
    Access API documentation at http://localhost:8000/docs.

### Docker Deployment

*   Follow the Docker Deployment instructions in the original [README](https://github.com/Yuliang-Liu/MonkeyOCR)

## Supported Hardware

*   MonkeyOCR has been tested on a range of GPUs, including 3090, 4090, A6000, H800, A100, and others. See the original [README](https://github.com/Yuliang-Liu/MonkeyOCR) for more details.

## News

*   **July 10, 2025:** MonkeyOCR-pro-1.2B is released!
*   **June 12, 2025:**  Trending on Hugging Face.
*   **June 5, 2025:** MonkeyOCR is released.

## Windows Support

*   See the [windows support guide](docs/windows_support.md) for details.

## Quantization

*   This model can be quantized using AWQ. Follow the instructions in the [quantization guide](docs/Quantization.md).

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

*   [MinerU](https://github.com/opendatalab/MinerU), [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO), [PyMuPDF](https://github.com/pymupdf/PyMuPDF), and other projects. See the [original README](https://github.com/Yuliang-Liu/MonkeyOCR) for a full list.

## Limitations

*   MonkeyOCR may not fully support photographed text, handwritten content, Traditional Chinese characters, or multilingual text.
*   Single GPU deployment may cause congestion during peak times. See the original [README](https://github.com/Yuliang-Liu/MonkeyOCR) for more.

## Copyright

*   For academic research and non-commercial use only.  Contact xbai@hust.edu.cn or ylliu@hust.edu.cn for faster (smaller) or stronger versions.

---

**For more details, usage examples, and comprehensive documentation, visit the original [MonkeyOCR Repository](https://github.com/Yuliang-Liu/MonkeyOCR).**