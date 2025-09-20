# MonkeyOCR: Effortless Document Parsing with Cutting-Edge Accuracy

**MonkeyOCR revolutionizes document processing using a Structure-Recognition-Relation (SRR) triplet paradigm, delivering superior performance in document understanding.**

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR-pro-3B)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

*   **[MonkeyOCR-pro-1.2B](https://huggingface.co/echo840/MonkeyOCR-pro-1.2B) Performance:** Outperforms MonkeyOCR-3B on Chinese documents by 7.4%, while delivering a 36% speed boost.
*   **State-of-the-Art Accuracy:** MonkeyOCR-pro-3B achieves best overall performance on English and Chinese documents, exceeding even closed-source models.
*   **Versatile Capabilities:** Processes a variety of document types, including formulas, tables, and text, with robust performance.

## Key Features

*   **SRR Paradigm:**  Employs a Structure-Recognition-Relation triplet approach for efficient document processing.
*   **High Accuracy:** Demonstrates superior performance compared to other models, including closed-source options.
*   **Fast Inference:**  Offers significant speed improvements with different model sizes.
*   **Multi-GPU Support:**  Supports multiple GPUs for faster processing.
*   **Gradio Demo & API:**  Provides a user-friendly Gradio demo and FastAPI service for easy experimentation and integration.

## Benchmarking & Performance

MonkeyOCR excels across various benchmarks, delivering impressive results.  See the complete comparison in the original [README](https://github.com/Yuliang-Liu/MonkeyOCR).

*   **[OmniDocBench](https://github.com/Yuliang-Liu/MonkeyOCR):** MonkeyOCR-pro-3B leads, demonstrating exceptional performance on both English and Chinese documents.
*   **[olmOCR-Bench](https://github.com/Yuliang-Liu/MonkeyOCR):** MonkeyOCR-pro-1.2B significantly surpasses Nanonets-OCR-3B by 7.3%.

### End-to-end Evaluation on OmniDocBench (Example)

The following are the edit metrics to show performance.

| Model                       | English Overall | Chinese Overall |
| --------------------------- | --------------- | --------------- |
| **MonkeyOCR-pro-3B**        | **0.138**       | **0.206**       |
| **MonkeyOCR-pro-1.2B**      | 0.153       | 0.223       |

See more detailed results in the original [README](https://github.com/Yuliang-Liu/MonkeyOCR).

## Getting Started

### Installation

1.  Install MonkeyOCR:
    ```bash
    See the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) to set up your environment.
    ```

### Quick Start

1.  **Install Dependencies:** Follow the install instructions above.
2.  **Download Model Weights:** Download our model from Huggingface:

    ```python
    pip install huggingface_hub

    python tools/download_model.py -n MonkeyOCR-pro-3B  # or MonkeyOCR
    ```
    You can also download our model from ModelScope.

    ```python
    pip install modelscope

    python tools/download_model.py -t modelscope -n MonkeyOCR-pro-3B  # or MonkeyOCR
    ```
3.  **Inference:**

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

    Further usage examples and options are in the original [README](https://github.com/Yuliang-Liu/MonkeyOCR).

4.  **Gradio Demo:**

    ```bash
    python demo/demo_gradio.py
    ```
    Access the demo at http://localhost:7860.

5.  **FastAPI:**

    ```bash
    uvicorn api.main:app --port 8000
    ```
    Access the API documentation at http://localhost:8000/docs.

## Docker Deployment

Simplified Docker instructions are available in the original [README](https://github.com/Yuliang-Liu/MonkeyOCR).

## Further Information

*   **Windows Support:** Refer to the [windows support guide](docs/windows_support.md).
*   **Quantization:** Explore the [quantization guide](docs/Quantization.md) for optimized performance.

## Demo

Experience MonkeyOCR firsthand with our interactive demo: [http://vlrlabmonkey.xyz:7685](http://vlrlabmonkey.xyz:7685)

## Learn More

Explore the complete documentation and contribute to the project on GitHub: [https://github.com/Yuliang-Liu/MonkeyOCR](https://github.com/Yuliang-Liu/MonkeyOCR)

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

We acknowledge the contributions of [MinerU](https://github.com/opendatalab/MinerU), [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO), [PyMuPDF](https://github.com/pymupdf/PyMuPDF), [layoutreader](https://github.com/ppaanngggg/layoutreader), [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [LMDeploy](https://github.com/InternLM/lmdeploy), [PP-StructureV3](https://github.com/PaddlePaddle/PaddleOCR), [PP-DocLayout_plus-L](https://huggingface.co/PaddlePaddle/PP-DocLayout_plus-L), and [InternVL3](https://github.com/OpenGVLab/InternVL) for base code and models, as well as their contributions to this field. We also thank [M6Doc](https://github.com/HCIILAB/M6Doc), [DocLayNet](https://github.com/DS4SD/DocLayNet), [CDLA](https://github.com/buptlihang/CDLA), [D4LA](https://github.com/AlibabaResearch/AdvancedLiterateMachinery), [DocGenome](https://github.com/Alpha-Innovator/DocGenome), [PubTabNet](https://github.com/ibm-aur-nlp/PubTabNet), and [UniMER-1M](https://github.com/opendatalab/UniMERNet) for providing valuable datasets. We also thank everyone who contributed to this open-source effort.

## Limitations

Current limitations and planned future improvements are detailed in the original [README](https://github.com/Yuliang-Liu/MonkeyOCR).

## Contact

For inquiries, contact xbai@hust.edu.cn or ylliu@hust.edu.cn.