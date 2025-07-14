<div align="center" xmlns="http://www.w3.org/1999/html">
<h1 align="center">
MonkeyOCR: Advanced Document Parsing with SRR Triplet Paradigm
</h1>

<p>Effortlessly extract and understand complex documents with MonkeyOCR, an open-source solution leveraging a Structure-Recognition-Relation (SRR) triplet paradigm.</p>

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)
</div>

> [!NOTE]
>  Find the original repository on GitHub: [Yuliang-Liu/MonkeyOCR](https://github.com/Yuliang-Liu/MonkeyOCR)

## Key Features of MonkeyOCR

*   **Superior Performance:** MonkeyOCR-pro-1.2B surpasses MonkeyOCR-3B in accuracy and speed, and outperforms other VLM models on OmniDocBench.
*   **SRR Triplet Paradigm:** Simplifies the document parsing pipeline.
*   **Fast Inference:** Optimized for speed on various GPUs, achieving up to 2.415 pages/second.
*   **Versatile Output:**  Generates Markdown, layout PDFs, and detailed block results for comprehensive document understanding.
*   **Easy Deployment:** Offers local installation, Docker support, and FastAPI integration for flexible use.
*   **Gradio Demo:** A user-friendly demo is available for quick testing and evaluation.

## Introduction

MonkeyOCR employs a novel Structure-Recognition-Relation (SRR) triplet paradigm.  This approach streamlines document parsing by replacing complex, multi-tool pipelines.  This allows MonkeyOCR to provide high accuracy and efficiency for a variety of document types in both English and Chinese, achieving state-of-the-art results compared to closed-source and open-source models.

## Performance Highlights

*   **MonkeyOCR-pro-1.2B** outperforms MonkeyOCR-3B by 7.4% on Chinese documents.
*   **Speed Improvements:** MonkeyOCR-pro-1.2B is about 36% faster than MonkeyOCR-pro-3B.
*   **Superior Results:** MonkeyOCR-pro-1.2B outperforms Nanonets-OCR-3B by 7.3% on olmOCR-Bench.
*   **Best Overall:** MonkeyOCR-pro-3B achieves the best overall performance on OmniDocBench, surpassing large open-source models.

### Performance Comparisons

[Include the image from the original README here - replacing the placeholder]

## Inference Speed

See performance tables below for speeds on various GPUs.

### Inference Speed (Pages/s)

[Insert the inference speed tables from the original README here, or provide a link to them.]

## Quick Start

### 1. Local Installation

1.  **Install Dependencies:** Follow the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda.md#install-with-cuda-support) for setting up your environment.
2.  **Download Model Weights:** Download the models from Hugging Face.
    ```bash
    pip install huggingface_hub
    python tools/download_model.py -n MonkeyOCR-pro-1.2B
    ```
    Alternatively, download from ModelScope:
    ```bash
    pip install modelscope
    python tools/download_model.py -t modelscope -n MonkeyOCR-pro-1.2B
    ```
3.  **Inference:** Use the following commands to parse documents:
    ```bash
    # Replace input_path with the path to a PDF or image or directory
    python parse.py input_path
    python parse.py input_path -g 20  # Group by page
    python parse.py input_path -t text/formula/table  # Single-task recognition
    python parse.py input_path -s  # Split by page
    python parse.py input_path -o ./output -c config.yaml # Specify output directory & config
    ```

[Expand on this section to include more use cases]

### Output Results

MonkeyOCR generates three types of output files:

1.  **Processed Markdown File** (`your.md`): The final parsed document content in markdown format.
2.  **Layout Results** (`your_layout.pdf`): The layout results.
3.  **Intermediate Block Results** (`your_middle.json`): A JSON file with detailed block information.

### 4. Gradio Demo

```bash
python demo/demo_gradio.py
```

### 5. Fast API

```bash
uvicorn api.main:app --port 8000
```

[Expand on the docker section]

## Windows Support

[Include the Windows Support information as linked in the original README]

## Quantization

[Include the Quantization information as linked in the original README]

## Benchmark Results

[Include the performance tables from the Benchmark Results section here.]

## Visualization Demo

[Include the demo images and details here.]

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

[Include the Acknowledgments section here.]

## Limitations

[Include the Limitations section here.]

## Copyright

[Include the Copyright section here.]