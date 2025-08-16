# MonkeyOCR: Revolutionizing Document Parsing with a Structure-Recognition-Relation Triplet Paradigm

**MonkeyOCR** is a powerful document parsing tool that leverages a Structure-Recognition-Relation (SRR) triplet paradigm to accurately extract information from various document types. ([Original Repo](https://github.com/Yuliang-Liu/MonkeyOCR))

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

## Key Features

*   **Superior Performance:** MonkeyOCR-pro-1.2B surpasses MonkeyOCR-3B in accuracy and speed.
*   **Faster Inference:** Achieve up to a 36% speed improvement compared to the 3B version with only a minor performance drop.
*   **Strong Results:** MonkeyOCR-pro-1.2B excels on olmOCR-Bench, and MonkeyOCR-pro-3B achieves state-of-the-art results on OmniDocBench.
*   **Versatile Support:** Supports a wide range of document types.
*   **Easy to Use:** Simple installation and quick start guides for local use, API deployment and Docker setup.

## Why Choose MonkeyOCR?

MonkeyOCR simplifies document parsing by using a Structure-Recognition-Relation (SRR) triplet paradigm, offering a more efficient and effective alternative to traditional multi-tool pipelines and large multimodal models for complete document processing.

## Key Advantages:

*   **Efficiency:** Streamlined SRR approach.
*   **Speed:** Faster processing compared to larger models.
*   **Accuracy:** Competitive performance on various benchmarks.
*   **Flexibility:** Supports both English and Chinese documents.
*   **Accessibility:** Easy to set up and use, with available API and Docker deployment.

## Benchmarks

[See the original README for detailed benchmark results.]

## Quick Start

### 1. Local Installation

*   Follow the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) to set up your environment.
*   Download model weights:

    ```bash
    pip install huggingface_hub
    python tools/download_model.py -n MonkeyOCR  # or MonkeyOCR-pro-1.2B
    ```
    or from ModelScope:
    ```bash
    pip install modelscope
    python tools/download_model.py -t modelscope -n MonkeyOCR  # or MonkeyOCR-pro-1.2B
    ```
### 2. Inference

Use the following commands to parse PDFs or images:

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

[See original README for More usage examples.]

### 3. Gradio Demo

Run the Gradio demo:

```bash
python demo/demo_gradio.py
```

Access the demo at http://localhost:7860.

### 4. Fast API

Start the FastAPI service:

```bash
uvicorn api.main:app --port 8000
```

Access API documentation at http://localhost:8000/docs.

## Docker Deployment

[See original README for Docker deployment instructions.]

## Windows Support

[See original README for Windows support guide.]

## Quantization

[See original README for Quantization guide.]

## Limitations

*   Limited support for photographed text, handwritten content, Traditional Chinese characters, and multilingual text (planned for future releases).
*   Demo performance may be affected by high traffic.
*   Inference speeds were measured on an H800 GPU (and may vary).

## Support and Contact

For feedback, contact xbai@hust.edu.cn or ylliu@hust.edu.cn.  

## Acknowledgments
[See original README for acknowledgments.]

## Citing MonkeyOCR

[See original README for BibTeX entry.]

## Copyright
[See original README for Copyright info.]
```
Key improvements and summary:

*   **SEO Optimization:** Headings, use of keywords ("document parsing," "OCR," "structure recognition," "triplet paradigm"), and a strong introductory sentence designed for search engines.
*   **Concise Summary:** The README is drastically shortened, focusing on the core benefits and features of MonkeyOCR.
*   **Key Features:** Key features are now bulleted for easier readability.
*   **Clear Structure:**  Organized with clear headings and sections.
*   **Actionable Quick Start:** The installation and usage instructions are clear.
*   **Call to Action:** Includes a call to action by encouraging users to contribute feedback.
*   **Links:** Kept all existing links, and added a direct link to the original repo.
*   **Removed Redundancy:**  Removed repetitive information.
*   **Simplified Language:** Used simpler language for a broader audience.
*   **Focus on Value:**  Highlights the benefits of using MonkeyOCR.
*   **Easy Navigation:**  Uses markdown formatting and hyperlinks.