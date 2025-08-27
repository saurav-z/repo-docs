# MonkeyOCR: Effortless Document Parsing with Advanced Structure Recognition

**Unlock the power of structured document analysis with MonkeyOCR, a cutting-edge solution leveraging a Structure-Recognition-Relation (SRR) triplet paradigm.  [Explore the original repository](https://github.com/Yuliang-Liu/MonkeyOCR) for more details!**

MonkeyOCR simplifies document parsing by efficiently extracting and organizing text, tables, formulas, and layout information from PDFs and images.  

## Key Features:

*   **SRR Paradigm:**  Streamlines the document parsing process for improved efficiency.
*   **Superior Performance:** Outperforms existing tools and large language models (LLMs) on various benchmarks.
*   **Blazing-Fast Speeds:** Optimized for high-speed document processing.
*   **Comprehensive Support:** Handles a wide range of document types, including books, reports, and scientific papers.
*   **Multiple Deployment Options:**  Supports local installation, Gradio demo, Docker deployment, and a FastAPI service.
*   **Quantization Support:** Compatible with AWQ quantization for resource-efficient deployment.
*   **Open Source:** Actively maintained with community contributions.

## Performance Highlights:

*   **MonkeyOCR-pro-1.2B** offers significant speed improvements and competitive accuracy.
*   **MonkeyOCR-pro-3B** achieves state-of-the-art performance on the OmniDocBench benchmark.
*   Competitive results on olmOCR-Bench compared to open-source and closed-source VLMs.

### [See the evaluation results](https://github.com/Yuliang-Liu/MonkeyOCR#benchmark-results) to explore performance in-depth!

## Quick Start:

### 1. Installation

Refer to the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) for setting up your environment.

### 2. Model Download

Download model weights from Hugging Face or ModelScope.

```bash
# Hugging Face
pip install huggingface_hub
python tools/download_model.py -n MonkeyOCR-pro-3B  # or MonkeyOCR

# ModelScope
pip install modelscope
python tools/download_model.py -t modelscope -n MonkeyOCR-pro-3B  # or MonkeyOCR
```

### 3. Inference

Use the `parse.py` script to process documents.  Here are some example commands:

```bash
# Parse a single PDF
python parse.py input.pdf

# Parse all files in a directory
python parse.py /path/to/folder

# Text recognition from a single image
python parse.py image.jpg -t text

# Use a group page num
python parse.py input_path -g 20
```

For more details, check out the [usage examples](https://github.com/Yuliang-Liu/MonkeyOCR#quick-start).

### 4. Try the Demo!

Experience MonkeyOCR firsthand with our interactive demo:  http://vlrlabmonkey.xyz:7685 (The latest model is available for selection)

## Additional Resources:

*   [Windows Support Guide](docs/windows_support.md)
*   [Quantization Guide](docs/Quantization.md)
*   [Demo Images](https://github.com/Yuliang-Liu/MonkeyOCR#visualization-demo)
*   [Citing MonkeyOCR](https://github.com/Yuliang-Liu/MonkeyOCR#citing-monkeyocr)
*   [Acknowledgments](https://github.com/Yuliang-Liu/MonkeyOCR#acknowledgments)

## Important Considerations:

*   **Limitations:**  Currently supports English and Chinese documents. Photographed text, handwriting, and multilingual text are not yet fully supported.
*   **Commercial Use:**  Intended for academic research and non-commercial use.

## Future Plans:

*   Expand language support.
*   Enhance support for photographed and handwritten content.
*   Optimize performance.
*   Further expand deployment options.

**Join us in shaping the future of document processing!**  Please see the [contributing guidelines](CONTRIBUTING.md) for more details on contributing to MonkeyOCR.  Contact us at xbai@hust.edu.cn or ylliu@hust.edu.cn if you're interested in faster (smaller) or stronger model, or have any feedback!