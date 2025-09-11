# MonkeyOCR: Unlock Document Understanding with Unmatched Precision and Speed

**Tired of clunky document parsing pipelines?** MonkeyOCR offers a streamlined, efficient, and highly accurate solution using a Structure-Recognition-Relation (SRR) triplet paradigm.  [View the original repo](https://github.com/Yuliang-Liu/MonkeyOCR) to get started today!

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR-pro-3B)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

## Key Features

*   **Superior Accuracy:** MonkeyOCR-pro-3B achieves state-of-the-art results on a range of benchmarks, surpassing even closed-source and extra-large open-source VLMs.
*   **Blazing Speed:**  Experience significant speed improvements compared to other solutions, enabling faster document processing.
*   **SRR Paradigm:** Simplifies document parsing with a novel Structure-Recognition-Relation triplet approach.
*   **Flexible Deployment:** Supports various GPUs (3090, 4090, A6000, H800, etc.) and includes Docker deployment for easy setup.
*   **Quantization Support:** Offers quantization options for optimized performance and resource utilization.
*   **Versatile Output:** Generates Markdown, layout results, and intermediate block results for comprehensive document understanding.
*   **Gradio and FastAPI Demo:** Provides an interactive Gradio demo and a FastAPI service for easy experimentation and integration.

## What's New

*   **MonkeyOCR-pro-1.2B Released:** 🚀 A leaner, faster, and even more accurate version that outperforms the 3B model.
*   **Trending on Hugging Face:** 🚀  Thanks for the support!
*   **Initial Release:** 🚀 MonkeyOCR is now available, offering powerful document parsing capabilities for English and Chinese documents.

## Performance Highlights

*   **Outperforms Leading Models:** MonkeyOCR-pro-3B and MonkeyOCR-pro-1.2B consistently outperform other open-source and commercial solutions on benchmarks like OmniDocBench and olmOCR-Bench.
*   **Speed Comparisons:**  See the tables below for detailed inference speed comparisons across different GPUs and document sizes.

### Inference Speed (Pages/s)

**On Different GPUs**  ([PDF](https://drive.google.com/drive/folders/1geumlJmVY7UUKdr8324sYZ0FHSAElh7m?usp=sharing))

| Model                | GPU   | 50 Pages | 100 Pages | 300 Pages | 500 Pages | 1000 Pages |
| :------------------- | :---- | :------- | :-------- | :-------- | :-------- | :--------- |
| MonkeyOCR-pro-3B     | 3090  | 0.492    | 0.484     | 0.497     | 0.492     | 0.496      |
| MonkeyOCR-pro-3B     | A6000 | 0.585    | 0.587     | 0.609     | 0.598     | 0.608      |
| MonkeyOCR-pro-3B     | H800  | 0.923    | 0.768     | 0.897     | 0.930     | 0.891      |
| MonkeyOCR-pro-3B     | 4090  | 0.972    | 0.969     | 1.006     | 0.986     | 1.006      |
| MonkeyOCR-pro-1.2B   | 3090  | 0.615    | 0.660     | 0.677     | 0.687     | 0.683      |
| MonkeyOCR-pro-1.2B   | A6000 | 0.709    | 0.786     | 0.825     | 0.829     | 0.825      |
| MonkeyOCR-pro-1.2B   | H800  | 0.965    | 1.082     | 1.101     | 1.145     | 1.015      |
| MonkeyOCR-pro-1.2B   | 4090  | 1.194    | 1.314     | 1.436     | 1.442     | 1.434      |

**VLM OCR Speed (Pages/s)**

| Model                | GPU   | 50 Pages | 100 Pages | 300 Pages | 500 Pages | 1000 Pages |
| :------------------- | :---- | :------- | :-------- | :-------- | :-------- | :--------- |
| MonkeyOCR-pro-3B     | 3090  | 0.705    | 0.680     | 0.711     | 0.700     | 0.724      |
| MonkeyOCR-pro-3B     | A6000 | 0.885    | 0.860     | 0.915     | 0.892     | 0.934      |
| MonkeyOCR-pro-3B     | H800  | 1.371    | 1.135     | 1.339     | 1.433     | 1.509      |
| MonkeyOCR-pro-3B     | 4090  | 1.321    | 1.300     | 1.384     | 1.343     | 1.410      |
| MonkeyOCR-pro-1.2B   | 3090  | 0.919    | 1.086     | 1.166     | 1.182     | 1.199      |
| MonkeyOCR-pro-1.2B   | A6000 | 1.177    | 1.361     | 1.506     | 1.525     | 1.569      |
| MonkeyOCR-pro-1.2B   | H800  | 1.466    | 1.719     | 1.763     | 1.875     | 1.650      |
| MonkeyOCR-pro-1.2B   | 4090  | 1.759    | 1.987     | 2.260     | 2.345     | 2.415      |

### Performance Metrics
The evaluation results on OmniDocBench are also included in the following tables:

### 1. The end-to-end evaluation results of different tasks.
* *Table showing the end-to-end evaluation results on the tasks of overall, text, formula, table, table editing and read order. The results are from the comparison of the pipeline tools, expert VLMs, general VLMs, and Mix models. The table is available on the original README.*

### 2. The end-to-end text recognition performance across 9 PDF page types.
*   *Table presenting the end-to-end text recognition performance across various PDF types, comparing performance across different models.*

### 3. The evaluation results of olmOCR-bench.
*   *Table illustrating the performance on the olmOCR-bench dataset.*

## Getting Started

### 1. Installation
See the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) to set up your environment.

### 2. Model Download
Download model weights from Hugging Face or ModelScope using the `tools/download_model.py` script.

### 3. Inference
Use the `parse.py` script to process PDFs or images.  Example commands are provided, including options for:

*   End-to-end parsing
*   Single-task recognition
*   Batch processing
*   Custom configurations

### 4. Gradio Demo
Run the interactive Gradio demo using `python demo/demo_gradio.py`.

### 5. Fast API
Start the FastAPI service using `uvicorn api.main:app --port 8000` and access the API documentation at http://localhost:8000/docs.

## Docker Deployment

Follow the instructions in the "Docker Deployment" section of the original README to deploy MonkeyOCR using Docker Compose.

## Windows Support

See the [windows support guide](docs/windows_support.md) for details.

## Quantization

Quantize the model using AWQ. Follow the instructions in the [quantization guide](docs/Quantization.md).

## Visualization Demo

Experience the power of MonkeyOCR firsthand with our interactive demo: http://vlrlabmonkey.xyz:7685 (The latest model is available for selection)

### Demo Features

*   Upload PDFs or images.
*   Parse documents with structure detection, content recognition, and relationship prediction.
*   Generate Markdown output.
*   Test with various prompts for content recognition.

### Example Output Images (Newspaper, Financial Report, etc.)
*   *Example images are provided in the original README*

## Citing MonkeyOCR

Cite our work using the provided BibTeX entry:

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

*   *List of acknowledgements from original README*

## Limitations

*   Limited support for photographed text, handwritten content, Traditional Chinese characters, and multilingual text.
*   Potential for demo slowdowns during high traffic due to single-GPU deployment.
*   Demo processing time includes overhead.

## Copyright

Please share your feedback to help us improve!  The model is for academic research and non-commercial use.  For faster (smaller) or stronger models, contact xbai@hust.edu.cn or ylliu@hust.edu.cn.