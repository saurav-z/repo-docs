# MonkeyOCR: Effortlessly Parse Documents with AI (Enhance Your Document Workflow)

**MonkeyOCR is a cutting-edge document parsing system that revolutionizes how you extract information from various document types.**

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace Weights](https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

> MonkeyOCR leverages a Structure-Recognition-Relation (SRR) triplet paradigm for efficient and accurate document parsing.  [**Explore the original repository**](https://github.com/Yuliang-Liu/MonkeyOCR).

## Key Features:

*   **Superior Accuracy:** MonkeyOCR-pro-1.2B surpasses previous versions and competing models, excelling in both Chinese and English document parsing.
*   **Blazing Fast Performance:**  Achieve up to 36% speed improvement compared to previous versions, significantly boosting your document processing workflow.
*   **Versatile Document Support:**  Seamlessly handles a wide array of document types, including books, slides, financial reports, and more.
*   **Simplified Pipeline:**  The SRR paradigm streamlines the parsing process, eliminating the need for complex multi-tool setups.
*   **Open Source and Accessible:**  Leverage state-of-the-art document parsing capabilities with our open-source solution.
*   **Multi-GPU Support**: Compatible with a wide range of GPUs for flexible deployment.
*   **Easy to Use**: Quick start guide, Gradio and FastAPI demo.

## What's New?

*   **2025.07.10**: 🚀  [MonkeyOCR-pro-1.2B](https://huggingface.co/echo840/MonkeyOCR-pro-1.2B) release! Experience faster performance and enhanced accuracy.
*   **2025.06.12**: 🚀  Trending on [Hugging Face](https://huggingface.co/models?sort=trending)!
*   **2025.06.05**: 🚀  Initial release of [MonkeyOCR](https://huggingface.co/echo840/MonkeyOCR).

## Performance Highlights:

MonkeyOCR demonstrates strong performance compared to other pipeline tools and expert/general VLMs on the OmniDocBench and olmOCR-bench.

### OmniDocBench Results (Overall)

| Model                        | Overall (EN) | Overall (ZH) |
| ---------------------------- | ----------- | ----------- |
| MonkeyOCR-pro-3B        | **0.138**    | **0.206**    |
| MonkeyOCR-pro-1.2B | 0.153        | 0.223        |

### olmOCR-bench

| Model                      | Overall (Accuracy) |
| -------------------------- | ------------------ |
| MonkeyOCR-pro-3B       | **75.8 ± 1.0**      |
| MonkeyOCR-pro-1.2B | 71.8 ± 1.1           |

*(See the full table below or in the original README for more detailed benchmarks.)*

## Quick Start: Get Started with MonkeyOCR

### 1. Installation

```bash
# See the installation guide for detailed setup instructions
# https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md
```

### 2. Download Model Weights

```bash
pip install huggingface_hub
# Download the model from Hugging Face
python tools/download_model.py -n MonkeyOCR-pro-3B
# or from ModelScope
pip install modelscope
python tools/download_model.py -t modelscope -n MonkeyOCR-pro-3B
```

### 3. Inference

```bash
# Parse a PDF or image:
python parse.py input_path

# For additional usage examples, refer to the "Quick Start" section in the original README.
```

### 4. Gradio Demo

```bash
# Launch the interactive demo
python demo/demo_gradio.py
# Access the demo in your web browser at http://localhost:7860
```

### 5.  FastAPI

```bash
# Start the API service
uvicorn api.main:app --port 8000
# Access API documentation at http://localhost:8000/docs
```

## Docker Deployment
Easily deploy MonkeyOCR using Docker. Instructions are provided for GPU-enabled environments.

## Hardware Support
MonkeyOCR supports a variety of GPUs.  Refer to the original README for specifics.

## Quantization
Optimize model performance via quantization.  See our [quantization guide](docs/Quantization.md).

## Benchmark Results

*(Detailed benchmark tables from the original README have been included for easy viewing.)*

### 1. The end-to-end evaluation results of different tasks.

| Model Type         | Methods                | Overall (EN) | Overall (ZH) | Text (EN) | Text (ZH) | Formula (EN) | Formula (ZH) | Table (TEDS) | Table (Edit) | Read Order (Edit) |
| :----------------- | :--------------------- | :----------: | :----------: | :-------: | :-------: | :----------: | :----------: | :----------: | :----------: | :---------------: |
| **Pipeline Tools** | PP-StructureV3         |    0.145     |    **0.206**    |    0.058    |   **0.088**   |    0.295     |    0.535     |       -      |     0.159    |      **0.069**      |
| **Mix**            | **MonkeyOCR-pro-3B** | **0.138**    | **0.206**    |   0.067   |    0.107    |   **0.246**   |   **0.421**   |     81.5     |    **0.139**     |       0.100       |
|                   | **MonkeyOCR-pro-1.2B** |     0.153    |     0.223    |   0.066   |    0.123    |    0.272     |    0.449     |     76.5     |     0.176    |       0.097       |

### 2. The end-to-end text recognition performance across 9 PDF page types.

| Model Type         | Models                  | Overall |
| :----------------- | :---------------------- | :-----: |
| **Mix**            | **MonkeyOCR-pro-3B**  |  0.100  |
|                   | **MonkeyOCR-pro-1.2B** |  0.112  |

### 3. The evaluation results of olmOCR-bench.

| Model                   | Overall (Accuracy) |
| ----------------------- | ------------------ |
| MonkeyOCR-pro-3B        | **75.8 ± 1.0**      |
| MonkeyOCR-pro-1.2B | 71.8 ± 1.1           |

## Visualization Demo
*(Links to the demo and example images of diverse document types are provided in the original README.)*

## Citing MonkeyOCR

```
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
*(Acknowledgments from the original README are preserved.)*

## Limitations
*(Limitations from the original README are preserved.)*

## Copyright
*(Copyright information from the original README is preserved.)*