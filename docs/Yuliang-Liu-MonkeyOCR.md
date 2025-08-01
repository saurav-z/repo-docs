# MonkeyOCR: Revolutionizing Document Parsing with a Triplet Paradigm

**MonkeyOCR leverages a Structure-Recognition-Relation (SRR) triplet paradigm to accurately and efficiently parse documents, outperforming leading models.** ([View Original Repo](https://github.com/Yuliang-Liu/MonkeyOCR))

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

## Key Features:

*   **Superior Performance:** MonkeyOCR-pro-1.2B outperforms its 3B counterpart in accuracy and speed on Chinese documents, and surpasses other models on multiple benchmarks.
*   **SRR Triplet Paradigm:** Simplifies multi-tool pipelines for efficient document processing.
*   **High Efficiency:** Offers significant speed improvements compared to other models, including a 36% speed increase over MonkeyOCR-pro-3B.
*   **Versatile Support:** Works well on various GPU types, including 3090, 4090, A6000, and H800.
*   **Comprehensive Output:** Generates markdown files, layout results, and intermediate block results for detailed analysis.
*   **Easy to Use:** Simple installation, model download, and inference steps with clear examples.
*   **Demo Available:** A user-friendly Gradio demo provides a quick hands-on experience ([Demo Link](http://vlrlabmonkey.xyz:7685/)).

## Why MonkeyOCR?

MonkeyOCR provides a cutting-edge solution for document parsing, offering superior speed and accuracy by employing a unique Structure-Recognition-Relation (SRR) triplet paradigm. It outperforms even closed-source and extra-large open-source VLMs.

## Recent Updates:

*   **July 10, 2025:** Released [MonkeyOCR-pro-1.2B](https://huggingface.co/echo840/MonkeyOCR-pro-1.2B) - a leaner and faster version.
*   **June 12, 2025:** Trending on [Hugging Face](https://huggingface.co/models?sort=trending).
*   **June 5, 2025:** Released [MonkeyOCR](https://huggingface.co/echo840/MonkeyOCR), supporting English and Chinese document parsing.

## Quick Start:

1.  **Installation:** Follow the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support).
2.  **Download Model Weights:**  Download from Hugging Face or ModelScope using `tools/download_model.py`.
3.  **Inference:** Use the `parse.py` script to process PDFs, images, or directories.  See the Quick Start section of the original README for detailed examples.
4.  **Gradio Demo:** Launch the demo with `python demo/demo_gradio.py`.
5.  **FastAPI:** Deploy the API using `uvicorn api.main:app --port 8000`.
6.  **Docker Deployment**: Deploy using the provided `docker` directory.
    *   Build the Docker image: `docker compose build monkeyocr`
    *   Run the Gradio demo: `docker compose up monkeyocr-demo`
    *   Run the FastAPI service: `docker compose up monkeyocr-api`

## Benchmarks and Results:

MonkeyOCR achieves state-of-the-art performance on OmniDocBench and olmOCR-Bench, exceeding the results of many existing models, including closed-source options.

### Comparison with Closed-Source and Extra-Large Open-Source VLMs
<a href="https://zimgs.com/i/EKhkhY"><img src="https://v1.ax1x.com/2025/07/15/EKhkhY.png" alt="EKhkhY.png" border="0" /></a>

### Inference Speed

*   **Inference Speed (Pages/s) on Different GPUs**

    *   **On Different GPUs and [PDF](https://drive.google.com/drive/folders/1geumlJmVY7UUKdr8324sYZ0FHSAElh7m?usp=sharing) Page Counts**

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

    *   **VLM OCR Speed (Pages/s) on Different GPUs**

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

### End-to-End Evaluation Results on OmniDocBench

| Model Type          | Methods                  | Overall | Text  | Formula | Table  | Read Order |
| :------------------ | :----------------------- | :------ | :---- | :------ | :----- | :--------- |
| Pipeline Tools      |  (various)      | 0.140 to 0.646  | 0.047 to 0.987  | 0.276 to 1 | 61.3 to 82.5 | 0.069 to 0.837        |
| Expert VLMs         | (various)             | 0.139 to 0.452 | 0.047 to 0.998 | 0.297 to 0.941 | 0 to 79 | 0.069 to 0.954        |
| General VLMs          | (various)            | 0.233 to 0.314   | 0.134 to 0.409 | 0.351 to 0.606 | 66.1 to 76.4 | 0.118 to 0.251       |
| **Mix**            | **MonkeyOCR (various)**  | **0.138 to 0.154** | **0.058 to 0.134** | **0.238 to 0.529** | **76.2 to 87.5**   | **0.100 to 0.244**   |

### Text Recognition Performance Across 9 PDF Page Types

| Model Type          | Models                      | Book | Slides | Financial Report | Textbook | Exam Paper | Magazine | Academic Papers | Notes | Newspaper | Overall |
| :------------------ | :-------------------------- | :--- | :----- | :--------------- | :------- | :---------- | :------- | :-------------- | :---- | :-------- | :------ |
| Pipeline Tools      | (various)          | 0.055 to 0.131| 0.120 to 0.340| 0.024 to 0.202 | 0.090 to 0.319 | 0.107 to 0.452 | 0.072 to 0.198 | 0.024 to 0.179 | 0.150 to 0.984| 0.107 to 0.771  | 0.149 to 0.300|
| Expert VLMs         | (various)                | 0.068 to 0.734 | 0.053 to 0.958 | 0.057 to 1.000 | 0.090 to 0.820 | 0.107 to 0.930 | 0.073 to 0.830 | 0.047 to 0.214  | 0.150 to 0.991 | 0.307 to 0.871 | 0.149 to 0.806   |
| General VLMs      | (various)                 | 0.148 to 0.163 | 0.053 to 0.163 | 0.107 to 0.348 | 0.109 to 0.187| 0.119 to 0.281 | 0.100 to 0.173 | 0.134 to 0.159  | 0.150 to 0.607 | 0.681 to 0.751 | 0.188 to 0.316   |
| **Mix**            | **MonkeyOCR (various)**   | **0.046 to 0.087**| **0.056 to 0.203** | **0.024 to 0.060** | **0.090 to 0.112** | **0.107 to 0.138** | **0.073 to 0.111** | **0.024 to 0.050**   | **0.171 to 0.643** | **0.107 to 0.136** | **0.100 to 0.155**   |

### Evaluation Results of olmOCR-bench

| Model                            | Overall         |
| :------------------------------- | :-------------- |
| GOT OCR                          | 48.3 ± 1.1      |
| Marker                           | 70.1 ± 1.1      |
| MinerU                           | 61.5 ± 1.1      |
| Mistral OCR                      | 72.0 ± 1.1      |
| Nanonets OCR                     | 64.5 ± 1.1      |
| GPT-4o (No Anchor)               | 68.9 ± 1.1      |
| GPT-4o (Anchored)                | 69.9 ± 1.1      |
| Gemini Flash 2 (No Anchor)       | 57.8 ± 1.1      |
| Gemini Flash 2 (Anchored)        | 63.8 ± 1.2      |
| Qwen 2 VL (No Anchor)            | 31.5 ± 0.9      |
| Qwen 2.5 VL (No Anchor)          | 65.5 ± 1.2      |
| olmOCR v0.1.75 (No Anchor)       | 74.7 ± 1.1      |
| olmOCR v0.1.75 (Anchored)        | 75.5 ± 1.0      |
| **MonkeyOCR-pro-3B**         | **75.8 ± 1.0**  |
| **MonkeyOCR-pro-1.2B**         | **71.8 ± 1.1**  |

## Visual Demo
### Get a Quick Hands-On Experience with Our Demo: http://vlrlabmonkey.xyz:7685
*   **Upload:** PDF or Image.
*   **Parse:**  Click "Parse" to detect, recognize, and predict the structure of the input document.
*   **Test by prompt:** Select a prompt and click "Test by prompt"

### Example Screenshots:
*   **Diverse Chinese and English PDF types**
    <p align="center">
      <img src="asserts/Visualization.GIF?raw=true" width="600"/>
    </p>

*   **Formula document**
    <img src="https://v1.ax1x.com/2025/06/10/7jVLgB.jpg" alt="7jVLgB.jpg" border="0" />

*   **Table document**
    <img src="https://v1.ax1x.com/2025/06/11/7jcOaa.png" alt="7jcOaa.png" border="0" />

*   **Newspaper**
    <img src="https://v1.ax1x.com/2025/06/11/7jcP5V.png" alt="7jcP5V.png" border="0" />

*   **Financial report**
    <img src="https://v1.ax1x.com/2025/06/11/7jc10I.png" alt="7jc10I.png" border="0" />
    <img src="https://v1.ax1x.com/2025/06/11/7jcRCL.png" alt="7jcRCL.png" border="0" />

## Supported Hardware:

*   Tested on GPUs such as 3090, 4090, A6000, H800, and even 4060 (for quantized models).
*   Also works on 50-series GPUs, H200, L20, V100, 2080 Ti, and npu.

## Limitations:

*   Limited support for photographed text, handwritten content, Traditional Chinese, and multilingual text.
*   Potential issues with demo responsiveness during high traffic.

## Acknowledgments:

(List of acknowledgments from original README)

## Copyright:
Our model is intended for academic research and non-commercial use only. Contact us at xbai@hust.edu.cn or ylliu@hust.edu.cn for faster or stronger models.