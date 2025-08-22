# MonkeyOCR: Advanced Document Parsing with Structure-Recognition-Relation Paradigm

**Effortlessly extract structured data from documents with MonkeyOCR, the state-of-the-art document parsing tool, achieving superior accuracy and speed.** 
[View the original repository](https://github.com/Yuliang-Liu/MonkeyOCR)

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

MonkeyOCR offers a novel approach to document parsing using a Structure-Recognition-Relation (SRR) triplet paradigm, excelling in accuracy and efficiency.

**Key Features:**

*   **Superior Accuracy:** MonkeyOCR-pro-3B outperforms leading open-source and closed-source VLMs on the OmniDocBench.
*   **Blazing Speed:** Achieve up to 36% faster processing speeds compared to previous versions.
*   **Versatile:** Parses various document types, including books, financial reports, academic papers, and more.
*   **Flexible Deployment:** Supports various hardware configurations and offers Docker and FastAPI integration.
*   **Open Source:** Built upon open-source foundations, fostering community collaboration and improvement.

## Key Improvements and Results

*   **MonkeyOCR-pro-1.2B:**  A lean and fast version that outperforms MonkeyOCR-3B.
*   **Benchmark Dominance:** MonkeyOCR-pro-3B and 1.2B achieve state-of-the-art results across multiple benchmarks, including OmniDocBench and olmOCR-bench, often surpassing even closed-source models.
*   **Chinese Document Performance:** Excels with significantly improved performance on Chinese documents.

### Performance Comparison

**Comparison with closed-source and extra-large open-source VLMs**
<a href="https://zimgs.com/i/EKhkhY"><img src="https://v1.ax1x.com/2025/07/15/EKhkhY.png" alt="EKhkhY.png" border="0" /></a>

## Inference Speed

**Inference Speed (Pages/s)**
See tables in original README

## Getting Started

### Installation

1.  **Install Dependencies:** See the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support)
2.  **Download Model Weights:**
    ```bash
    pip install huggingface_hub
    python tools/download_model.py -n MonkeyOCR-pro-1.2B  # or MonkeyOCR
    ```
    Also available from ModelScope:
    ```bash
    pip install modelscope
    python tools/download_model.py -t modelscope -n MonkeyOCR-pro-1.2B  # or MonkeyOCR
    ```
3.  **Inference:**
    ```bash
    python parse.py input_path
    ```
    See the original README for more detailed usage examples for file parsing, single-task recognition, folder processing, and more.

4.  **Gradio Demo:** Run the demo with:
    ```bash
    python demo/demo_gradio.py
    ```
    Access the demo at http://localhost:7860.

5.  **FastAPI:** Run the FastAPI service
    ```bash
    uvicorn api.main:app --port 8000
    ```
    Access the API documentation at http://localhost:8000/docs.

## Deployment

### Docker Deployment

1.  Navigate to the `docker` directory:

    ```bash
    cd docker
    ```

2.  **Prerequisite:** Ensure NVIDIA GPU support is available in Docker (via `nvidia-docker2`).
    If GPU support is not enabled, run the following to set up the environment:

    ```bash
    bash env.sh
    ```

3.  Build the Docker image:

    ```bash
    docker compose build monkeyocr
    ```

    > [!IMPORTANT]
    >
    > If your GPU is from the 20/30/40-series, V100, L20/L40 or similar, please build the patched Docker image for LMDeploy compatibility:
    >
    > ```bash
    > docker compose build monkeyocr-fix
    > ```
    >
    > Otherwise, you may encounter the following error: `triton.runtime.errors.OutOfResources: out of resource: shared memory`

4.  Run the container with the Gradio demo (accessible on port 7860):

    ```bash
    docker compose up monkeyocr-demo
    ```

    Alternatively, start an interactive development environment:

    ```bash
    docker compose run --rm monkeyocr-dev
    ```

5.  Run the FastAPI service (accessible on port 7861):
    ```bash
    docker compose up monkeyocr-api
    ```
    Once the API service is running, you can access the API documentation at http://localhost:7861/docs to explore available endpoints.

### Windows Support

See the [windows support guide](docs/windows_support.md) for details.

### Quantization

This model can be quantized using AWQ. Follow the instructions in the [quantization guide](docs/Quantization.md).

## Benchmark Results

Detailed evaluation results are available in the original README.

## Demo and Visualizations

Experience MonkeyOCR firsthand with our interactive demo: http://vlrlabmonkey.xyz:7685

### Supports diverse Chinese and English PDF types
<p align="center">
  <img src="asserts/Visualization.GIF?raw=true" width="600"/>
</p>

### Example for formula document
<img src="https://v1.ax1x.com/2025/06/10/7jVLgB.jpg" alt="7jVLgB.jpg" border="0" />

### Example for table document
<img src="https://v1.ax1x.com/2025/06/11/7jcOaa.png" alt="7jcOaa.png" border="0" />

### Example for newspaper
<img src="https://v1.ax1x.com/2025/06/11/7jcP5V.png" alt="7jcP5V.png" border="0" />

### Example for financial report
<img src="https://v1.ax1x.com/2025/06/11/7jc10I.png" alt="7jc10I.png" border="0" />
<img src="https://v1.ax1x.com/2025/06/11/7jcRCL.png" alt="7jcRCL.png" border="0" />

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

We are very grateful for the resources and collaborations.

## Limitations

MonkeyOCR does not fully support some character sets and is currently deployed on a single GPU.

## Copyright

The model is intended for academic research and non-commercial use only.