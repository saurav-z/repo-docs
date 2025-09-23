# MonkeyOCR: Effortless Document Parsing with Intelligent Structure Recognition (OCR)

**MonkeyOCR simplifies document processing with a cutting-edge Structure-Recognition-Relation (SRR) triplet paradigm, delivering superior performance and speed.**  ([View the original repository](https://github.com/Yuliang-Liu/MonkeyOCR))

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR-pro-3B)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

## Key Features

*   **Advanced SRR Paradigm:** Streamlines document parsing with a novel Structure-Recognition-Relation approach.
*   **Superior Performance:** Achieves state-of-the-art results, outperforming other models on OmniDocBench.
*   **Fast Inference:** Offers impressive speeds, with significant improvements over previous versions.
*   **Versatile Compatibility:** Supports a range of GPUs, from 3090 to H800 and 4090, and even 4060, supporting both quantization.
*   **Easy to Use:** Offers a simple and user-friendly Gradio demo and a fast API for effortless integration.
*   **Open Source:** Based on open-source code and resources, so that anyone can test and use the library.

## What's New

*   **July 10, 2025:** 🚀 Released [MonkeyOCR-pro-1.2B](https://huggingface.co/echo840/MonkeyOCR-pro-1.2B), a leaner and faster version.
*   **June 12, 2025:** 🚀 Trending on [Hugging Face](https://huggingface.co/models?sort=trending).
*   **June 05, 2025:** 🚀 Released [MonkeyOCR](https://huggingface.co/echo840/MonkeyOCR), the English and Chinese documents parsing model.

## Quick Start

### 1.  Installation
Follow the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) to set up your environment.

### 2. Model Download

Download the model weights from Hugging Face:

```bash
pip install huggingface_hub
python tools/download_model.py -n MonkeyOCR-pro-3B  # or MonkeyOCR
```

Alternatively, download from ModelScope:

```bash
pip install modelscope
python tools/download_model.py -t modelscope -n MonkeyOCR-pro-3B  # or MonkeyOCR
```

### 3.  Inference

Run the parsing script on PDF or image files:

```bash
python parse.py input_path
```

Refer to the original README for more parsing options.

### 4.  Gradio Demo

Launch the interactive demo:

```bash
python demo/demo_gradio.py
```

### 5. Fast API
Launch the API using `uvicorn api.main:app --port 8000` and access the documentation at http://localhost:8000/docs.

## Docker Deployment

Build and run MonkeyOCR with Docker for easy deployment.

1.  Navigate to the `docker` directory:

    ```bash
    cd docker
    ```

2.  Build the Docker image:

    ```bash
    docker compose build monkeyocr
    ```

3. Run the container with the Gradio demo (accessible on port 7860):

   ```bash
   docker compose up monkeyocr-demo
   ```

4. Run the FastAPI service (accessible on port 7861):
   ```bash
   docker compose up monkeyocr-api
   ```
   Once the API service is running, you can access the API documentation at http://localhost:7861/docs to explore available endpoints.

### Important Notes

*   **GPU Support:** Requires an NVIDIA GPU for optimal performance.
*   **Quantization:** Utilize AWQ for model quantization, as shown in the [quantization guide](docs/Quantization.md).

## Benchmark Results

See the detailed results in the original README.  MonkeyOCR outperforms other methods on OmniDocBench and olmOCR-bench.

## [Visualization Demo](http://vlrlabmonkey.xyz:7685)

Experience MonkeyOCR's capabilities firsthand with our interactive demo.  Upload a document and see it parsed instantly!

## Citation

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

We thank the open-source community for the models and datasets that have contributed to the success of the project.