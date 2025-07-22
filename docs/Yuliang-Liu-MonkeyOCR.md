# MonkeyOCR: Unlock Document Insights with Advanced OCR and Structure Recognition

**MonkeyOCR is a cutting-edge document parsing system utilizing a Structure-Recognition-Relation (SRR) paradigm to extract and understand information from documents.** ([Original Repo](https://github.com/Yuliang-Liu/MonkeyOCR))

## Key Features

*   **SRR Triplet Paradigm:** Simplifies document parsing, eliminating the need for complex, multi-tool pipelines.
*   **Superior Performance:** Outperforms leading open-source and closed-source models on various benchmarks, including OmniDocBench and olmOCR-Bench.
*   **High Speed and Efficiency:** Achieves fast processing speeds, with MonkeyOCR-pro-1.2B offering significant speed improvements.
*   **Versatile Output:** Generates markdown files, layout visualizations, and detailed JSON results for in-depth analysis.
*   **Ease of Use:** Provides straightforward installation, model download, and intuitive usage with both command-line interface and a Gradio demo.
*   **Windows & Docker Support:** Offers cross-platform support, including Windows and Docker deployment.

## Quick Start

Follow these steps to get started with MonkeyOCR:

### 1. Installation

See the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) to set up your environment.

### 2. Model Download

Download the model weights from Hugging Face or ModelScope:

```bash
pip install huggingface_hub

python tools/download_model.py -n MonkeyOCR  # or MonkeyOCR-pro-1.2B
```

```bash
pip install modelscope

python tools/download_model.py -t modelscope -n MonkeyOCR  # or MonkeyOCR-pro-1.2B
```

### 3. Inference

Use the following commands to parse documents:

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

### 4. Gradio Demo

Launch the interactive demo to visualize results:

```bash
python demo/demo_gradio.py
```

Access the demo at http://localhost:7860.

### 5. FastAPI

Run the FastAPI service for API access:

```bash
uvicorn api.main:app --port 8000
```

Access API documentation at http://localhost:8000/docs.

## Docker Deployment

1.  Navigate to the `docker` directory:

    ```bash
    cd docker
    ```

2.  **Prerequisite:** Ensure NVIDIA GPU support is available in Docker (via `nvidia-docker2`). If GPU support is not enabled, run the following to set up the environment:

    ```bash
    bash env.sh
    ```

3.  Build the Docker image:

    ```bash
    docker compose build monkeyocr
    ```

    > [!IMPORTANT]
    >
    > If your GPU is from the 20/30/40-series, V100, or similar, please build the patched Docker image for LMDeploy compatibility:
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

## Windows Support

See the [windows support guide](docs/windows_support.md) for details.

## Quantization

This model can be quantized using AWQ. Follow the instructions in the [quantization guide](docs/Quantization.md).

## Benchmark Results

MonkeyOCR delivers competitive results against leading models:

### Performance Results (OmniDocBench)

See the benchmark results tables in the original README.

### Text Recognition Performance

See the benchmark results tables in the original README.

### Performance on olmOCR-bench

See the benchmark results tables in the original README.

## Demo

Get hands-on with MonkeyOCR via the interactive demo: http://vlrlabmonkey.xyz:7685

## Citing MonkeyOCR

If you use MonkeyOCR in your research, please cite it using the provided BibTeX entry:

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

See the acknowledgments section in the original README.

## Limitations

See the limitations section in the original README.

## Copyright

See the copyright section in the original README.