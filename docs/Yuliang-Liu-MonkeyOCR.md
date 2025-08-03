# MonkeyOCR: Effortless Document Parsing with a Structure-Recognition-Relation Triplet Paradigm

MonkeyOCR empowers you to effortlessly extract structured data from documents, offering a streamlined approach to document processing. [(See the original repo here)](https://github.com/Yuliang-Liu/MonkeyOCR)

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

**Key Features:**

*   **SRR Triplet Paradigm:** Simplifies document parsing by using a Structure-Recognition-Relation (SRR) triplet paradigm, avoiding complex modular pipelines.
*   **Superior Performance:** MonkeyOCR-pro-1.2B and MonkeyOCR-pro-3B deliver state-of-the-art results, outperforming even closed-source and extra-large open-source VLMs on various benchmarks.
*   **High Speed:**  Provides significant speed improvements, with up to a 36% speed increase compared to previous versions.
*   **Multi-GPU Support:** Optimized for performance on various GPUs, including 3090, 4090, A6000, H800, and others.
*   **Flexible Deployment:** Supports local installation, Gradio demo, and Docker deployment, including configurations for GPU-enabled environments and fast API.
*   **Quantization Support:**  Compatible with AWQ quantization for optimized model deployment.
*   **Comprehensive Output:** Generates Markdown files, layout results (PDF), and detailed block results (JSON) for versatile use.

## Quick Start

Get up and running with MonkeyOCR in no time:

### 1.  Install Dependencies
See the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) to set up your environment.

### 2. Download Model Weights

Download the model weights from Hugging Face or ModelScope:

```bash
pip install huggingface_hub
python tools/download_model.py -n MonkeyOCR  # or MonkeyOCR-pro-1.2B
```

```bash
pip install modelscope
python tools/download_model.py -t modelscope -n MonkeyOCR  # or MonkeyOCR-pro-1.2B
```

### 3.  Inference

Use the following command to parse documents:

```bash
python parse.py input_path
```

*   **input_path:**  Replace with the path to your PDF, image, or directory containing documents.

**Additional Options:**

*   `-g <page_group_size>`: Process multiple files at once.
*   `-t <task>`: Specify single tasks, like text, formula, or table extraction.
*   `-s`: Split PDF results by page.
*   `-o <output_dir>`: Specify an output directory.
*   `-c <config_file>`: Use a custom configuration file.

<details>
<summary><b>More usage examples</b></summary>

```bash
# Single file processing
python parse.py input.pdf                           # Parse single PDF file
python parse.py input.pdf -o ./output               # Parse with custom output dir
python parse.py input.pdf -s                        # Parse PDF with page splitting
python parse.py image.jpg                           # Parse single image file

# Single task recognition
python parse.py image.jpg -t text                   # Text recognition from image
python parse.py image.jpg -t formula                # Formula recognition from image
python parse.py image.jpg -t table                  # Table recognition from image
python parse.py document.pdf -t text                # Text recognition from all PDF pages

# Folder processing (all files individually)
python parse.py /path/to/folder                     # Parse all files in folder
python parse.py /path/to/folder -s                  # Parse with page splitting
python parse.py /path/to/folder -t text             # Single task recognition for all files

# Multi-file grouping (batch processing by page count)
python parse.py /path/to/folder -g 5                # Group files with max 5 total pages
python parse.py /path/to/folder -g 10 -s            # Group files with page splitting
python parse.py /path/to/folder -g 8 -t text        # Group files for single task recognition

# Advanced configurations
python parse.py input.pdf -c model_configs.yaml     # Custom model configuration
python parse.py /path/to/folder -g 15 -s -o ./out   # Group files, split pages, custom output
python parse.py input.pdf --pred-abandon            # Enable predicting abandon elements
```

</details>

<details>
<summary><b>Output Results</b></summary>

MonkeyOCR mainly generates three types of output files:

1.  **Processed Markdown File** (`your.md`): The final parsed document content in markdown format, containing text, formulas, tables, and other structured elements.
2.  **Layout Results** (`your_layout.pdf`): The layout results drawed on origin PDF.
3.  **Intermediate Block Results** (`your_middle.json`): A JSON file containing detailed information about all detected blocks, including:
    *   Block coordinates and positions
    *   Block content and type information
    *   Relationship information between blocks

These files provide both the final formatted output and detailed intermediate results for further analysis or processing.

</details>

### 4. Gradio Demo

Run a user-friendly demo:

```bash
python demo/demo_gradio.py
```

Access the demo at http://localhost:7860.

### 5.  FastAPI Service

Deploy MonkeyOCR as an API:

```bash
uvicorn api.main:app --port 8000
```

Access API documentation at http://localhost:8000/docs.

> [!TIP]
> To improve API concurrency performance, consider configuring the inference backend as `lmdeploy_queue` or `vllm_queue`.

## Docker Deployment

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

See the original README for the detailed evaluation results on various benchmarks, demonstrating MonkeyOCR's superior performance.

## Visualization Demo

Try out a live demo: [http://vlrlabmonkey.xyz:7685](http://vlrlabmonkey.xyz:7685) (Select latest model.)

### Support diverse Chinese and English PDF types

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

If you wish to refer to the baseline results published here, please use the following BibTeX entries:

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

Thanks to the contributors and the projects mentioned in the original README file.

## Limitations

*   Limited support for photographed text, handwritten content, Traditional Chinese, and multilingual text.
*   Demo may experience slowdowns during high traffic due to single GPU deployment.
*   Processing time in the demo includes overhead in addition to computation.

## Copyright

This model is intended for academic research and non-commercial use only. For inquiries regarding faster (smaller) or stronger models, please contact xbai@hust.edu.cn or ylliu@hust.edu.cn.