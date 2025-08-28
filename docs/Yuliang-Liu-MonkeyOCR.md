# MonkeyOCR: Unlock Document Structure with AI-Powered Parsing

**Transform documents into structured data with MonkeyOCR, a cutting-edge solution that excels at document parsing using a Structure-Recognition-Relation (SRR) triplet paradigm. [See the original repository](https://github.com/Yuliang-Liu/MonkeyOCR) for more details.**

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

## Key Features

*   **SRR Paradigm:** Simplifies document parsing, avoiding complex multi-tool pipelines.
*   **Superior Performance:** MonkeyOCR-pro-1.2B outperforms other models on key benchmarks.
    *   7.4% improvement on Chinese documents compared to MonkeyOCR-3B.
    *   Up to 36% speed improvement over MonkeyOCR-pro-3B.
    *   Outperforms Nanonets-OCR-3B by 7.3% on olmOCR-Bench.
    *   Achieves top results on OmniDocBench, even surpassing closed-source and extra-large open-source VLMs.
*   **Speed and Efficiency:** Optimized for fast inference across various GPU configurations.
*   **Easy to Use:** Includes a user-friendly Gradio demo and FastAPI service for seamless integration.
*   **Open Source & Community Driven:**  Supported by a dedicated community, offering flexibility and customization.

## What's New

*   **2025.07.10:**  🚀 Released [MonkeyOCR-pro-1.2B](https://huggingface.co/echo840/MonkeyOCR-pro-1.2B), a faster, leaner model with improved accuracy and efficiency.
*   **2025.06.12:**  🚀 Trending on [Hugging Face](https://huggingface.co/models?sort=trending).
*   **2025.06.05:**  🚀 Released [MonkeyOCR](https://huggingface.co/echo840/MonkeyOCR), supporting English and Chinese document parsing.

## Quick Start

Follow these steps to quickly set up and run MonkeyOCR:

### 1.  Install MonkeyOCR

Refer to the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) for detailed instructions.

### 2.  Download Model Weights

Choose your preferred method:

*   **Hugging Face:**

    ```bash
    pip install huggingface_hub
    python tools/download_model.py -n MonkeyOCR-pro-3B  # or MonkeyOCR
    ```
*   **ModelScope:**

    ```bash
    pip install modelscope
    python tools/download_model.py -t modelscope -n MonkeyOCR-pro-3B  # or MonkeyOCR
    ```

### 3.  Inference

Run the `parse.py` script with your input file or directory:

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
  python parse.py /path/to/folder -g 10 -m            # Group files and merge text blocks in output
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

```bash
python demo/demo_gradio.py
```

Access the demo at http://localhost:7860.

### 5. Fast API

```bash
uvicorn api.main:app --port 8000
```

Explore the API documentation at http://localhost:8000/docs.

> [!TIP]
> To improve API concurrency performance, consider configuring the inference backend as `lmdeploy_queue` or `vllm_queue`.

## Docker Deployment

### 1. Navigate to the `docker` directory:

   ```bash
   cd docker
   ```

### 2. Prerequisite:

Ensure NVIDIA GPU support is available in Docker (via `nvidia-docker2`).
If GPU support is not enabled, run the following to set up the environment:

   ```bash
   bash env.sh
   ```

### 3. Build the Docker image:

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

### 4. Run the container with the Gradio demo (accessible on port 7860):

   ```bash
   docker compose up monkeyocr-demo
   ```

   Alternatively, start an interactive development environment:

   ```bash
   docker compose run --rm monkeyocr-dev
   ```

### 5. Run the FastAPI service (accessible on port 7861):
   ```bash
   docker compose up monkeyocr-api
   ```
   Once the API service is running, you can access the API documentation at http://localhost:7861/docs to explore available endpoints.

## Windows Support

See the [windows support guide](docs/windows_support.md) for details.

## Quantization

This model can be quantized using AWQ. Follow the instructions in the [quantization guide](docs/Quantization.md).

## Benchmark Results

[Detailed benchmark results comparing MonkeyOCR's performance against other models are available in the original README.](https://github.com/Yuliang-Liu/MonkeyOCR#benchmark-results)

## Demo Visualization

[Explore the live demo](http://vlrlabmonkey.xyz:7685) to experience MonkeyOCR's capabilities.

### Support Diverse Chinese and English PDF Types

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

We extend our gratitude to the contributors and creators of [the listed libraries, datasets, and models](https://github.com/Yuliang-Liu/MonkeyOCR#acknowledgments).

## Limitations

*   Limited support for photographed text, handwritten content, Traditional Chinese characters, and multilingual text.
*   Demo performance may be affected during periods of high traffic.

## Copyright

Our model is intended for academic research and non-commercial use only. Contact us at xbai@hust.edu.cn or ylliu@hust.edu.cn for inquiries about commercial use or for faster/stronger model versions.