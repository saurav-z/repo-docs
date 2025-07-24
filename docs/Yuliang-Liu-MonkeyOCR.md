# MonkeyOCR: Effortlessly Parse Documents with AI

MonkeyOCR is a cutting-edge document parsing solution that leverages a Structure-Recognition-Relation (SRR) triplet paradigm to accurately extract information from various document types. **Explore MonkeyOCR on GitHub: [https://github.com/Yuliang-Liu/MonkeyOCR](https://github.com/Yuliang-Liu/MonkeyOCR)**

## Key Features:

*   **Superior Performance:** MonkeyOCR achieves state-of-the-art results, outperforming existing solutions on key benchmarks.
    *   MonkeyOCR-pro-1.2B surpasses MonkeyOCR-3B by 7.4% on Chinese documents.
    *   MonkeyOCR-pro-1.2B delivers approximately a 36% speed improvement over MonkeyOCR-pro-3B, with approximately 1.6% drop in performance.
    *   On olmOCR-Bench, MonkeyOCR-pro-1.2B outperforms Nanonets-OCR-3B by 7.3%.
    *   On OmniDocBench, MonkeyOCR-pro-3B achieves the best overall performance on both English and Chinese documents, outperforming even closed-source and extra-large open-source VLMs such as Gemini 2.0-Flash, Gemini 2.5-Pro, Qwen2.5-VL-72B, GPT-4o, and InternVL3-78B.
*   **SRR Triplet Paradigm:** Simplifies document parsing pipelines, avoiding the inefficiencies of large multimodal models for full-page processing.
*   **High Accuracy:**  Achieves superior results across a range of document types.
*   **Fast Inference:** Optimized for speed, with performance detailed in the tables below.
*   **Flexible Deployment:** Supports local installation, Docker deployment, and an API for easy integration.
*   **Open Source:** Freely available for research and non-commercial use.
*   **Gradio Demo:** Quick hands-on experience:  [http://vlrlabmonkey.xyz:7685](http://vlrlabmonkey.xyz:7685).

## Performance and Benchmarks:

### Comparison with Closed-Source and Open-Source VLMs

[Insert Image of Comparison Table Here]
<!--  Image Source: https://v1.ax1x.com/2025/07/15/EKhkhY.png -->

### Inference Speed (Pages/s) on Different GPUs

#### MonkeyOCR-pro-3B
| Model               | GPU    | 50 Pages | 100 Pages | 300 Pages | 500 Pages | 1000 Pages |
| ------------------- | ------ | -------- | --------- | --------- | --------- | ---------- |
| MonkeyOCR-pro-3B | 3090   | 0.492    | 0.484     | 0.497     | 0.492     | 0.496      |
| MonkeyOCR-pro-3B  | A6000  | 0.585    | 0.587     | 0.609     | 0.598     | 0.608      |
| MonkeyOCR-pro-3B  | H800   | 0.923    | 0.768     | 0.897     | 0.930     | 0.891      |
| MonkeyOCR-pro-3B  | 4090   | 0.972    | 0.969     | 1.006     | 0.986     | 1.006      |
#### MonkeyOCR-pro-1.2B
| Model               | GPU    | 50 Pages | 100 Pages | 300 Pages | 500 Pages | 1000 Pages |
| ------------------- | ------ | -------- | --------- | --------- | --------- | ---------- |
| MonkeyOCR-pro-1.2B | 3090   | 0.615    | 0.660     | 0.677     | 0.687     | 0.683      |
| MonkeyOCR-pro-1.2B  | A6000  | 0.709    | 0.786     | 0.825     | 0.829     | 0.825      |
| MonkeyOCR-pro-1.2B  | H800   | 0.965    | 1.082     | 1.101     | 1.145     | 1.015      |
| MonkeyOCR-pro-1.2B  | 4090   | 1.194    | 1.314     | 1.436     | 1.442     | 1.434      |
### VLM OCR Speed (Pages/s) on Different GPUs

#### MonkeyOCR-pro-3B
| Model               | GPU    | 50 Pages | 100 Pages | 300 Pages | 500 Pages | 1000 Pages |
| ------------------- | ------ | -------- | --------- | --------- | --------- | ---------- |
| MonkeyOCR-pro-3B | 3090   | 0.705    | 0.680     | 0.711     | 0.700     | 0.724      |
| MonkeyOCR-pro-3B  | A6000  | 0.885    | 0.860     | 0.915     | 0.892     | 0.934      |
| MonkeyOCR-pro-3B  | H800   | 1.371    | 1.135     | 1.339     | 1.433     | 1.509      |
| MonkeyOCR-pro-3B  | 4090   | 1.321    | 1.300     | 1.384     | 1.343     | 1.410      |
#### MonkeyOCR-pro-1.2B
| Model               | GPU    | 50 Pages | 100 Pages | 300 Pages | 500 Pages | 1000 Pages |
| ------------------- | ------ | -------- | --------- | --------- | --------- | ---------- |
| MonkeyOCR-pro-1.2B | 3090   | 0.919    | 1.086     | 1.166     | 1.182     | 1.199      |
| MonkeyOCR-pro-1.2B  | A6000  | 1.177    | 1.361     | 1.506     | 1.525     | 1.569      |
| MonkeyOCR-pro-1.2B  | H800   | 1.466    | 1.719     | 1.763     | 1.875     | 1.650      |
| MonkeyOCR-pro-1.2B  | 4090   | 1.759    | 1.987     | 2.260     | 2.345     | 2.415      |

## Quick Start

### 1. Install
See the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) to set up your environment.
### 2. Download Model Weights
Download our model from Huggingface.
```python
pip install huggingface_hub

python tools/download_model.py -n MonkeyOCR  # or MonkeyOCR-pro-1.2B
```
You can also download our model from ModelScope.

```python
pip install modelscope

python tools/download_model.py -t modelscope -n MonkeyOCR  # or MonkeyOCR-pro-1.2B
```
### 3. Inference
You can parse a file or a directory containing PDFs or images using the following commands:
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
Once the demo is running, you can access it at http://localhost:7860.

### 5. Fast API
You can start the MonkeyOCR FastAPI service with the following command:
```bash
uvicorn api.main:app --port 8000
```
Once the API service is running, you can access the API documentation at http://localhost:8000/docs to explore available endpoints.

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

### End-to-End Evaluation Results

[Insert the table "1. The end-to-end evaluation results of different tasks." here]

### End-to-End Text Recognition Performance

[Insert the table "2. The end-to-end text recognition performance across 9 PDF page types." here]

### olmOCR-bench Evaluation Results

[Insert the table "3. The evaluation results of olmOCR-bench." here]

##  Visualization Demo

Experience MonkeyOCR's capabilities through our interactive demo: [http://vlrlabmonkey.xyz:7685](http://vlrlabmonkey.xyz:7685). The latest model is available for selection.

> Our demo is simple and easy to use:
>
> 1.  Upload a PDF or image.
> 2.  Click “Parse (解析)” to let the model perform structure detection, content recognition, and relationship prediction on the input document. The final output will be a markdown-formatted version of the document.
> 3.  Select a prompt and click “Test by prompt” to let the model perform content recognition on the image based on the selected prompt.

###  Support Diverse Chinese and English PDF Types

[Insert the image  "Visualization.GIF?raw=true" here. ]

###  Example Formula Document

[Insert the image "7jVLgB.jpg" here]

###  Example Table Document

[Insert the image "7jcOaa.png" here]

###  Example Newspaper

[Insert the image "7jcP5V.png" here]

###  Example Financial Report

[Insert the images "7jc10I.png" and "7jcRCL.png" here]

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

[See original for acknowledgments.]

## Limitations

[See original for limitations.]

## Copyright

[See original for copyright information.]