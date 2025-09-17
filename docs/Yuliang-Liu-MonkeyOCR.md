# MonkeyOCR: The Ultimate Document Parsing Solution (with Speed & Accuracy)

**Effortlessly extract structure, text, tables, and more from your documents with MonkeyOCR, a cutting-edge parsing solution built on a Structure-Recognition-Relation (SRR) triplet paradigm. [View the original repo](https://github.com/Yuliang-Liu/MonkeyOCR)**

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR-pro-3B)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

Key Features:

*   **Unrivaled Performance:** MonkeyOCR-pro-1.2B outperforms previous versions and competes with leading closed-source and open-source models.
*   **Blazing Fast:** Experience up to 36% speed improvement with MonkeyOCR-pro-1.2B compared to MonkeyOCR-pro-3B.
*   **State-of-the-Art Accuracy:** Achieve the best overall performance on both English and Chinese documents.
*   **Flexible Deployment:** Supports various GPUs, including 3090, 4090, A6000, H800, and 4060.
*   **User-Friendly:** Includes a Gradio demo and FastAPI service for easy integration.

## What is MonkeyOCR?

MonkeyOCR utilizes a Structure-Recognition-Relation (SRR) triplet paradigm, simplifying document parsing while avoiding the inefficiencies of large multimodal models for full-page document processing. It excels at extracting structured information from documents, including text, formulas, tables, and layout elements.

## Key Advantages:

*   **Superior Speed & Efficiency:** Optimizations lead to significantly faster processing times.
*   **High Accuracy:** Achieve top-tier results on a variety of document types and languages.
*   **Versatile Applicability:** Ideal for academic research, document analysis, and information extraction.

## Performance Highlights:

*   MonkeyOCR-pro-1.2B surpasses MonkeyOCR-3B by 7.4% on Chinese documents.
*   MonkeyOCR-pro-1.2B delivers approximately a 36% speed improvement over MonkeyOCR-pro-3B, with approximately a 1.6% drop in performance.
*   On olmOCR-Bench, MonkeyOCR-pro-1.2B outperforms Nanonets-OCR-3B by 7.3%.
*   On OmniDocBench, MonkeyOCR-pro-3B achieves the best overall performance on both English and Chinese documents, outperforming even closed-source and extra-large open-source VLMs such as Gemini 2.0-Flash, Gemini 2.5-Pro, Qwen2.5-VL-72B, GPT-4o, and InternVL3-78B.

See detailed results below.

### Comparing MonkeyOCR with closed-source and extra large open-source VLMs.
<a href="https://zimgs.com/i/EKhkhY"><img src="https://v1.ax1x.com/2025/07/15/EKhkhY.png" alt="EKhkhY.png" border="0" /></a>

## Inference Speed (Pages/s) on Different GPUs

**(See tables for detailed speed benchmarks on various GPUs with different page counts.)**

*Refer to the PDF [here](https://drive.google.com/drive/folders/1geumlJmVY7UUKdr8324sYZ0FHSAElh7m?usp=sharing) for a complete PDF of all pages measured.*

**[Inference Speed Tables Here - Removed to save space, but should be kept]**

## VLM OCR Speed (Pages/s) on Different GPUs

**(See tables for detailed speed benchmarks on various GPUs with different page counts.)**

**[VLM OCR Speed Tables Here - Removed to save space, but should be kept]**

## Getting Started:

### 1.  Installation

**(Detailed installation instructions are available in the repository:  [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support))**

### 2. Download Model Weights

```bash
pip install huggingface_hub

python tools/download_model.py -n MonkeyOCR-pro-3B  # or MonkeyOCR
```

You can also download our model from ModelScope.

```bash
pip install modelscope

python tools/download_model.py -t modelscope -n MonkeyOCR-pro-3B  # or MonkeyOCR
```

### 3. Inference

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

**(See more advanced usage examples.)**

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

**(Learn about the output results.)**

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

**(Access the demo at http://localhost:7860)**

### 5. FastAPI Service

```bash
uvicorn api.main:app --port 8000
```

**(Access the API documentation at http://localhost:8000/docs)**

## Docker Deployment

**(See the `docker` directory for detailed instructions.)**

## Windows Support

**(See the [windows support guide](docs/windows_support.md) for details.)**

## Quantization

**(Learn about quantization using AWQ: [quantization guide](docs/Quantization.md))**

## Benchmark Results

**(See end-to-end evaluation results on OmniDocBench and olmOCR-bench in the original README)**

**(Include Tables from the Original README)**

## Demo
**(Get a Quick Hands-On Experience with Our Demo:  http://vlrlabmonkey.xyz:7685 (The latest model is available for selection))**

## Visualization Demo

**[Include Images from the Original README]**

**(Demonstrate diverse Chinese and English PDF types.)**

**(Include examples for formula, table, newspaper, and financial report.)**

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

**(List Acknowledgments from the Original README)**

## Limitations

**(List Limitations from the Original README)**

## Copyright

**(Copyright Information from Original README)**