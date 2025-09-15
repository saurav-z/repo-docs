# MonkeyOCR: Unlock Document Insights with Powerful Structure Recognition

**MonkeyOCR revolutionizes document processing by employing a Structure-Recognition-Relation (SRR) triplet paradigm, delivering superior performance in understanding and extracting information from various document types.**  ([Original Repo](https://github.com/Yuliang-Liu/MonkeyOCR))

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR-pro-3B)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

> **MonkeyOCR: Document Parsing with a Structure-Recognition-Relation Triplet Paradigm**
> Zhang Li, Yuliang Liu, Qiang Liu, Zhiyin Ma, Ziyang Zhang, Shuo Zhang, Zidun Guo, Jiarui Zhang, Xinyu Wang, Xiang Bai
[![arXiv](https://img.shields.io/badge/Arxiv-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![Source_code](https://img.shields.io/badge/Code-Available-white)](README.md)
[![Model Weight](https://img.shields.io/badge/HuggingFace-gray)](https://huggingface.co/echo840/MonkeyOCR)
[![Model Weight](https://img.shields.io/badge/ModelScope-green)](https://modelscope.cn/models/l1731396519/MonkeyOCR)
[![Public Courses](https://img.shields.io/badge/Openbayes-yellow)](https://openbayes.com/console/public/tutorials/91ESrGvEvBq)
[![Demo](https://img.shields.io/badge/Demo-blue)](http://vlrlabmonkey.xyz:7685/)

## Key Features

*   **Superior Performance:** Outperforms leading closed-source and open-source VLMs on various benchmarks, including OmniDocBench.
*   **SRR Triplet Paradigm:** Employs a novel structure recognition approach, simplifying the document processing pipeline.
*   **Faster and Efficient:** MonkeyOCR-pro-1.2B offers significant speed improvements over previous versions with comparable performance.
*   **Versatile Support:** Handles diverse document types, including academic papers, financial reports, and more.
*   **Open Source:** Built on open-source code and models, and open to the community.

## Performance Highlights

*   **MonkeyOCR-pro-1.2B** surpasses **MonkeyOCR-3B** by 7.4% on Chinese documents.
*   **MonkeyOCR-pro-1.2B** is approximately **36% faster** than **MonkeyOCR-pro-3B**, with about a 1.6% drop in performance.
*   On **olmOCR-Bench, MonkeyOCR-pro-1.2B** outperforms **Nanonets-OCR-3B** by 7.3%.
*   On **OmniDocBench**, **MonkeyOCR-pro-3B** achieves the best overall performance on both English and Chinese documents, exceeding even closed-source and extra-large open-source VLMs like **Gemini 2.0-Flash, Gemini 2.5-Pro, Qwen2.5-VL-72B, GPT-4o, and InternVL3-78B.**

### Comparison with Leading Models

[Include the image of the comparison table here.  This helps with SEO by providing visual context.]

## Inference Speed

[Include the Inference Speed tables for both models.  These are important for demonstrating practical value.]

## VLM OCR Speed

[Include the VLM OCR Speed tables here.]

## Supported Hardware

MonkeyOCR has been tested on a wide variety of hardware configurations, including 3090, 4090, A6000, H800, A100, and 4060 GPUs.  Community contributions have extended support to 50-series GPUs, H200, L20, V100, 2080 Ti and npu.

## Getting Started

### Installation

1.  **Install Dependencies:** Follow the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) for environment setup.
2.  **Download Model Weights:** Choose your preferred download method:

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
3.  **Inference:** Use the `parse.py` script for processing documents:

    ```bash
    # Replace input_path with your file/directory

    python parse.py input_path  # Basic Parsing
    python parse.py input_path -g 20  # Group by page count
    python parse.py input_path -t text/formula/table  # Single-task recognition
    python parse.py input_path -s  # Split results by pages
    python parse.py input_path -o ./output -c config.yaml  # Specify output
    ```

    *   **Example Usage:**

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

4.  **Gradio Demo:** Run the interactive demo:

    ```bash
    python demo/demo_gradio.py
    ```
    Access the demo at http://localhost:7860.

5.  **FastAPI:** Run the API service:

    ```bash
    uvicorn api.main:app --port 8000
    ```
    API documentation: http://localhost:8000/docs

## Output Results

MonkeyOCR generates:

1.  **Markdown File:** (`your.md`) - Parsed content.
2.  **Layout Results:** (`your_layout.pdf`) - Layout results drawed on origin PDF.
3.  **Intermediate Block Results:** (`your_middle.json`) - Detailed block information.

## Docker Deployment

1.  `cd docker`
2.  Prerequisites: `bash env.sh` (if NVIDIA GPU support is missing)
3.  Build: `docker compose build monkeyocr`
    *   If using 20/30/40-series, V100, L20/L40 GPUs, use `docker compose build monkeyocr-fix` to build LMDeploy compatible image.
4.  Run demo: `docker compose up monkeyocr-demo` (port 7860)
    *   Dev environment: `docker compose run --rm monkeyocr-dev`
5.  Run API: `docker compose up monkeyocr-api` (port 7861, docs at http://localhost:7861/docs)

## Windows Support

See the [windows support guide](docs/windows_support.md) for details.

## Quantization

Quantize using AWQ following the [quantization guide](docs/Quantization.md).

## Benchmark Results

[Include the Benchmark tables here.]

## Demo

Get a quick hands-on experience with our demo: http://vlrlabmonkey.xyz:7685

[Include the images of the examples here.]

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

[Include the acknowledgments here.]

## Limitations

*   Limited support for photographed text, handwriting, Traditional Chinese, and multilingual text.
*   Demo may experience slowdowns during high traffic due to single GPU deployment.  Processing time includes overhead.

## Copyright

[Include the copyright and contact information here.]