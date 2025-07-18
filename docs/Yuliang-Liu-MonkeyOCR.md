# MonkeyOCR: Revolutionizing Document Parsing with SRR Triplet Paradigm

**Unlock unparalleled document understanding with MonkeyOCR, a cutting-edge solution built on a Structure-Recognition-Relation (SRR) triplet paradigm.** [Explore the original repository](https://github.com/Yuliang-Liu/MonkeyOCR) for more details.

*   **State-of-the-Art Performance:** MonkeyOCR-pro-1.2B excels, surpassing the 3B version on Chinese documents by 7.4%.
*   **Blazing-Fast Speed:** Enjoy a 36% speed boost with MonkeyOCR-pro-1.2B compared to the 3B model, with a minimal performance trade-off (1.6% drop).
*   **Superior Accuracy:** MonkeyOCR-pro-1.2B outperforms Nanonets-OCR-3B by 7.3% on olmOCR-Bench, and MonkeyOCR-pro-3B excels on OmniDocBench.
*   **Versatile Compatibility:** Works seamlessly with English and Chinese documents.
*   **Easy to Use:** Comes with a simple API and Gradio demo.

## Key Features

*   **SRR Triplet Paradigm:** Simplifies the document processing pipeline, avoiding the complexities of large multimodal models.
*   **High-Speed Inference:** Optimized for efficient processing, with benchmark results showcasing impressive pages-per-second performance on various GPUs.
*   **Comprehensive Output:** Delivers parsed documents in markdown format, with layout results and intermediate block results for detailed analysis.
*   **Model Weights & Demos Available:** Quickly get started with pre-trained models available on Hugging Face, ModelScope, and a user-friendly demo.
*   **Flexible Deployment:** Supports local installation, Docker deployment (including NVIDIA GPU support) and a FastAPI service.

## Performance Highlights

**MonkeyOCR Demonstrates Superior Performance, outperforming closed-source and large open-source VLM models across multiple benchmarks:**

![MonkeyOCR vs Competitors](https://v1.ax1x.com/2025/07/15/EKhkhY.png)

## Quick Start

### 1. Installation
See the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) to setup your environment.

### 2. Download Model Weights
Download the model from Hugging Face or ModelScope:

```bash
# Hugging Face
pip install huggingface_hub
python tools/download_model.py -n MonkeyOCR  # or MonkeyOCR-pro-1.2B

# ModelScope
pip install modelscope
python tools/download_model.py -t modelscope -n MonkeyOCR  # or MonkeyOCR-pro-1.2B
```

### 3. Inference
Run the following commands to parse PDFs, images, or directories containing PDFs or images:

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
```bash
python demo/demo_gradio.py
```
### 5. Fast API
```bash
uvicorn api.main:app --port 8000
```

## Benchmarks

### Performance on OmniDocBench

| Model                  | Overall (EN) | Overall (ZH) |
| ---------------------- | ------------ | ------------ |
| MonkeyOCR-pro-3B       | **0.138**    | **0.206**    |
| MonkeyOCR-pro-1.2B     | 0.153        | 0.223        |

### Performance on olmOCR-bench

| Model                        | Overall |
| ---------------------------- | ------- |
| MonkeyOCR-pro-3B            | **75.8 ± 1.0** |
| MonkeyOCR-pro-1.2B          | 71.8 ± 1.1       |

##  Try the Demo
Experience MonkeyOCR's capabilities firsthand at  http://vlrlabmonkey.xyz:7685

## Learn More

*   [GitHub Repository](https://github.com/Yuliang-Liu/MonkeyOCR)
*   [ArXiv Paper](https://arxiv.org/abs/2506.05218)
*   [Hugging Face Model](https://huggingface.co/echo840/MonkeyOCR)
*   [ModelScope](https://modelscope.cn/models/l1731396519/MonkeyOCR)
*   [Demo](http://vlrlabmonkey.xyz:7685/)