<div align="center">

<p align="center">
    <img src="https://raw.githubusercontent.com/rednote-hilab/dots.ocr/master/assets/logo.png" width="300" alt="dots.ocr Logo"/>
<p>

<h1 align="center">
dots.ocr: Revolutionizing Document Parsing with a Single Vision-Language Model
</h1>

[![Blog](https://img.shields.io/badge/Blog-View_on_GitHub-333.svg?logo=github)](https://github.com/rednote-hilab/dots.ocr/blob/master/assets/blog.md)
[![HuggingFace](https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace)](https://huggingface.co/rednote-hilab/dots.ocr)

<div align="center">
  <a href="https://dotsocr.xiaohongshu.com" target="_blank" rel="noopener noreferrer"><strong>🖥️ Live Demo</strong></a> |
  <a href="https://raw.githubusercontent.com/rednote-hilab/dots.ocr/master/assets/wechat.jpg" target="_blank" rel="noopener noreferrer"><strong>💬 WeChat</strong></a> |
  <a href="https://www.xiaohongshu.com/user/profile/683ffe42000000001d021a4c" target="_blank" rel="noopener noreferrer"><strong>📕 rednote</strong></a>
</div>

</div>

**Dots.ocr is a cutting-edge, multilingual document parsing solution, unifying layout detection and content recognition in a single, efficient vision-language model.**  Explore the original repository: [dots.ocr GitHub](https://github.com/rednote-hilab/dots.ocr)

## Key Features

*   **State-of-the-Art Performance:** Achieves SOTA results on OmniDocBench for text, tables, and reading order, with formula recognition comparable to larger models.
*   **Multilingual Support:** Robust parsing capabilities for low-resource languages, excelling in both layout detection and content recognition on a multilingual benchmark.
*   **Unified Architecture:** Simplifies document parsing with a single vision-language model, eliminating complex, multi-model pipelines.
*   **Efficient and Fast:** Built on a compact 1.7B LLM, offering faster inference speeds compared to many high-performing models.
*   **Comprehensive Parsing:** Processes images and PDFs, extracting layout information, text, formulas (LaTeX format), and tables (HTML format). Includes tools for reading order and text extraction.
*   **Flexible Deployment:** Supports deployment via vLLM (recommended for speed) and Hugging Face, along with a Docker image for easy setup.

## Performance Highlights

### Benchmark Results

**Dots.ocr consistently outperforms competing models across various benchmarks.**  (See the original README for detailed comparison tables and metrics on OmniDocBench, dots.ocr-bench, and olmOCR-bench.)

### Performance Comparison: dots.ocr vs. Competing Models
<img src="https://raw.githubusercontent.com/rednote-hilab/dots.ocr/master/assets/chart.png" border="0" alt="dots.ocr Performance Chart"/>

## Quick Start

### 1. Installation

```bash
# Create conda environment (optional)
conda create -n dots_ocr python=3.12
conda activate dots_ocr

# Clone the repository
git clone https://github.com/rednote-hilab/dots.ocr.git
cd dots.ocr

# Install pytorch (adjust for your CUDA version - see https://pytorch.org/get-started/previous-versions/)
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

**Alternatively, use the Docker Image:**

```bash
docker pull rednotehilab/dots.ocr
#  (and skip the conda/pip install steps below and move to the '2. Deployment' vLLM section.)

# Clone the repository
git clone https://github.com/rednote-hilab/dots.ocr.git
cd dots.ocr
pip install -e .
```

### 2. Download Model Weights

```bash
python3 tools/download_model.py

# with modelscope
python3 tools/download_model.py --type modelscope
```

### 3. Deployment

**vLLM Inference (Recommended):**

*   **Register model to vLLM:**  (from '2. Download Model Weights')
*   **Set environment variables:**  `export hf_model_path=./weights/DotsOCR` (or your model directory). Remember to use a directory name without periods (e.g., `DotsOCR` instead of `dots.ocr`) for the model save path.
*   **Modify `vllm`:**  (See instructions in original README for modifying `vllm` to include `DotsOCR`.)
*   **Launch vLLM server:**  `CUDA_VISIBLE_DEVICES=0 vllm serve ${hf_model_path} --tensor-parallel-size 1 --gpu-memory-utilization 0.95  --chat-template-content-format string --served-model-name model --trust-remote-code`
*   **Run vLLM demo:** `python3 ./demo/demo_vllm.py --prompt_mode prompt_layout_all_en`

**Hugging Face Inference:**
```bash
python3 demo/demo_hf.py
```

### 4. Document Parsing

**Using vLLM Server:**

```bash
# Parse a single image (all info)
python3 dots_ocr/parser.py demo/demo_image1.jpg
# Parse a single PDF (all info)
python3 dots_ocr/parser.py demo/demo_pdf1.pdf --num_thread 64  # adjust thread count
# Layout detection only
python3 dots_ocr/parser.py demo/demo_image1.jpg --prompt prompt_layout_only_en
# Text extraction only (excluding headers/footers)
python3 dots_ocr/parser.py demo/demo_image1.jpg --prompt prompt_ocr
# Parse by bbox
python3 dots_ocr/parser.py demo/demo_image1.jpg --prompt prompt_grounding_ocr --bbox 163 241 1536 705
```

**Using Transformers:**
Run the same commands as above, but add `--use_hf true` to the `parser.py` command.

### Output Results

*   JSON file with structured layout data.
*   Markdown file generated from extracted text.
*   Image with layout bounding boxes visualized.

### Examples

*   **Formula Document:** [formula1.png, formula2.png, formula3.png](https://raw.githubusercontent.com/rednote-hilab/dots.ocr/master/assets/showcase/formula1.png)
*   **Table Document:** [table1.png, table2.png, table3.png](https://raw.githubusercontent.com/rednote-hilab/dots.ocr/master/assets/showcase/table1.png)
*   **Multilingual Document:** [Tibetan.png, tradition_zh.png, nl.png, kannada.png, russian.png](https://raw.githubusercontent.com/rednote-hilab/dots.ocr/master/assets/showcase/Tibetan.png)
*   **Reading Order:** [reading_order.png](https://raw.githubusercontent.com/rednote-hilab/dots.ocr/master/assets/showcase/reading_order.png)
*   **Grounding OCR:** [grounding.png](https://raw.githubusercontent.com/rednote-hilab/dots.ocr/master/assets/showcase/grounding.png)

## Acknowledgments

(See original README for a complete list of acknowledgements.)

## Limitations & Future Work

*   **Areas for Improvement:** Complex tables and formulas, picture parsing.
*   **Potential Issues:** High character-to-pixel ratios, continuous special characters.
*   **Future Goals:**  Enhanced accuracy, broader OCR capabilities, a more general-purpose VLM for document understanding, and parsing of pictures.

If you are interested in advancing the field of document intelligence, please contact us at [yanqing4@xiaohongshu.com](mailto:yanqing4@xiaohongshu.com).
```
Key improvements and summaries:

*   **SEO Optimization:**  Added "Document Parsing," "OCR," "Multilingual," and other relevant keywords to headings and descriptions.
*   **Concise Hook:** Created a compelling one-sentence introduction.
*   **Clear Headings and Structure:** Organized content for easy readability.
*   **Bulleted Key Features:** Highlighted core functionalities effectively.
*   **Removed Redundancy:** Streamlined the introduction and performance comparisons.
*   **Simplified Installation:** Condenses steps.
*   **Added Docker:** Highlights docker as an easier installation option.
*   **Focus on Value:** Emphasized the benefits of using dots.ocr.
*   **Concise Steps and Examples:**  Provided a clear path to get started.
*   **Direct Links:**  Kept essential links to the demo and relevant resources.
*   **Contact Information:** Included a call to action for potential contributors.
*   **Added "Alt" text to images:**  For accessibility and SEO.
*   **Cleaned up the output examples.**