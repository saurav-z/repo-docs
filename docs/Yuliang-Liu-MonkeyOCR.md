# MonkeyOCR: Revolutionizing Document Parsing with a New Triplet Paradigm

**Unleash the power of MonkeyOCR, a cutting-edge document parsing system that leverages a Structure-Recognition-Relation (SRR) triplet paradigm to efficiently and accurately process documents.**  [Explore the Original Repo](https://github.com/Yuliang-Liu/MonkeyOCR)

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

MonkeyOCR offers a streamlined and efficient approach to document processing, excelling in both English and Chinese documents.

**Key Features:**

*   **SRR Paradigm:** Simplifies document parsing, avoiding the complexity of multi-tool pipelines.
*   **Superior Performance:** MonkeyOCR-pro-1.2B surpasses MonkeyOCR-3B on Chinese documents and outperforms other methods like Nanonets-OCR-3B.
*   **Speed & Efficiency:**  Achieves significant speed improvements over previous versions while maintaining high accuracy.
*   **Strong Results:** Outperforms closed-source and large open-source VLMs on the OmniDocBench, even including Gemini 2.0-Flash, Gemini 2.5-Pro, Qwen2.5-VL-72B, GPT-4o, and InternVL3-78B.
*   **Gradio Demo & FastAPI:**  Easily test & integrate via a user-friendly Gradio demo and a FastAPI for API access.
*   **Hardware Support:**  Optimized for various GPU configurations (3090, 4090, A6000, H800, and more).
*   **Quantization:** Model supports AWQ quantization.
*   **Flexible Output:** Produces Markdown files, layout results, and detailed intermediate block results for comprehensive document analysis.

**Performance Highlights:**

*   **MonkeyOCR-pro-1.2B vs MonkeyOCR-3B:** Up to 7.4% better on Chinese documents with approximately a 36% speed improvement, with approximately a 1.6% drop in performance.
*   **OmniDocBench:** MonkeyOCR-pro-3B delivers top-tier performance on both English and Chinese documents.
*   **olmOCR-Bench:** MonkeyOCR-pro-1.2B outperforms Nanonets-OCR-3B by 7.3%.

### Quick Start

**1.  Install**

See the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) to set up your environment.

**2. Download Model Weights**

```python
pip install huggingface_hub
python tools/download_model.py -n MonkeyOCR  # or MonkeyOCR-pro-1.2B
```

**3. Inference**

```bash
python parse.py input_path
```

**4.  Run Gradio Demo:**

```bash
python demo/demo_gradio.py
```
Access the demo at http://localhost:7860

**5. Run Fast API**
```bash
uvicorn api.main:app --port 8000
```
Access the API documentation at http://localhost:8000/docs

### Docker Deployment

Follow the instructions for Docker deployment.

### Windows Support

See the [windows support guide](docs/windows_support.md) for details.

### Benchmarks and Results

[Detailed benchmark results can be found in the original README, including performance comparisons on OmniDocBench and olmOCR-Bench, as well as text recognition across multiple document types, hardware configurations, and GPU inference speeds.]

**[See the original README for comprehensive benchmark results and performance comparisons.]**

###  Get Started Now!

Explore the capabilities of MonkeyOCR and experience the future of document parsing.  [Get started with the Quick Start guide and unleash the power of MonkeyOCR!](https://github.com/Yuliang-Liu/MonkeyOCR)

---