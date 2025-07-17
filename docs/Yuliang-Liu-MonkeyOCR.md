# MonkeyOCR: Revolutionizing Document Parsing with SRR Triplet Paradigm

**Tired of cumbersome multi-tool pipelines for document processing?** MonkeyOCR offers a streamlined solution, leveraging a Structure-Recognition-Relation (SRR) triplet paradigm to efficiently parse documents with impressive accuracy and speed. [Explore the original repository](https://github.com/Yuliang-Liu/MonkeyOCR)!

*   **SRR Paradigm:** Simplifies document parsing by combining structure recognition, content recognition, and relationship prediction.
*   **Superior Performance:** Outperforms leading closed-source and open-source VLMs on various benchmarks, including OmniDocBench and olmOCR-Bench.
*   **Multiple Model Options:** Offers both MonkeyOCR-pro-3B and a faster, more efficient MonkeyOCR-pro-1.2B.
*   **Flexible Deployment:** Supports local installation, Hugging Face integration, ModelScope, Gradio demo, and Docker deployment for easy integration.
*   **Comprehensive Outputs:** Generates Markdown, layout results, and intermediate JSON files for detailed analysis and customization.

## Key Features

*   **Advanced Document Parsing:**  Effectively processes complex documents, including those with formulas, tables, and diverse layouts.
*   **Benchmark Dominance:** Achieves state-of-the-art results on standard document understanding benchmarks.
*   **Speed & Efficiency:** Offers faster processing speeds compared to existing solutions, optimizing for efficient performance on various GPUs.
*   **User-Friendly Interface:**  Includes a Gradio demo and a fast API for easy experimentation and integration.
*   **Extensive Customization:**  Provides options for single-task recognition, page splitting, and output directory specification.

## Quick Start

### 1. Install MonkeyOCR

*   Follow the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) for setting up your environment with CUDA support.

### 2. Download Model Weights

*   **From Hugging Face:**
    ```bash
    pip install huggingface_hub
    python tools/download_model.py -n MonkeyOCR  # or MonkeyOCR-pro-1.2B
    ```
*   **From ModelScope:**
    ```bash
    pip install modelscope
    python tools/download_model.py -t modelscope -n MonkeyOCR  # or MonkeyOCR-pro-1.2B
    ```

### 3. Inference

*   **Basic Usage:**
    ```bash
    python parse.py input_path
    ```
    Replace `input_path` with the path to your PDF, image, or directory.

*   **Additional Options:**  See detailed usage examples in the original README.

### 4. Gradio Demo

*   Run the interactive demo:
    ```bash
    python demo/demo_gradio.py
    ```
    Access the demo at http://localhost:7860.

### 5. Fast API

*   Start the API service:
    ```bash
    uvicorn api.main:app --port 8000
    ```
    Access the API documentation at http://localhost:8000/docs.

## Deployment Options

*   **Docker:** Detailed instructions are provided in the original README and the `docker` directory.  Supports NVIDIA GPU.
*   **Windows Support:**  Find guidance in the `docs/windows_support.md`.
*   **Quantization:**  Explore model quantization using AWQ in the `docs/Quantization.md`.

## Benchmark Results

*   Comprehensive evaluation results are provided in the original README, demonstrating MonkeyOCR's performance on various benchmarks. Key results include those achieved on OmniDocBench and olmOCR-bench.

## Visualization Demo

*   Experience the power of MonkeyOCR via the easy-to-use demo: http://vlrlabmonkey.xyz:7685

## How to Cite

If you use MonkeyOCR, please cite the following:

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

*   (as in original README)

## Limitations

*   (as in original README)

## Copyright

*   (as in original README)