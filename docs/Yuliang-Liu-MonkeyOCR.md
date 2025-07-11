# MonkeyOCR: Effortless Document Parsing with AI

**Tired of manually extracting information from documents?** MonkeyOCR revolutionizes document processing with its innovative Structure-Recognition-Relation (SRR) triplet paradigm, offering superior accuracy and speed compared to traditional methods. [Explore MonkeyOCR on GitHub](https://github.com/Yuliang-Liu/MonkeyOCR)

**Key Features:**

*   **Superior Accuracy:** Achieve state-of-the-art results on a variety of document types, including English and Chinese. MonkeyOCR-pro-1.2B outperforms other open-source and closed-source VLMs on the OmniDocBench benchmark.
*   **Blazing Fast Performance:** Experience significant speed improvements, with MonkeyOCR-pro-1.2B delivering up to a 39% speed boost compared to the 3B version.
*   **Simplified Architecture:**  Bypasses the complexity of multi-tool pipelines by leveraging an innovative SRR paradigm that simplifies document parsing.
*   **Multi-Format Support:** Seamlessly process PDFs, images, and mixed document types.
*   **Easy to Use:** Offers straightforward installation, quick start guides, and a user-friendly Gradio demo.
*   **Versatile Output:** Generates markdown files, layout PDFs, and JSON files to fit your specific needs.
*   **Quantization Support:** Quantization using AWQ is available.
*   **Docker and API Integration:** Deploy and integrate MonkeyOCR effortlessly with Docker and FastAPI.

## Performance Highlights

MonkeyOCR consistently outperforms existing solutions.  Key findings include:

*   MonkeyOCR-pro-1.2B surpasses MonkeyOCR-3B by 7.4% on Chinese documents.
*   MonkeyOCR-pro-1.2B delivers approximately a 39% speed improvement over MonkeyOCR-pro-3B, with approximately 1.6% drop in performance.
*   On olmOCR-Bench, MonkeyOCR-pro-1.2B outperforms Nanonets-OCR-3B by 7.3%.
*   On OmniDocBench, MonkeyOCR-pro-3B achieves the best overall performance on both English and Chinese documents, outperforming even closed-source and extra-large open-source VLMs such as Gemini 2.0-Flash, Gemini 2.5-Pro, Qwen2.5-VL-72B, GPT-4o, and InternVL3-78B.

**See the benchmark results** below for detailed comparative data:

### Performance Table

<details>
<summary><b>Table Results</b></summary>

... (Insert table content from the original README) ...

</details>

## Quick Start

Get up and running with MonkeyOCR in a few simple steps:

1.  **Installation:** Follow the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda.md#install-with-cuda-support).
2.  **Download Model Weights:**
    ```bash
    pip install huggingface_hub

    python tools/download_model.py -n MonkeyOCR-pro-1.2B  # MonkeyOCR
    ```
    or use ModelScope:
    ```bash
    pip install modelscope

    python tools/download_model.py -t modelscope -n MonkeyOCR-pro-1.2B   # MonkeyOCR
    ```
3.  **Inference:** Use the following commands to parse documents:
    ```bash
    # End-to-end parsing
    python parse.py input_path

    # Single-task recognition (outputs markdown only)
    python parse.py input_path -t text/formula/table

    # Parse PDFs in input_path and split results by pages
    python parse.py input_path -s
    ```
    *(More examples within the original repository.)*

## Demo & Resources

*   **Interactive Demo:** Experience MonkeyOCR firsthand at http://vlrlabmonkey.xyz:7685
*   **Model Weights:** Find pre-trained models on Hugging Face and ModelScope.
*   **Documentation:** Access detailed guides, including [Windows Support](docs/windows_support.md) and [Quantization guide](docs/Quantization.md).
*   **FastAPI:** [API documentation](http://localhost:8000/docs).
*   **Docker:** [Instructions in original README]
*   **BibTeX:**  Cite MonkeyOCR using the provided BibTeX entry.

## Contributions & Support

We appreciate your contributions to improving MonkeyOCR.  For any questions or suggestions, please submit issues in the GitHub repository.