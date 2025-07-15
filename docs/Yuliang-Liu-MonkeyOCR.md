# MonkeyOCR: Effortless Document Parsing with AI-Powered Structure Recognition

MonkeyOCR revolutionizes document processing with its innovative Structure-Recognition-Relation (SRR) triplet paradigm, enabling efficient and accurate parsing of diverse documents.  [Explore the original repository](https://github.com/Yuliang-Liu/MonkeyOCR)

**Key Features:**

*   **Superior Accuracy:** Outperforms state-of-the-art models on various benchmarks for both English and Chinese documents.
*   **Blazing Fast Performance:** Achieves impressive inference speeds, significantly faster than other methods on different GPUs.
*   **Versatile Document Support:**  Effectively parses a wide range of document types, including books, slides, financial reports, and more.
*   **Flexible Output:**  Generates markdown files, layout results, and detailed intermediate block results for comprehensive analysis.
*   **User-Friendly:** Provides a Gradio demo and FastAPI service for easy integration and use.

**Key improvements:**
*   MonkeyOCR-pro-1.2B surpasses MonkeyOCR-3B by 7.4% on Chinese documents.
*   MonkeyOCR-pro-1.2B delivers approximately a 36% speed improvement over MonkeyOCR-pro-3B, with approximately 1.6% drop in performance.
*   On olmOCR-Bench, MonkeyOCR-pro-1.2B outperforms Nanonets-OCR-3B by 7.3%.
*   On OmniDocBench, MonkeyOCR-pro-3B achieves the best overall performance on both English and Chinese documents, outperforming even closed-source and extra-large open-source VLMs such as Gemini 2.0-Flash, Gemini 2.5-Pro, Qwen2.5-VL-72B, GPT-4o, and InternVL3-78B.

## Key Performance Metrics & Benchmarks

*   **OmniDocBench:** MonkeyOCR-pro-3B demonstrates the best overall performance on English and Chinese documents compared to Pipeline Tools, Expert VLMs, General VLMs and Mix models.
*   **olmOCR-Bench:** MonkeyOCR-pro-3B leads the way with the best overall performance, demonstrating a high degree of accuracy across a range of document types.

## Quick Start

1.  **Install:** Follow the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda.md#install-with-cuda-support) to set up your environment.
2.  **Download Model Weights:** Get your preferred model from Hugging Face or ModelScope using `tools/download_model.py`.
3.  **Inference:** Use the `parse.py` script to process your documents:

    ```bash
    python parse.py input_path
    ```

    *   For more detailed usage, including examples for file/folder processing, single-task recognition, and advanced configurations, refer to the original README.

4.  **Gradio Demo:** Launch the interactive demo:

    ```bash
    python demo/demo_gradio.py
    ```
5.  **FastAPI:** Run the API service:

    ```bash
    uvicorn api.main:app --port 8000
    ```

## Docker Deployment

Follow the instructions in the original README to deploy MonkeyOCR using Docker.

## Further Information

*   **Windows Support:** Details available in the [Windows Support](docs/windows_support.md) guide.
*   **Quantization:** Quantize your model with AWQ, as detailed in the [Quantization guide](docs/Quantization.md).

## News
* ```2025.07.10 ``` 🚀 We release [MonkeyOCR-pro-1.2B](https://huggingface.co/echo840/MonkeyOCR-pro-1.2B), — a leaner and faster version model that outperforms our previous 3B version in accuracy, speed, and efficiency.
* ```2025.06.12 ``` 🚀 The model’s trending on [Hugging Face](https://huggingface.co/models?sort=trending). Thanks for the love!
* ```2025.06.05 ``` 🚀 We release [MonkeyOCR](https://huggingface.co/echo840/MonkeyOCR), an English and Chinese documents parsing model.

## Acknowledgements and Copyright

The acknowledgements and copyright information remain the same as in the original README.