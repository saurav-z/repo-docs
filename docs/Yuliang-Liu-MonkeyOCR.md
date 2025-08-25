# MonkeyOCR: Effortless Document Parsing with Structure-Recognition-Relation Triplet Paradigm

**Unlock insights from your documents with MonkeyOCR, an innovative OCR solution that leverages a Structure-Recognition-Relation (SRR) triplet paradigm for efficient and accurate document parsing.**

[View the Original Repository](https://github.com/Yuliang-Liu/MonkeyOCR)

---

## Key Features

*   **Superior Accuracy:** Achieve state-of-the-art results on diverse document types.
*   **Efficient Structure Recognition:** Simplifies complex document processing pipelines for faster results.
*   **Supports Diverse Documents:** Perfect for parsing PDFs, images, and various document layouts.
*   **Optimized for Speed:** Experience rapid processing with improved inference speed, especially on supported GPUs.
*   **Comprehensive Output:** Get markdown files, layout results, and intermediate block results.
*   **Model Flexibility:** Utilize MonkeyOCR-pro-1.2B, which improves speed and accuracy.

---

## What is MonkeyOCR?

MonkeyOCR adopts a Structure-Recognition-Relation (SRR) triplet paradigm, which simplifies the multi-tool pipeline of modular approaches while avoiding the inefficiency of using large multimodal models for full-page document processing.

---

## Key Advantages

*   **Performance Boost:** MonkeyOCR-pro-1.2B surpasses MonkeyOCR-3B by 7.4% on Chinese documents.
*   **Speed Improvement:** Enjoy approximately a 36% speed improvement over MonkeyOCR-pro-3B.
*   **Competitive Edge:** MonkeyOCR-pro-1.2B outperforms Nanonets-OCR-3B by 7.3% on olmOCR-Bench.
*   **Top-Tier Results:** MonkeyOCR-pro-3B achieves the best overall performance on OmniDocBench, surpassing closed-source and extra-large open-source VLMs like Gemini 2.0-Flash and GPT-4o.

---

## Key Performance Highlights

### Comparing MonkeyOCR with closed-source and extra large open-source VLMs.

[See comparison results](https://zimgs.com/i/EKhkhY)

### Inference Speed (Pages/s) on Different GPUs
*   [**See Benchmarking Tables**](https://github.com/Yuliang-Liu/MonkeyOCR#inference-speed-pagess-on-different-gpus-and-pdffile-page-counts)
*   [**See VLM OCR Speed Tables**](https://github.com/Yuliang-Liu/MonkeyOCR#vlm-ocr-speed-pagess-on-different-gpus-and-pdffile-page-counts)

---

## Getting Started

### 1. Install MonkeyOCR

Follow the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) to configure your environment.

### 2. Download Model Weights

*   From Hugging Face:

    ```bash
    pip install huggingface_hub
    python tools/download_model.py -n MonkeyOCR-pro-3B  # or MonkeyOCR
    ```
*   From ModelScope:

    ```bash
    pip install modelscope
    python tools/download_model.py -t modelscope -n MonkeyOCR-pro-3B  # or MonkeyOCR
    ```

### 3. Run Inference

```bash
python parse.py input_path
```

### 4. Run Gradio Demo

```bash
python demo/demo_gradio.py
```
View the demo at http://localhost:7860

### 5. Run FastAPI

```bash
uvicorn api.main:app --port 8000
```
View the API documentation at http://localhost:8000/docs

---

## Deployment Options

*   **Docker:** Simplify deployment with Docker Compose. See the [Docker guide](https://github.com/Yuliang-Liu/MonkeyOCR#docker-deployment).
*   **Windows Support:** Detailed instructions are available in the [windows support guide](docs/windows_support.md).
*   **Quantization:** Optimize model size and performance with AWQ quantization. See the [quantization guide](docs/Quantization.md).

---

## Further Information

*   **Benchmark Results:** Check the detailed [benchmark results](https://github.com/Yuliang-Liu/MonkeyOCR#benchmark-results) to see how MonkeyOCR performs across various datasets.
*   **Gradio Demo:** Experience MonkeyOCR in action with the [interactive demo](http://vlrlabmonkey.xyz:7685).

---

## Important Notes

*   **Citing MonkeyOCR:** If you use MonkeyOCR in your research, please cite the paper using the provided BibTeX entry.
*   **Limitations:** Currently, MonkeyOCR does not fully support photographed text, handwritten content, Traditional Chinese characters, or multilingual text.

---

**(The rest of the content regarding acknowledgements, limitations, copyright, and contact information can be kept or edited for brevity)**