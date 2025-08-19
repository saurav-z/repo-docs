# MonkeyOCR: Unlock Document Insights with AI 🚀

**MonkeyOCR is a cutting-edge document parsing system leveraging a Structure-Recognition-Relation (SRR) triplet paradigm to efficiently extract and understand information from documents.**  ([Original Repo](https://github.com/Yuliang-Liu/MonkeyOCR))

## Key Features:

*   **Advanced Document Understanding:** Accurately parses and extracts text, tables, formulas, and layout information from diverse document types.
*   **SRR Paradigm:** Employs a novel Structure-Recognition-Relation triplet paradigm for efficient and effective document processing.
*   **Superior Performance:** Outperforms leading open-source and closed-source models, including Gemini and GPT-4o, on benchmarks like OmniDocBench and olmOCR-Bench.
*   **Optimized for Speed and Efficiency:** Offers faster inference speeds and improved performance, especially with MonkeyOCR-pro-1.2B.
*   **Easy to Use:** Provides a simple installation process, with support for local setup, Hugging Face, ModelScope, and Docker deployment.
*   **Gradio Demo & Fast API:**  A user-friendly Gradio demo is available for easy testing and a FastAPI service for integration into applications.
*   **Comprehensive Hardware Support:** Works on a variety of GPUs, including 3090, 4090, A6000, H800, and even the 4060 (quantized model).

## Performance Highlights:

*   **MonkeyOCR-pro-1.2B:** Surpasses MonkeyOCR-3B by 7.4% on Chinese documents, providing approximately a 36% speed boost (with only a 1.6% performance drop)
*   **OmniDocBench Dominance:**  MonkeyOCR-pro-3B achieves the best overall performance across both English and Chinese documents.
*   **olmOCR-Bench Leader:** MonkeyOCR-pro-1.2B outperforms Nanonets-OCR-3B by 7.3%.

### Benchmark Results:

*   **OmniDocBench:**  MonkeyOCR family outperforms competitors in Overall, Text, Formula, Table (TEDS & Edit), and Read Order metrics.  See tables in original README for specific comparisons.
*   **olmOCR-Bench:** MonkeyOCR-pro-3B and MonkeyOCR-pro-1.2B show strong performance, exceeding scores of models like Gemini and GPT-4o.

## Getting Started:

### 1. Local Installation
*   **Install Dependencies:**
    ```bash
    # See the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) to set up your environment.
    ```
*   **Download Model Weights:**
    ```python
    pip install huggingface_hub
    python tools/download_model.py -n MonkeyOCR  # or MonkeyOCR-pro-1.2B
    ```
    or from ModelScope:
    ```python
    pip install modelscope
    python tools/download_model.py -t modelscope -n MonkeyOCR  # or MonkeyOCR-pro-1.2B
    ```
*   **Inference:**
    ```bash
    python parse.py input_path  # Basic parsing
    python parse.py input_path -t text/formula/table # Single task recognition
    python parse.py input_path -s # Page splitting
    python parse.py input_path -o ./output -c config.yaml # Specify output & config
    ```
    See original README for details and more options.

### 2. Gradio Demo
*   Run the demo locally:
    ```bash
    python demo/demo_gradio.py
    ```
    Access it at `http://localhost:7860`.

### 3. FastAPI
*   Start the API service:
    ```bash
    uvicorn api.main:app --port 8000
    ```
    Explore the API at `http://localhost:8000/docs`.

### 4. Docker Deployment
*   Build and run with NVIDIA GPU support:
    ```bash
    # Navigate to the docker directory
    cd docker
    # Build the Docker image (monkeyocr-fix if GPU is 20/30/40-series)
    docker compose build monkeyocr # or docker compose build monkeyocr-fix
    # Run the Gradio demo
    docker compose up monkeyocr-demo
    # Run the FastAPI service
    docker compose up monkeyocr-api
    ```
    Access API documentation at `http://localhost:7861/docs`.

### 5. Windows Support

*   See the [windows support guide](docs/windows_support.md) for details.

## Quantization

*   This model can be quantized using AWQ. Follow the instructions in the [quantization guide](docs/Quantization.md).

## Demo

*   Try the interactive demo:  http://vlrlabmonkey.xyz:7685

##  Citing MonkeyOCR
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

##  Acknowledgments

(List of acknowledgments from original README)

## Limitations

(List of limitations from original README)

##  Copyright
(Copyright information from original README)