# MonkeyOCR: Revolutionizing Document Parsing with SRR Triplet Paradigm ([Original Repo](https://github.com/Yuliang-Liu/MonkeyOCR))

MonkeyOCR is a cutting-edge document parsing solution that leverages a Structure-Recognition-Relation (SRR) triplet paradigm to accurately and efficiently extract information from complex documents.

**Key Features:**

*   **SRR Paradigm:** Simplifies document parsing, avoiding complex multi-tool pipelines.
*   **Superior Performance:** Outperforms leading closed-source and open-source models like Gemini 2.5-Pro, Qwen2.5-VL-72B, and GPT-4o on OmniDocBench.
*   **High Accuracy:** Delivers state-of-the-art results in text, formula, and table extraction.
*   **Fast Inference:** Optimized for speed with impressive pages-per-second processing on various GPUs.
*   **Open-Source:** Built upon open-source tools and datasets, fostering community contribution and improvement.
*   **Gradio Demo & FastAPI:** Easy-to-use demo for quick evaluation and FastAPI service for seamless integration.
*   **Model Variants:** Offers multiple models, including the highly efficient MonkeyOCR-pro-1.2B, for diverse needs.
*   **Windows & Docker Support:** Compatible with Windows and includes Docker images for straightforward deployment.

**What's New:**

*   🚀 **MonkeyOCR-pro-1.2B Released:** A leaner and faster model version, providing improved accuracy, speed, and efficiency.

## Performance Highlights

MonkeyOCR achieves impressive results across various benchmarks and tasks:

### End-to-End Performance on OmniDocBench

MonkeyOCR's end-to-end performance surpasses other VLMs and pipeline tools.

### End-to-End Text Recognition Performance

MonkeyOCR-pro-3B and MonkeyOCR-pro-1.2B achieve state-of-the-art text recognition results across 9 PDF page types.

### OlmOCR-Bench Evaluation

MonkeyOCR-pro-3B demonstrates top-tier performance on the rigorous OlmOCR-Bench, outperforming many expert models.

## Getting Started

### 1. Installation
Follow the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) to set up your environment.

### 2. Model Download
Download the pre-trained model weights.

```bash
pip install huggingface_hub
python tools/download_model.py -n MonkeyOCR  # or MonkeyOCR-pro-1.2B
```

### 3. Inference
Use the following commands to parse documents:

```bash
python parse.py input_path  # Parse a single PDF or image
python parse.py input_path -t text/formula/table # Perform Single Task
```

### 4. Gradio Demo
Launch the interactive demo:

```bash
python demo/demo_gradio.py
```

Access the demo at http://localhost:7860.

### 5. FastAPI Service
Start the FastAPI service for API integration:

```bash
uvicorn api.main:app --port 8000
```

Access the API documentation at http://localhost:8000/docs.

## Further Resources

*   **[Demo](http://vlrlabmonkey.xyz:7685/)**: Experience MonkeyOCR's capabilities firsthand.
*   **[Hugging Face](https://huggingface.co/echo840/MonkeyOCR)**: Access model weights and resources.
*   **[Windows Support Guide](docs/windows_support.md)**: Instructions for Windows users.
*   **[Quantization Guide](docs/Quantization.md)**: Optimize model performance through quantization.

## Citation

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

Special thanks to all contributors and the open-source community.

## Limitations & Copyright

MonkeyOCR is for academic/non-commercial use. For commercial options, contact the authors.