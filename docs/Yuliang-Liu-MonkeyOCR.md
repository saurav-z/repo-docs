# MonkeyOCR: Unleashing Advanced Document Parsing with AI

**Effortlessly extract information from documents with MonkeyOCR, leveraging a cutting-edge Structure-Recognition-Relation (SRR) triplet paradigm for superior accuracy and speed. ([Original Repo](https://github.com/Yuliang-Liu/MonkeyOCR))**

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR-pro-3B)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

MonkeyOCR empowers you to seamlessly parse complex documents, excelling in accuracy and speed across both English and Chinese datasets.

## Key Features:

*   **Superior Accuracy:** MonkeyOCR-pro-1.2B outperforms other models across English and Chinese documents.
*   **Blazing Speed:** Achieve up to 36% faster processing compared to MonkeyOCR-pro-3B.
*   **Exceptional Performance:** Top performance on OmniDocBench, even surpassing closed-source and large open-source VLMs such as Gemini 2.0-Flash and GPT-4o.
*   **Structure-Recognition-Relation (SRR) Paradigm:** Simplifies the document processing pipeline, leading to improved efficiency and performance.
*   **Gradio Demo and FastAPI Support:** Easy-to-use demo and API integration for quick testing and deployment.

## What's New

*   **[2025.07.10]**: Released [MonkeyOCR-pro-1.2B](https://huggingface.co/echo840/MonkeyOCR-pro-1.2B) - a faster and leaner version.
*   **[2025.06.12]**: Trending on [Hugging Face](https://huggingface.co/models?sort=trending).
*   **[2025.06.05]**: Released [MonkeyOCR](https://huggingface.co/echo840/MonkeyOCR), a document parsing model for English and Chinese documents.

## Performance Highlights

**MonkeyOCR excels in speed and accuracy:**

### Performance Comparison

*   **OmniDocBench:** MonkeyOCR achieves best overall performance.
*   **olmOCR-Bench:** MonkeyOCR-pro-1.2B outperforms Nanonets-OCR-3B by 7.3%.

### Inference Speed (Pages/s) on Different GPUs

See detailed inference speed results for various models and GPUs in the original README tables.  Links to the performance tables and the image are included in the original README.

## Quick Start

### Installation

1.  Install MonkeyOCR following the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support).
2.  Download Model Weights from Hugging Face or ModelScope.

### Inference

Use the `parse.py` script to process documents:

```bash
# End-to-end parsing
python parse.py input_path
```

Additional commands and usage examples are detailed in the original README.

*   **Gradio Demo:**  Run `python demo/demo_gradio.py` for interactive testing.
*   **FastAPI:**  Deploy via `uvicorn api.main:app --port 8000`.

### Docker Deployment

Detailed steps for Docker deployment are available in the original README.

### Quantization

This model can be quantized using AWQ, follow the instructions in the [quantization guide](docs/Quantization.md).

## Documentation & Resources

*   **Detailed Results:** Explore the comprehensive benchmark results in the original README.
*   **Visualization Demo:** Experience MonkeyOCR firsthand: [http://vlrlabmonkey.xyz:7685](http://vlrlabmonkey.xyz:7685)
*   **Windows Support Guide:** Find specific instructions for Windows users in the [windows support guide](docs/windows_support.md).
*   **Quantization Guide:** Further optimize the model with quantization, find instructions in the [quantization guide](docs/Quantization.md).
*   **API Documentation:** Review the available endpoints at http://localhost:8000/docs

## Citations

If you are using MonkeyOCR in your research, please cite the following:

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

We would like to thank the individuals and projects listed in the original README.

## Limitations

Please review the limitations described in the original README.

## Copyright

Please note that MonkeyOCR is intended for academic research and non-commercial use only.