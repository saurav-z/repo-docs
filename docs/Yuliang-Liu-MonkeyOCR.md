# MonkeyOCR: Unlock Document Intelligence with Advanced Structure Recognition

**Effortlessly extract structured data from documents using the power of MonkeyOCR, a state-of-the-art document parsing system.**

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR-pro-3B)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

> **Dive deeper into document parsing with our latest models!** Learn more about MonkeyOCR's innovative Structure-Recognition-Relation (SRR) triplet paradigm, simplifying the document processing pipeline.

## Key Features

*   **Advanced Parsing Paradigm:** Employs a Structure-Recognition-Relation (SRR) triplet approach for efficient and accurate document parsing.
*   **Superior Performance:** Outperforms leading closed-source and open-source models on diverse benchmarks, including OmniDocBench and olmOCR-Bench.
*   **Multiple Model Options:** Offers both MonkeyOCR-pro-1.2B and MonkeyOCR-pro-3B, providing flexibility in speed and performance.
*   **Comprehensive Task Support:** Excels in extracting text, formulas, and tables from various document types (books, slides, financial reports, etc.).
*   **Easy Deployment:**  Supports local installation, Gradio demo, and Docker deployment for flexible usage.
*   **Accelerated Speed**: MonkeyOCR-pro-1.2B delivers approximately a 36% speed improvement over MonkeyOCR-pro-3B.

## What is MonkeyOCR?

MonkeyOCR is a cutting-edge document parsing system designed to revolutionize how you process and extract information from documents. It leverages a unique Structure-Recognition-Relation (SRR) triplet paradigm.  This method simplifies the complex multi-tool pipeline, ensuring both efficiency and accuracy in handling full-page documents.

## Key Advantages

*   **Accuracy:** MonkeyOCR achieves state-of-the-art results across various document types.
*   **Efficiency:** The SRR paradigm streamlines the parsing process, resulting in faster processing times.
*   **Versatility:** Supports diverse document formats and element extraction (text, formulas, tables).
*   **User-Friendly:** Easy to install, deploy, and use with multiple deployment options.

## Performance Highlights

*   **OmniDocBench:** MonkeyOCR-pro-3B excels on both English and Chinese documents, surpassing even advanced models.
*   **olmOCR-Bench:** MonkeyOCR-pro-1.2B outperforms Nanonets-OCR-3B by 7.3%.
*   **Speed Improvements:** MonkeyOCR-pro-1.2B delivers a 36% speed improvement over MonkeyOCR-pro-3B.

[View the detailed benchmark results in the original README](https://github.com/Yuliang-Liu/MonkeyOCR#benchmark-results).

## Quick Start

Get up and running with MonkeyOCR in a few simple steps:

1.  **Installation:** Follow the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) to set up your environment.
2.  **Model Download:** Download the pre-trained models from Hugging Face or ModelScope:

```bash
pip install huggingface_hub
python tools/download_model.py -n MonkeyOCR-pro-3B  # or MonkeyOCR
```

3.  **Inference:** Use the `parse.py` script to process your documents:

```bash
python parse.py input_path
```

   Refer to the original README for more [detailed usage examples](https://github.com/Yuliang-Liu/MonkeyOCR#quick-start).

## Deployment Options

*   **Local Installation:** Detailed instructions are available in the Quick Start section.
*   **Gradio Demo:** Launch the interactive demo using `python demo/demo_gradio.py`.
*   **FastAPI Service:** Deploy the API using `uvicorn api.main:app --port 8000`.
*   **Docker:** Build and run the Docker image for easy deployment. See the [Docker Deployment](https://github.com/Yuliang-Liu/MonkeyOCR#docker-deployment) section.
*   **Windows Support:**  Windows support details can be found in the [windows support guide](docs/windows_support.md).

## Additional Information

*   **Supported Hardware:** The model has been tested on various GPUs (3090, 4090, A6000, H800, etc.).
*   **Quantization:** The model can be quantized using AWQ. Follow the [quantization guide](docs/Quantization.md).
*   **News:** Stay up-to-date with the latest releases and updates.
    *   ```2025.07.10 ``` 🚀 We release [MonkeyOCR-pro-1.2B](https://huggingface.co/echo840/MonkeyOCR-pro-1.2B).
    *   ```2025.06.12 ``` 🚀 The model’s trending on [Hugging Face](https://huggingface.co/models?sort=trending).
    *   ```2025.06.05 ``` 🚀 We release [MonkeyOCR](https://huggingface.co/echo840/MonkeyOCR).

## Get Started Today!

Explore the possibilities of intelligent document processing with MonkeyOCR.  Visit the [GitHub repository](https://github.com/Yuliang-Liu/MonkeyOCR) to get started and unlock the potential of your documents.

## Citing MonkeyOCR

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

Thanks to the contributors listed in the original README.

## Limitations

See the original README for information on limitations, including support for photographed text, handwritten content, Traditional Chinese characters, or multilingual text.

## Copyright

See the original README for information on copyright and usage.  For commercial inquiries, please contact xbai@hust.edu.cn or ylliu@hust.edu.cn.