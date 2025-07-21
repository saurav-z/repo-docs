# MonkeyOCR: Advanced Document Parsing with SRR Triplet Paradigm

**Unleash the power of MonkeyOCR to effortlessly parse complex documents using a novel Structure-Recognition-Relation (SRR) triplet paradigm.** ([Original Repo](https://github.com/Yuliang-Liu/MonkeyOCR))

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)


## Key Features

*   **Simplified Pipeline:** Leverages the innovative SRR triplet paradigm to streamline document parsing, eliminating the need for complex, multi-tool pipelines.
*   **Superior Performance:** MonkeyOCR-pro-1.2B outperforms MonkeyOCR-3B by 7.4% on Chinese documents and delivers a 36% speed improvement.
*   **Competitive Benchmarking:**  MonkeyOCR-pro-1.2B excels on olmOCR-Bench, surpassing Nanonets-OCR-3B by 7.3%.
*   **State-of-the-Art Results:** MonkeyOCR-pro-3B achieves top-tier performance on OmniDocBench, outclassing even closed-source and large open-source VLMs.
*   **Fast Inference:** Experience high-speed document processing, optimizing efficiency and productivity with detailed benchmarking.
*   **Easy Deployment:**  Deploy with ease using the provided Quick Start guide, including local installation, Gradio demo, Docker deployment and Fast API.
*   **Model Quantization:**  Optimize model size and performance with AWQ quantization support.
*   **Comprehensive Output:** Provides Markdown output, layout results, and intermediate block results for detailed analysis.

## Benchmarking Performance Highlights

MonkeyOCR demonstrates exceptional performance across multiple benchmarks:

*   **OmniDocBench:** MonkeyOCR-pro-3B achieves the best overall performance, outperforming even closed-source and extra-large open-source VLMs.
*   **olmOCR-Bench:** MonkeyOCR-pro-1.2B outperforms Nanonets-OCR-3B by 7.3%.

### Comparing MonkeyOCR with closed-source and extra large open-source VLMs.
<a href="https://zimgs.com/i/EKhkhY"><img src="https://v1.ax1x.com/2025/07/15/EKhkhY.png" alt="EKhkhY.png" border="0" /></a>

### Inference Speed (Pages/s) on Different GPUs

See the original README for detailed tables with performance metrics.  
**[PDF](https://drive.google.com/drive/folders/1geumlJmVY7UUKdr8324sYZ0FHSAElh7m?usp=sharing)** Page Counts

## Quick Start Guide

1.  **Installation:** Follow the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) to set up your environment.
2.  **Model Download:** Download model weights from Hugging Face or ModelScope using the provided `download_model.py` script.
3.  **Inference:** Run the `parse.py` script with your input files or directories, including options for single-task recognition and output customization.
4.  **Gradio Demo:** Launch the user-friendly Gradio demo with `python demo/demo_gradio.py`.
5.  **FastAPI Service:** Deploy the FastAPI service using `uvicorn api.main:app --port 8000`.
6.  **Docker Deployment:** Simplify deployment with Docker, including NVIDIA GPU support using `docker compose`.

## Docker Deployment

Follow the instructions in the [docker section](https://github.com/Yuliang-Liu/MonkeyOCR#docker-deployment) to build and run the Docker image.

## Windows Support

See the [windows support guide](docs/windows_support.md) for details.

## Quantization

This model can be quantized using AWQ. Follow the instructions in the [quantization guide](docs/Quantization.md).

## Benchmark Results

See the original README for detailed tables with performance metrics.  

## Visualization Demo

Get a Quick Hands-On Experience with Our Demo:  http://vlrlabmonkey.xyz:7685 (The latest model is available for selection)

> Our demo is simple and easy to use:
>
> 1. Upload a PDF or image.
> 2. Click “Parse (解析)” to let the model perform structure detection, content recognition, and relationship prediction on the input document. The final output will be a markdown-formatted version of the document.
> 3. Select a prompt and click “Test by prompt” to let the model perform content recognition on the image based on the selected prompt.

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

We would like to thank [MinerU](https://github.com/opendatalab/MinerU), [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO), [PyMuPDF](https://github.com/pymupdf/PyMuPDF), [layoutreader](https://github.com/ppaanngggg/layoutreader), [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [LMDeploy](https://github.com/InternLM/lmdeploy), [PP-StructureV3](https://github.com/PaddlePaddle/PaddleOCR), [PP-DocLayout_plus-L](https://huggingface.co/PaddlePaddle/PP-DocLayout_plus-L), and [InternVL3](https://github.com/OpenGVLab/InternVL) for providing base code and models, as well as their contributions to this field. We also thank [M6Doc](https://github.com/HCIILAB/M6Doc), [DocLayNet](https://github.com/DS4SD/DocLayNet), [CDLA](https://github.com/buptlihang/CDLA), [D4LA](https://github.com/AlibabaResearch/AdvancedLiterateMachinery), [DocGenome](https://github.com/Alpha-Innovator/DocGenome), [PubTabNet](https://github.com/ibm-aur-nlp/PubTabNet), and [UniMER-1M](https://github.com/opendatalab/UniMERNet) for providing valuable datasets. We also thank everyone who contributed to this open-source effort.

## Limitation

Currently, MonkeyOCR do not yet fully support for photographed text, handwritten content, Traditional Chinese characters, or multilingual text. We plan to consider adding support for these features in future public releases. Additionally, our model is deployed on a single GPU, so if too many users upload files at the same time, issues like “This application is currently busy” may occur. The processing time shown on the demo page does not reflect computation time alone—it also includes result uploading and other overhead. During periods of high traffic, this time may be longer. The inference speeds of MonkeyOCR, MinerU, and Qwen2.5 VL-7B were measured on an H800 GPU.

## Copyright

Please don’t hesitate to share your valuable feedback — it’s a key motivation that drives us to continuously improve our framework. Note: Our model is intended for academic research and non-commercial use only. If you are interested in faster (smaller) or stronger one, please contact us at xbai@hust.edu.cn or ylliu@hust.edu.cn.