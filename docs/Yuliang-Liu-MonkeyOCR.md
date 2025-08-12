# MonkeyOCR: The Revolutionary Document Parsing Solution

**MonkeyOCR empowers you to effortlessly parse and understand complex documents with unmatched accuracy and speed.** ([View the original repository](https://github.com/Yuliang-Liu/MonkeyOCR))

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

MonkeyOCR, built on a Structure-Recognition-Relation (SRR) triplet paradigm, offers a streamlined approach to document parsing, surpassing traditional methods.

## Key Features:

*   **Superior Accuracy**: MonkeyOCR-pro-1.2B outperforms MonkeyOCR-3B by 7.4% on Chinese documents.
*   **Blazing-Fast Speed**: MonkeyOCR-pro-1.2B delivers approximately a 36% speed improvement over MonkeyOCR-pro-3B.
*   **Competitive Performance**:  Outperforms Nanonets-OCR-3B by 7.3% on olmOCR-Bench.
*   **Industry-Leading Results**: Achieves top performance on OmniDocBench, surpassing even closed-source and large open-source models.
*   **Versatile Compatibility**: Supports various GPUs, including 3090, 4090, A6000, H800, and more.
*   **Easy Deployment**:  Quick start guide with local installation, Hugging Face and ModelScope downloads, and Docker deployment options.
*   **Flexible Output**: Generates Markdown output, layout results, and detailed intermediate block results (JSON).
*   **User-Friendly Demo**: Interactive Gradio demo for easy experimentation (http://vlrlabmonkey.xyz:7685).

## Performance Highlights:

MonkeyOCR consistently delivers exceptional results, outperforming other models on key benchmarks. Explore the detailed results below, showcasing impressive improvements in various tasks.

### [Performance table]

*  **[Performance table]**
### [Performance table]

## Quick Start Guide:

### 1. Installation

See the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support)

### 2. Download Model Weights

Download models from Hugging Face or ModelScope.

```python
# Hugging Face
pip install huggingface_hub
python tools/download_model.py -n MonkeyOCR  # or MonkeyOCR-pro-1.2B
# ModelScope
pip install modelscope
python tools/download_model.py -t modelscope -n MonkeyOCR  # or MonkeyOCR-pro-1.2B
```

### 3. Inference

Use the provided commands to parse PDFs or images:

```bash
python parse.py input_path
```

*   **[More usage examples]**

*   **[Output Results]**

### 4. Demo

Launch the interactive Gradio demo:

```bash
python demo/demo_gradio.py
```

### 5. API

Start the FastAPI service:

```bash
uvicorn api.main:app --port 8000
```

*   **[More information about the API]**

## Docker Deployment

### [Docker Instructions]

## Windows Support

See the [windows support guide](docs/windows_support.md)

## Quantization

This model can be quantized using AWQ. Follow the instructions in the [quantization guide](docs/Quantization.md).

## Citation

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

## Acknowledgements
*   **[List of acknowledgements]**

## Limitations
*   **[List of Limitations]**

## Copyright
*   **[Copyright info]**