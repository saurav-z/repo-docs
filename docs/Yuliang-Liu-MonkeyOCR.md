# MonkeyOCR: Effortless Document Parsing with Advanced Structure Recognition

**MonkeyOCR revolutionizes document parsing by employing a cutting-edge Structure-Recognition-Relation (SRR) triplet paradigm, delivering superior accuracy and efficiency.** Explore the original repository [here](https://github.com/Yuliang-Liu/MonkeyOCR).

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

## Key Features

*   **Superior Accuracy:** Outperforms leading open-source and closed-source VLMs, including Gemini 2.0-Flash, Gemini 2.5-Pro, Qwen2.5-VL-72B, GPT-4o, and InternVL3-78B on OmniDocBench.
*   **Exceptional Speed:** MonkeyOCR-pro-1.2B offers a 36% speed improvement over MonkeyOCR-pro-3B while maintaining impressive performance.
*   **SRR Paradigm:** Simplifies document processing, avoiding the complexity of traditional multi-tool pipelines and the inefficiencies of large multimodal models.
*   **Versatile Deployment:** Supports a wide range of hardware, from 3090s to H800s and even 4060s, offering flexibility in your environment.
*   **Comprehensive Output:** Generates Markdown files, layout results, and detailed intermediate block results (JSON), enabling flexible usage and further analysis.
*   **Gradio Demo:** Interact easily with a user-friendly web demo showcasing structure detection, content recognition, and relationship prediction.
*   **FastAPI:** Deployable via a Fast API service for easy integration into your workflows.
*   **Docker Support:** Easily deploy MonkeyOCR with Docker Compose.
*   **Quantization Support:** Offers model quantization for improved performance on a wider range of hardware.

## Performance Highlights

### Competitive Evaluation
MonkeyOCR-pro-3B excels across various metrics on the OmniDocBench benchmark, showcasing remarkable improvements over other VLM methods.

### Comparison to Other Models on OmniDocBench

<!-- Table data formatted for markdown -->
<table>
<thead>
<tr>
<th rowspan="2"><strong>Model<br>Type</strong></th>
<th rowspan="2"><strong>Methods</strong></th>
<th colspan="2"><strong>Overall<sup>Edit</sup>↓</strong></th>
<th colspan="2"><strong>Text<sup>Edit</sup>↓</strong></th>
<th colspan="2"><strong>Formula<sup>Edit</sup>↓</strong></th>
<th colspan="2"><strong>Table<sup>TEDS</sup>↑</strong></th>
<th colspan="2"><strong>Table<sup>Edit</sup>↓</strong></th>
<th colspan="2"><strong>Read Order<sup>Edit</sup>↓</strong></th>
</tr>
<tr>
<th><em>EN</em></th>
<th><em>ZH</em></th>
<th><em>EN</em></th>
<th><em>ZH</em></th>
<th><em>EN</em></th>
<th><em>ZH</em></th>
<th><em>EN</em></th>
<th><em>ZH</em></th>
<th><em>EN</em></th>
<th><em>ZH</em></th>
<th><em>EN</em></th>
<th><em>ZH</em></th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="4"><strong>Mix</strong></td>
<td><strong>MonkeyOCR-3B <a href="https://huggingface.co/echo840/MonkeyOCR/blob/main/Structure/doclayout_yolo_docstructbench_imgsz1280_2501.pt">[Weight]</a></strong></td>
<td>0.140</td>
<td>0.297</td>
<td>0.058</td>
<td>0.185</td>
<td>0.238</td>
<td>0.506</td>
<td>80.2</td>
<td>77.7</td>
<td>0.170</td>
<td>0.253</td>
<td>0.093</td>
<td>0.244</td>
</tr>
<tr>
<td><strong>MonkeyOCR-3B* <a href="https://huggingface.co/echo840/MonkeyOCR/blob/main/Structure/layout_zh.pt">[Weight]</a></strong></td>
<td>0.154</td>
<td>0.277</td>
<td>0.073</td>
<td>0.134</td>
<td>0.255</td>
<td>0.529</td>
<td>78.2</td>
<td>76.2</td>
<td>0.182</td>
<td>0.262</td>
<td>0.105</td>
<td>0.183</td>
</tr>
<tr>
<td><strong>MonkeyOCR-pro-3B <a href="http://vlrlabmonkey.xyz:7685/">[Demo]</a></strong></td>
<td><strong>0.138</strong></td>
<td><strong>0.206</strong></td>
<td>0.067</td>
<td>0.107</td>
<td><strong>0.246</strong></td>
<td><strong>0.421</strong></td>
<td>81.5</td>
<td><strong>87.5</strong></td>
<td><strong>0.139</strong></td>
<td>0.111</td>
<td>0.100</td>
<td>0.185</td>
</tr>
<tr>
<td><strong>MonkeyOCR-pro-1.2B <a href="https://huggingface.co/echo840/MonkeyOCR-pro-1.2B">[Weight]</a></strong></td>
<td>0.153</td>
<td>0.223</td>
<td>0.066</td>
<td>0.123</td>
<td>0.272</td>
<td>0.449</td>
<td>76.5</td>
<td>83.7</td>
<td>0.176</td>
<td>0.131</td>
<td>0.097</td>
<td>0.187</td>
</tr>
</tbody>
</table>

### Text Recognition Performance Across 9 PDF Page Types

<!-- Table data formatted for markdown -->
<table>
<thead>
<tr>
<th><strong>Model<br>Type</strong></th>
<th><strong>Models</strong></th>
<th><strong>Book</strong></th>
<th><strong>Slides</strong></th>
<th><strong>Financial<br>Report</strong></th>
<th><strong>Textbook</strong></th>
<th><strong>Exam<br>Paper</strong></th>
<th><strong>Magazine</strong></th>
<th><strong>Academic<br>Papers</strong></th>
<th><strong>Notes</strong></th>
<th><strong>Newspaper</strong></th>
<th><strong>Overall</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="4"><strong>Mix</strong></td>
<td><strong>MonkeyOCR-3B <a href="https://huggingface.co/echo840/MonkeyOCR/blob/main/Structure/doclayout_yolo_docstructbench_imgsz1280_2501.pt">[Weight]</a></strong></td>
<td><strong>0.046</strong></td>
<td>0.120</td>
<td><strong>0.024</strong></td>
<td>0.100</td>
<td>0.129</td>
<td>0.086</td>
<td><strong>0.024</strong></td>
<td>0.643</td>
<td><u>0.131</u></td>
<td>0.155</td>
</tr>
<tr>
<td><strong>MonkeyOCR-3B* <a href="https://huggingface.co/echo840/MonkeyOCR/blob/main/Structure/layout_zh.pt">[Weight]</a></strong></td>
<td><u>0.054</u></td>
<td>0.203</td>
<td>0.038</td>
<td>0.112</td>
<td>0.138</td>
<td>0.111</td>
<td>0.032</td>
<td>0.194</td>
<td>0.136</td>
<td>0.120</td>
</tr>
<tr>
<td><strong>MonkeyOCR-pro-3B <a href="http://vlrlabmonkey.xyz:7685/">[Demo]</a></strong></td>
<td>0.084</td>
<td>0.129</td>
<td>0.060</td>
<td><strong>0.090</strong></td>
<td><strong>0.107</strong></td>
<td><u>0.073</u></td>
<td>0.050</td>
<td><u>0.171</u></td>
<td><strong>0.107</strong></td>
<td><strong>0.100</strong></td>
</tr>
<tr>
<td><strong>MonkeyOCR-pro-1.2B <a href="https://huggingface.co/echo840/MonkeyOCR-pro-1.2B">[Weight]</a></strong></td>
<td>0.087</td>
<td>0.142</td>
<td>0.059</td>
<td><u>0.093</u></td>
<td><u>0.115</u></td>
<td>0.085</td>
<td>0.045</td>
<td>0.226</td>
<td>0.122</td>
<td><u>0.112</u></td>
</tr>
</tbody>
</table>

### olmOCR-bench Evaluation Results
MonkeyOCR consistently excels in diverse categories and outperforms other systems.

### olmOCR-bench evaluation results, with MonkeyOCR-pro-3B scoring the highest overall!
<!-- Table data formatted for markdown -->
<table>
<thead>
<tr>
<th>Model</th>
<th>ArXiv</th>
<th>Old Scans<br>Math</th>
<th>Tables</th>
<th>Old Scans</th>
<th>Headers and<br>Footers</th>
<th>Multi<br>column</th>
<th>Long Tiny<br>Text</th>
<th>Base</th>
<th>Overall</th>
</tr>
</thead>
<tbody>
<tr>
<td>MonkeyOCR-pro-3B <a href="http://vlrlabmonkey.xyz:7685/">[Demo]</a></td>
<td><strong>83.8</strong></td>
<td>68.8</td>
<td>74.6</td>
<td>36.1</td>
<td>91.2</td>
<td>76.6</td>
<td>80.1</td>
<td>95.3</td>
<td><strong>75.8 ± 1.0</strong></td>
</tr>
<tr>
<td>MonkeyOCR-pro-1.2B <a href="https://huggingface.co/echo840/MonkeyOCR-pro-1.2B">[Weight]</a></td>
<td>80.5</td>
<td>62.9</td>
<td>71.1</td>
<td>32.9</td>
<td>92.2</td>
<td>68.3</td>
<td>74.0</td>
<td>92.6</td>
<td>71.8 ± 1.1</td>
</tr>
</tbody>
</table>

## Quick Start

### Installation

1.  Follow the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) to set up your environment.
2.  Download the model weights:

```bash
pip install huggingface_hub
python tools/download_model.py -n MonkeyOCR  # or MonkeyOCR-pro-1.2B
```

### Inference

```bash
python parse.py input_path # Parses a PDF, image, or directory
```

### Gradio Demo

```bash
python demo/demo_gradio.py
```

## Deployment

*   **FastAPI:** Run `uvicorn api.main:app --port 8000` and access API documentation at http://localhost:8000/docs.
*   **Docker:**  Build and run the container using `docker compose`. The Gradio demo is available at port 7860 and the FastAPI service on port 7861.
    *   For LMDeploy compatibility with 20/30/40-series GPUs, V100, L20/L40, or similar, use the `monkeyocr-fix` image: `docker compose build monkeyocr-fix`

## Windows Support

Refer to the [windows support guide](docs/windows_support.md) for details.

## Quantization

Quantize the model using AWQ following the instructions in the [quantization guide](docs/Quantization.md).

## Visualization Demo

Experience MonkeyOCR's capabilities firsthand with the interactive demo.  [http://vlrlabmonkey.xyz:7685](http://vlrlabmonkey.xyz:7685) (select the latest available model).

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

The project acknowledges the contributions of various open-source resources and datasets including [MinerU](https://github.com/opendatalab/MinerU), [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO), and others.