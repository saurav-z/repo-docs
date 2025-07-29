# MonkeyOCR: Effortless Document Parsing with AI 🚀

**MonkeyOCR transforms documents into structured formats with remarkable speed and accuracy, using an innovative Structure-Recognition-Relation (SRR) triplet paradigm.**  [View the original repository](https://github.com/Yuliang-Liu/MonkeyOCR)

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

> **MonkeyOCR: Document Parsing with a Structure-Recognition-Relation Triplet Paradigm**<br>
> Zhang Li, Yuliang Liu, Qiang Liu, Zhiyin Ma, Ziyang Zhang, Shuo Zhang, Zidun Guo, Jiarui Zhang, Xinyu Wang, Xiang Bai <br>
[![arXiv](https://img.shields.io/badge/Arxiv-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218) 
[![Source_code](https://img.shields.io/badge/Code-Available-white)](README.md)
[![Model Weight](https://img.shields.io/badge/HuggingFace-gray)](https://huggingface.co/echo840/MonkeyOCR)
[![Model Weight](https://img.shields.io/badge/ModelScope-green)](https://modelscope.cn/models/l1731396519/MonkeyOCR)
[![Public Courses](https://img.shields.io/badge/Openbayes-yellow)](https://openbayes.com/console/public/tutorials/91ESrGvEvBq)
[![Demo](https://img.shields.io/badge/Demo-blue)](http://vlrlabmonkey.xyz:7685/)

## Key Features

*   **Superior Performance:** MonkeyOCR-pro-1.2B excels on Chinese documents, surpassing the 3B version and competing with leading closed-source and open-source models.
*   **Blazing Speed:** Experience significant speed improvements, with MonkeyOCR-pro-1.2B providing up to 36% faster processing compared to the 3B version.
*   **SRR Triplet Paradigm:** This innovative approach simplifies document parsing by recognizing document structure, text, formulas, and tables.
*   **Versatile Compatibility:** Supported by a range of GPUs, including 3090, 4090, A6000, and H800. Quantized models also support lower VRAM GPUs.
*   **Easy to Use:** Ready-to-use command-line and Gradio demo.
*   **Multi-Format Support:** Processes PDF, images, and directories containing documents.
*   **API and Docker Deployment:** Utilize a FastAPI service and readily available Docker images for straightforward integration.

## Why Choose MonkeyOCR?

MonkeyOCR simplifies the often complex process of document parsing. It leverages a Structure-Recognition-Relation (SRR) triplet paradigm, ensuring efficiency without sacrificing performance, making it a powerful tool for various applications.

## Performance Highlights

*   MonkeyOCR-pro-1.2B outperforms Nanonets-OCR-3B by 7.3% on olmOCR-Bench.
*   MonkeyOCR-pro-3B shows best overall performance on OmniDocBench for both English and Chinese Documents, even outperforming other open-source models.
*   MonkeyOCR models consistently provide strong performance across various document types (book, slides, financial reports, etc.).

### [See the comparison with other VLM models here](https://v1.ax1x.com/2025/07/15/EKhkhY.png)

## Inference Speed

The tables below show the performance of our models on different GPUs:

### Inference Speed (Pages/s)

<table>
    <thead>
		<tr align='center'>
    		<th>Model</th>
        	<th>GPU</th>
        	<th>50 Pages</th>
        	<th>100 Pages</th>
        	<th>300 Pages</th>
        	<th>500 Pages</th>
        	<th>1000 Pages</th>
    	</tr>
    </thead>
    <tbody>
    	<tr align='center'>
    		<td rowspan='4'>MonkeyOCR-pro-3B</td>
        	<td>3090</td>
        	<td>0.492</td>
        	<td>0.484</td>
        	<td>0.497</td>
        	<td>0.492</td>
        	<td>0.496</td>
    	</tr>
    	<tr align='center'>
        	<td>A6000</td>
        	<td>0.585</td>
        	<td>0.587</td>
        	<td>0.609</td>
        	<td>0.598</td>
        	<td>0.608</td>
    	</tr>
    	<tr align='center'>
        	<td>H800</td>
        	<td>0.923</td>
        	<td>0.768</td>
        	<td>0.897</td>
        	<td>0.930</td>
        	<td>0.891</td>
    	</tr>
    	<tr align='center'>
        	<td>4090</td>
        	<td>0.972</td>
        	<td>0.969</td>
        	<td>1.006</td>
        	<td>0.986</td>
        	<td>1.006</td>
    	</tr>
    	<tr align='center'>
    		<td rowspan='4'>MonkeyOCR-pro-1.2B</td>
        	<td>3090</td>
        	<td>0.615</td>
        	<td>0.660</td>
        	<td>0.677</td>
        	<td>0.687</td>
        	<td>0.683</td>
    	</tr>
    	<tr align='center'>
        	<td>A6000</td>
        	<td>0.709</td>
        	<td>0.786</td>
        	<td>0.825</td>
        	<td>0.829</td>
        	<td>0.825</td>
   		</tr>
    	<tr align='center'>
        	<td>H800</td>
        	<td>0.965</td>
        	<td>1.082</td>
        	<td>1.101</td>
        	<td>1.145</td>
        	<td>1.015</td>
    	</tr>
    	<tr align='center'>
        	<td>4090</td>
        	<td>1.194</td>
        	<td>1.314</td>
        	<td>1.436</td>
        	<td>1.442</td>
        	<td>1.434</td>
    	</tr>
    </tbody>
</table>

### VLM OCR Speed (Pages/s)

<table>
    <thead>
		<tr align='center'>
    		<th>Model</th>
        	<th>GPU</th>
        	<th>50 Pages</th>
        	<th>100 Pages</th>
        	<th>300 Pages</th>
        	<th>500 Pages</th>
        	<th>1000 Pages</th>
    	</tr>
    </thead>
    <tbody>
    	<tr align='center'>
    		<td rowspan='4'>MonkeyOCR-pro-3B</td>
        	<td>3090</td>
        	<td>0.705</td>
        	<td>0.680</td>
        	<td>0.711</td>
        	<td>0.700</td>
        	<td>0.724</td>
    	</tr>
    	<tr align='center'>
        	<td>A6000</td>
        	<td>0.885</td>
        	<td>0.860</td>
        	<td>0.915</td>
        	<td>0.892</td>
        	<td>0.934</td>
    	</tr>
    	<tr align='center'>
        	<td>H800</td>
        	<td>1.371</td>
        	<td>1.135</td>
        	<td>1.339</td>
        	<td>1.433</td>
        	<td>1.509</td>
    	</tr>
    	<tr align='center'>
        	<td>4090</td>
        	<td>1.321</td>
        	<td>1.300</td>
        	<td>1.384</td>
        	<td>1.343</td>
        	<td>1.410</td>
    	</tr>
    	<tr align='center'>
    		<td rowspan='4'>MonkeyOCR-pro-1.2B</td>
        	<td>3090</td>
        	<td>0.919</td>
        	<td>1.086</td>
        	<td>1.166</td>
        	<td>1.182</td>
        	<td>1.199</td>
    	</tr>
    	<tr align='center'>
        	<td>A6000</td>
        	<td>1.177</td>
        	<td>1.361</td>
        	<td>1.506</td>
        	<td>1.525</td>
        	<td>1.569</td>
   		</tr>
    	<tr align='center'>
        	<td>H800</td>
        	<td>1.466</td>
        	<td>1.719</td>
        	<td>1.763</td>
        	<td>1.875</td>
        	<td>1.650</td>
    	</tr>
    	<tr align='center'>
        	<td>4090</td>
        	<td>1.759</td>
        	<td>1.987</td>
        	<td>2.260</td>
        	<td>2.345</td>
        	<td>2.415</td>
    	</tr>
    </tbody>
</table>

## Supported Hardware

MonkeyOCR has been tested on a wide range of GPUs, including the 3090, 4090, A6000, H800, A100, and 4060, as well as the 50-series GPUs, H200, L20, V100, 2080 Ti, and npu.

## News

*   ```2025.07.10``` 🚀 [MonkeyOCR-pro-1.2B](https://huggingface.co/echo840/MonkeyOCR-pro-1.2B) Released: A faster, leaner version that boosts accuracy and efficiency.
*   ```2025.06.12``` 🚀 Trending on [Hugging Face](https://huggingface.co/models?sort=trending).
*   ```2025.06.05``` 🚀 [MonkeyOCR](https://huggingface.co/echo840/MonkeyOCR) Released: A model for English and Chinese document parsing.

## Quick Start

### Local Installation

1.  **Install MonkeyOCR:** Follow the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) to set up your environment.
2.  **Download Model Weights:** Download the model from Hugging Face or ModelScope:

    ```bash
    pip install huggingface_hub
    python tools/download_model.py -n MonkeyOCR  # or MonkeyOCR-pro-1.2B
    ```

    or

    ```bash
    pip install modelscope
    python tools/download_model.py -t modelscope -n MonkeyOCR  # or MonkeyOCR-pro-1.2B
    ```
3.  **Inference:** Use the following commands to parse files:

    ```bash
    # Replace input_path with a PDF, image, or directory.
    python parse.py input_path  # End-to-end parsing
    python parse.py input_path -g 20 # Group files by page count
    python parse.py input_path -t text/formula/table  # Single-task recognition
    python parse.py input_path -s  # Split results by pages
    python parse.py input_path -o ./output -c config.yaml # Specify output directory and config
    ```

    [See more usage examples here](#quick-start)
4.  **Gradio Demo:** Run the demo with:

    ```bash
    python demo/demo_gradio.py
    ```

    Access the demo at http://localhost:7860.
5.  **FastAPI:** Start the service with:

    ```bash
    uvicorn api.main:app --port 8000
    ```

    Access the API documentation at http://localhost:8000/docs.
    > [!TIP]
    > To improve API concurrency performance, consider configuring the inference backend as `lmdeploy_queue` or `vllm_queue`.

### Docker Deployment

1.  Navigate to the `docker` directory:

    ```bash
    cd docker
    ```

2.  **Prerequisite:** Ensure NVIDIA GPU support is available in Docker (via `nvidia-docker2`).
   If GPU support is not enabled, run the following to set up the environment:

   ```bash
   bash env.sh
   ```
3.  Build the Docker image:

    ```bash
    docker compose build monkeyocr
    ```

> [!IMPORTANT]
>
> If your GPU is from the 20/30/40-series, V100, or similar, please build the patched Docker image for LMDeploy compatibility:
>
> ```bash
> docker compose build monkeyocr-fix
> ```
>
> Otherwise, you may encounter the following error: `triton.runtime.errors.OutOfResources: out of resource: shared memory`
4.  Run the Gradio demo:

    ```bash
    docker compose up monkeyocr-demo
    ```
    Or, for an interactive environment:
    ```bash
    docker compose run --rm monkeyocr-dev
    ```

5.  Run the FastAPI service:

    ```bash
    docker compose up monkeyocr-api
    ```

    Access the API documentation at http://localhost:7861/docs.

### Windows Support

See the [windows support guide](docs/windows_support.md) for details.

### Quantization

Quantize the model using AWQ; see the [quantization guide](docs/Quantization.md).

## Benchmark Results

### 1. OmniDocBench Results

The tables below present the end-to-end evaluation results of different tasks and text recognition performance.

### 2. Text Recognition Performance Across PDF Page Types

[See the evaluation tables here](#benchmark-results)

### 3. olmOCR-bench

[See the evaluation tables here](#benchmark-results)

## Visualization Demo

Get a quick hands-on experience with our demo at:  http://vlrlabmonkey.xyz:7685 (select the latest model.)

> Our demo is simple:
>
> 1.  Upload a PDF or image.
> 2.  Click “Parse (解析)” for structure detection and content recognition.
> 3.  Select a prompt and click “Test by prompt” to perform recognition based on the prompt.

### Examples:

*   [Support diverse Chinese and English PDF types](https://github.com/Yuliang-Liu/MonkeyOCR#visualization-demo)
*   [Formula document](https://v1.ax1x.com/2025/06/10/7jVLgB.jpg)
*   [Table document](https://v1.ax1x.com/2025/06/11/7jcOaa.png)
*   [Newspaper](https://v1.ax1x.com/2025/06/11/7jcP5V.png)
*   [Financial report](https://v1.ax1x.com/2025/06/11/7jc10I.png)

## Citing MonkeyOCR

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

We thank the creators of [MinerU](https://github.com/opendatalab/MinerU), [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO), [PyMuPDF](https://github.com/pymupdf/PyMuPDF), [layoutreader](https://github.com/ppaanngggg/layoutreader), [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [LMDeploy](https://github.com/InternLM/lmdeploy), [PP-StructureV3](https://github.com/PaddlePaddle/PaddleOCR), [PP-DocLayout_plus-L](https://huggingface.co/PaddlePaddle/PP-DocLayout_plus-L), and [InternVL3](https://github.com/OpenGVLab/InternVL) for their contributions. We also thank [M6Doc](https://github.com/HCIILAB/M6Doc), [DocLayNet](https://github.com/DS4SD/DocLayNet), [CDLA](https://github.com/buptlihang/CDLA), [D4LA](https://github.com/AlibabaResearch/AdvancedLiterateMachinery), [DocGenome](https://github.com/Alpha-Innovator/DocGenome), [PubTabNet](https://github.com/ibm-aur-nlp/PubTabNet), and [UniMER-1M](https://github.com/opendatalab/UniMERNet) for the datasets. We also thank everyone who contributed to this open-source effort.

## Limitation

MonkeyOCR currently lacks full support for photographed and handwritten text, Traditional Chinese characters, and multilingual text. Single GPU deployment may lead to slowdowns during heavy traffic.

## Copyright

We welcome your feedback. For inquiries, please contact xbai@hust.edu.cn or ylliu@hust.edu.cn. Note: MonkeyOCR is intended for academic research and non-commercial use only.