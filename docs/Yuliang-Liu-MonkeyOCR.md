# MonkeyOCR: Extract Insights from Documents with Unmatched Precision (with a Structure-Recognition-Relation Triplet Paradigm)

**Unlock the power of automated document parsing with MonkeyOCR, a cutting-edge solution designed for efficient and accurate information extraction. Get started with our [original repo](https://github.com/Yuliang-Liu/MonkeyOCR)!**

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

## Key Features

*   **Superior Accuracy:**  MonkeyOCR-pro-1.2B significantly outperforms other models on both Chinese and English documents.
*   **Blazing-Fast Speed:** Experience a remarkable 36% speed increase with MonkeyOCR-pro-1.2B compared to MonkeyOCR-pro-3B.
*   **Robust Performance:** Achieves state-of-the-art results on various benchmarks, including OmniDocBench and olmOCR-Bench.
*   **Versatile Support:**  Handles diverse document types, including books, slides, financial reports, and more.
*   **Easy Deployment:**  Offers flexible deployment options via local installation, Gradio demo, FastAPI, and Docker.
*   **Open-Source Community:**  Benefit from ongoing development, community contributions, and comprehensive documentation.

## What is MonkeyOCR?

MonkeyOCR utilizes a Structure-Recognition-Relation (SRR) triplet paradigm. This innovative approach simplifies document parsing, outperforming traditional methods without relying on large multimodal models.

## Performance Highlights

MonkeyOCR shines in both accuracy and speed:

*   **OmniDocBench:** MonkeyOCR-pro-3B achieves the best overall performance on both English and Chinese documents, outperforming even closed-source and extra-large open-source VLMs.
*   **olmOCR-Bench:** MonkeyOCR-pro-1.2B surpasses Nanonets-OCR-3B by 7.3%.
*   **Chinese Documents:** MonkeyOCR-pro-1.2B surpasses MonkeyOCR-3B by 7.4% on Chinese documents.
*   **Speed Improvements:** MonkeyOCR-pro-1.2B delivers approximately a 36% speed improvement over MonkeyOCR-pro-3B, with approximately 1.6% drop in performance.

**[See the detailed evaluation results](https://github.com/Yuliang-Liu/MonkeyOCR#benchmark-results)**

## Inference Speed

Find out the speed of processing documents:

### Inference Speed (Pages/s) on Different GPUs and [PDF](https://drive.google.com/drive/folders/1geumlJmVY7UUKdr8324sYZ0FHSAElh7m?usp=sharing) Page Counts

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

## VLM OCR Speed

Find out the speed of processing documents:

### VLM OCR Speed (Pages/s) on Different GPUs and [PDF](https://drive.google.com/drive/folders/1geumlJmVY7UUKdr8324sYZ0FHSAElh7m?usp=sharing) Page Counts

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

MonkeyOCR is tested and compatible with various GPUs, including 3090, 4090, A6000, H800, and more.

## Quick Start

Get up and running with MonkeyOCR in a few simple steps:

### 1.  Installation

Follow the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) to set up your environment.

### 2. Download Model Weights

Download the model weights from Hugging Face or ModelScope:

```bash
# Hugging Face
pip install huggingface_hub
python tools/download_model.py -n MonkeyOCR-pro-3B  # or MonkeyOCR

# ModelScope
pip install modelscope
python tools/download_model.py -t modelscope -n MonkeyOCR-pro-3B  # or MonkeyOCR
```

### 3. Inference

Use the following commands to parse documents:

```bash
# Replace input_path with the path to a PDF or image or directory

# End-to-end parsing
python parse.py input_path

# Parse files in a dir with specific group page num
python parse.py input_path -g 20

# Single-task recognition (outputs markdown only)
python parse.py input_path -t text/formula/table

# Parse PDFs in input_path and split results by pages
python parse.py input_path -s

# Specify output directory and model config file
python parse.py input_path -o ./output -c config.yaml
```

**[See more usage examples](https://github.com/Yuliang-Liu/MonkeyOCR#quick-start)**

### 4. Gradio Demo

Launch the interactive demo:

```bash
python demo/demo_gradio.py
```

Access the demo at http://localhost:7860.

### 5. Fast API

Start the FastAPI service:

```bash
uvicorn api.main:app --port 8000
```

Explore the API documentation at http://localhost:8000/docs.

## Docker Deployment

Easily deploy MonkeyOCR using Docker:

1.  Navigate to the `docker` directory:

    ```bash
    cd docker
    ```

2.  Build the Docker image:

    ```bash
    docker compose build monkeyocr
    ```

3.  Run the container with the Gradio demo:

    ```bash
    docker compose up monkeyocr-demo
    ```

4.  Run the FastAPI service:

    ```bash
    docker compose up monkeyocr-api
    ```

**[See the full Docker instructions](https://github.com/Yuliang-Liu/MonkeyOCR#docker-deployment)**

## Windows Support

Detailed instructions are available in the [windows support guide](docs/windows_support.md).

## Quantization

Quantize the model for improved performance using AWQ.  Follow the instructions in the [quantization guide](docs/Quantization.md).

## Demo

Get a Quick Hands-On Experience with Our Demo:  http://vlrlabmonkey.xyz:7685 (The latest model is available for selection)

> Our demo is simple and easy to use:
>
> 1. Upload a PDF or image.
> 2. Click “Parse (解析)” to let the model perform structure detection, content recognition, and relationship prediction on the input document. The final output will be a markdown-formatted version of the document.
> 3. Select a prompt and click “Test by prompt” to let the model perform content recognition on the image based on the selected prompt.

### Support diverse Chinese and English PDF types

<p align="center">
  <img src="asserts/Visualization.GIF?raw=true" width="600"/>
</p>

### Example for formula document
<img src="https://v1.ax1x.com/2025/06/10/7jVLgB.jpg" alt="7jVLgB.jpg" border="0" />

### Example for table document
<img src="https://v1.ax1x.com/2025/06/11/7jcOaa.png" alt="7jcOaa.png" border="0" />

### Example for newspaper
<img src="https://v1.ax1x.com/2025/06/11/7jcP5V.png" alt="7jcP5V.png" border="0" />

### Example for financial report
<img src="https://v1.ax1x.com/2025/06/11/7jc10I.png" alt="7jc10I.png" border="0" />
<img src="https://v1.ax1x.com/2025/06/11/7jcRCL.png" alt="7jcRCL.png" border="0" />

## Citing MonkeyOCR

If you use MonkeyOCR in your work, please cite it using the following BibTeX entry:

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

We thank [MinerU](https://github.com/opendatalab/MinerU), [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO), [PyMuPDF](https://github.com/pymupdf/PyMuPDF), [layoutreader](https://github.com/ppaanngggg/layoutreader), [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [LMDeploy](https://github.com/InternLM/lmdeploy), [PP-StructureV3](https://github.com/PaddlePaddle/PaddleOCR), [PP-DocLayout_plus-L](https://huggingface.co/PaddlePaddle/PP-DocLayout_plus-L), and [InternVL3](https://github.com/OpenGVLab/InternVL) for providing base code and models, as well as their contributions to this field. We also thank [M6Doc](https://github.com/HCIILAB/M6Doc), [DocLayNet](https://github.com/DS4SD/DocLayNet), [CDLA](https://github.com/buptlihang/CDLA), [D4LA](https://github.com/AlibabaResearch/AdvancedLiterateMachinery), [DocGenome](https://github.com/Alpha-Innovator/DocGenome), [PubTabNet](https://github.com/ibm-aur-nlp/PubTabNet), and [UniMER-1M](https://github.com/opendatalab/UniMERNet) for providing valuable datasets. We also thank everyone who contributed to this open-source effort.

## Limitations

Please note the current limitations of MonkeyOCR:

*   Limited support for photographed text, handwritten content, Traditional Chinese characters, and multilingual text. Future releases will address these limitations.
*   Potential performance issues during peak demo usage due to single-GPU deployment.
*   The demo's processing time includes overhead beyond computation.
*   Inference speeds were measured on an H800 GPU.

## Copyright

MonkeyOCR is intended for academic research and non-commercial use only.  Your feedback is greatly appreciated!  For faster (smaller) or stronger models, contact xbai@hust.edu.cn or ylliu@hust.edu.cn.