# MonkeyOCR: Unlock the Power of Document Parsing with AI

**Tired of tedious manual document processing? MonkeyOCR revolutionizes document parsing with a cutting-edge Structure-Recognition-Relation (SRR) triplet paradigm.**  ([Original Repo](https://github.com/Yuliang-Liu/MonkeyOCR))

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

## Key Features:

*   **Superior Accuracy:** MonkeyOCR-pro-1.2B surpasses MonkeyOCR-3B in Chinese document processing by 7.4%.
*   **Blazing-Fast Speed:** Experience a ~36% speed boost with MonkeyOCR-pro-1.2B compared to MonkeyOCR-pro-3B.
*   **Exceptional Performance:** Outperforms competitors on benchmarks like olmOCR-Bench and OmniDocBench, even exceeding closed-source and large open-source VLMs.
*   **SRR Paradigm:**  Simplifies the document parsing pipeline with the innovative Structure-Recognition-Relation triplet paradigm.
*   **Flexible Deployment:** Supports various hardware configurations, including 3090, 4090, A6000, H800, A100, and even 4060 (quantized models).
*   **Easy to Use:** Simple installation, model download, and intuitive command-line interface.
*   **Comprehensive Output:** Provides Markdown output, layout results, and detailed intermediate block results for in-depth analysis.
*   **Gradio and FastAPI Demo:** Quickly test and integrate MonkeyOCR with user-friendly Gradio demo and API support.
*   **Docker Deployment:**  Simple Docker setup for easy deployment.

## Introduction

MonkeyOCR adopts a Structure-Recognition-Relation (SRR) triplet paradigm, which simplifies the multi-tool pipeline of modular approaches while avoiding the inefficiency of using large multimodal models for full-page document processing.

## Performance Highlights

*   MonkeyOCR-pro-1.2B outperforms MonkeyOCR-3B by 7.4% on Chinese documents.
*   MonkeyOCR-pro-1.2B delivers approximately a 36% speed improvement over MonkeyOCR-pro-3B, with approximately 1.6% drop in performance.
*   On olmOCR-Bench, MonkeyOCR-pro-1.2B outperforms Nanonets-OCR-3B by 7.3%.
*   On OmniDocBench, MonkeyOCR-pro-3B achieves the best overall performance on both English and Chinese documents, outperforming even closed-source and extra-large open-source VLMs such as Gemini 2.0-Flash, Gemini 2.5-Pro, Qwen2.5-VL-72B, GPT-4o, and InternVL3-78B.

### Comparing MonkeyOCR with closed-source and extra large open-source VLMs.
<a href="https://zimgs.com/i/EKhkhY"><img src="https://v1.ax1x.com/2025/07/15/EKhkhY.png" alt="EKhkhY.png" border="0" /></a>

## Inference Speed (Pages/s) on Different GPUs and [PDF](https://drive.google.com/drive/folders/1geumlJmVY7UUKdr8324sYZ0FHSAElh7m?usp=sharing) Page Counts

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

## VLM OCR Speed (Pages/s) on Different GPUs and [PDF](https://drive.google.com/drive/folders/1geumlJmVY7UUKdr8324sYZ0FHSAElh7m?usp=sharing) Page Counts

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

MonkeyOCR has been tested on various GPUs, including 3090, 4090, A6000, H800, A100, and 4060. Community contributions have also extended support to 50-series GPUs, H200, L20, V100 and the 2080 Ti.

## News

*   **2025.07.10** 🚀 Released [MonkeyOCR-pro-1.2B](https://huggingface.co/echo840/MonkeyOCR-pro-1.2B).
*   **2025.06.12** 🚀 Trending on [Hugging Face](https://huggingface.co/models?sort=trending).
*   **2025.06.05** 🚀 Released [MonkeyOCR](https://huggingface.co/echo840/MonkeyOCR), for English and Chinese document parsing.

## Quick Start

### 1. Install MonkeyOCR

Follow the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) for environment setup.

### 2. Download Model Weights

Download from Hugging Face:

```bash
pip install huggingface_hub
python tools/download_model.py -n MonkeyOCR  # or MonkeyOCR-pro-1.2B
```

Or download from ModelScope:

```bash
pip install modelscope
python tools/download_model.py -t modelscope -n MonkeyOCR  # or MonkeyOCR-pro-1.2B
```

### 3. Inference

Run these commands to parse PDF or image files:

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

<details>
<summary><b>More usage examples</b></summary>

```bash
# Single file processing
python parse.py input.pdf                           # Parse single PDF file
python parse.py input.pdf -o ./output               # Parse with custom output dir
python parse.py input.pdf -s                        # Parse PDF with page splitting
python parse.py image.jpg                           # Parse single image file

# Single task recognition
python parse.py image.jpg -t text                   # Text recognition from image
python parse.py image.jpg -t formula                # Formula recognition from image
python parse.py image.jpg -t table                  # Table recognition from image
python parse.py document.pdf -t text                # Text recognition from all PDF pages

# Folder processing (all files individually)
python parse.py /path/to/folder                     # Parse all files in folder
python parse.py /path/to/folder -s                  # Parse with page splitting
python parse.py /path/to/folder -t text             # Single task recognition for all files

# Multi-file grouping (batch processing by page count)
python parse.py /path/to/folder -g 5                # Group files with max 5 total pages
python parse.py /path/to/folder -g 10 -s            # Group files with page splitting
python parse.py /path/to/folder -g 8 -t text        # Group files for single task recognition

# Advanced configurations
python parse.py input.pdf -c model_configs.yaml     # Custom model configuration
python parse.py /path/to/folder -g 15 -s -o ./out   # Group files, split pages, custom output
python parse.py input.pdf --pred-abandon            # Enable predicting abandon elements
```

</details>

<details>
<summary><b>Output Results</b></summary>

MonkeyOCR generates these files:

1.  `your.md`: Markdown file with parsed content.
2.  `your_layout.pdf`:  Layout results.
3.  `your_middle.json`: Detailed block information in JSON.

</details>

### 4. Gradio Demo

```bash
python demo/demo_gradio.py
```

Access the demo at `http://localhost:7860`.

### 5. Fast API

```bash
uvicorn api.main:app --port 8000
```

Access API documentation at `http://localhost:8000/docs`.  Consider `lmdeploy_queue` or `vllm_queue` for improved API concurrency.

## Docker Deployment

1.  `cd docker`
2.  **Prerequisite:**  NVIDIA GPU support in Docker (`nvidia-docker2`). If needed, run `bash env.sh`.
3.  Build:  `docker compose build monkeyocr`
    *   **Important:**  For 20/30/40-series, V100, or similar GPUs, build the patched image for LMDeploy: `docker compose build monkeyocr-fix`.
4.  Run Gradio demo: `docker compose up monkeyocr-demo`
    *   Or, interactive dev environment: `docker compose run --rm monkeyocr-dev`
5.  Run FastAPI: `docker compose up monkeyocr-api`

    Access API documentation at `http://localhost:7861/docs`.

## Windows Support

See the [windows support guide](docs/windows_support.md).

## Quantization

Quantize with AWQ:  Follow the [quantization guide](docs/Quantization.md).

## Benchmark Results

### 1. The end-to-end evaluation results of different tasks.

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
<td rowspan="8"><strong>Pipeline<br>Tools</strong></td>
<td>MinerU</td>
<td>0.150</td>
<td>0.357</td>
<td>0.061</td>
<td>0.215</td>
<td>0.278</td>
<td>0.577</td>
<td>78.6</td>
<td>62.1</td>
<td>0.180</td>
<td>0.344</td>
<td>0.079</td>
<td>0.292</td>
</tr>
<tr>
<td>Marker</td>
<td>0.336</td>
<td>0.556</td>
<td>0.080</td>
<td>0.315</td>
<td>0.530</td>
<td>0.883</td>
<td>67.6</td>
<td>49.2</td>
<td>0.619</td>
<td>0.685</td>
<td>0.114</td>
<td>0.340</td>
</tr>
<tr>
<td>Mathpix</td>
<td>0.191</td>
<td>0.365</td>
<td>0.105</td>
<td>0.384</td>
<td>0.306</td>
<td><strong>0.454</strong></td>
<td>77.0</td>
<td>67.1</td>
<td>0.243</td>
<td>0.320</td>
<td>0.108</td>
<td>0.304</td>
</tr>
<tr>
<td>Docling</td>
<td>0.589</td>
<td>0.909</td>
<td>0.416</td>
<td>0.987</td>
<td>0.999</td>
<td>1</td>
<td>61.3</td>
<td>25.0</td>
<td>0.627</td>
<td>0.810</td>
<td>0.313</td>
<td>0.837</td>
</tr>
<tr>
<td>Pix2Text</td>
<td>0.320</td>
<td>0.528</td>
<td>0.138</td>
<td>0.356</td>
<td>0.276</td>
<td>0.611</td>
<td>73.6</td>
<td>66.2</td>
<td>0.584</td>
<td>0.645</td>
<td>0.281</td>
<td>0.499</td>
</tr>
<tr>
<td>Unstructured</td>
<td>0.586</td>
<td>0.716</td>
<td>0.198</td>
<td>0.481</td>
<td>0.999</td>
<td>1</td>
<td>0</td>
<td>0.06</td>
<td>1</td>
<td>0.998</td>
<td>0.145</td>
<td>0.387</td>
</tr>
<tr>
<td>OpenParse</td>
<td>0.646</td>
<td>0.814</td>
<td>0.681</td>
<td>0.974</td>
<td>0.996</td>
<td>1</td>
<td>64.8</td>
<td>27.5</td>
<td>0.284</td>
<td>0.639</td>
<td>0.595</td>
<td>0.641</td>
</tr>
<tr>
<td>PPStruct-V3</td>
<td>0.145</td>
<td><strong>0.206</strong></td>
<td>0.058</td>
<td><strong>0.088</strong></td>
<td>0.295</td>
<td>0.535</td>
<td>-</td>
<td>-</td>
<td>0.159</td>
<td><strong>0.109</strong></td>
<td><strong>0.069</strong></td>
<td><strong>0.091</strong></td>
</tr>
<tr>
<td rowspan="8"><strong>Expert<br>VLMs</strong></td>
<td>GOT-OCR</td>
<td>0.287</td>
<td>0.411</td>
<td>0.189</td>
<td>0.315</td>
<td>0.360</td>
<td>0.528</td>
<td>53.2</td>
<td>47.2</td>
<td>0.459</td>
<td>0.520</td>
<td>0.141</td>
<td>0.280</td>
</tr>
<tr>
<td>Nougat</td>
<td>0.452</td>
<td>0.973</td>
<td>0.365</td>
<td>0.998</td>
<td>0.488</td>
<td>0.941</td>
<td>39.9</td>
<td>0</td>
<td>0.572</td>
<td>1.000</td>
<td>0.382</td>
<td>0.954</td>
</tr>
<tr>
<td>Mistral OCR</td>
<td>0.268</td>
<td>0.439</td>
<td>0.072</td>
<td>0.325</td>
<td>0.318</td>
<td>0.495</td>
<td>75.8</td>
<td>63.6</td>
<td>0.600</td>
<td>0.650</td>
<td>0.083</td>
<td>0.284</td>
</tr>
<tr>
<td>OLMOCR-sglang</td>
<td>0.326</td>
<td>0.469</td>
<td>0.097</td>
<td>0.293</td>
<td>0.455</td>
<td>0.655</td>
<td>68.1</td>
<td>61.3</td>
<td>0.608<td>0.652</td>
<td>0.145</td>
<td>0.277</td>
</tr>
<tr>
<td>SmolDocling-256M</td>
<td>0.493</td>
<td>0.816</td>
<td>0.262</td>
<td>0.838</td>
<td>0.753</td>
<td>0.997</td>
<td>44.9</td>
<td>16.5</td>
<td>0.729</td>
<td>0.907</td>
<td>0.227</td>
<td>0.522</td>
</tr>
<tr>
<td>Dolphin</td>
<td>0.206</td>
<td>0.306</td>
<td>0.107</td>
<td>0.197</td>
<td>0.447</td>
<td>0.580</td>
<td>77.3</td>
<td>67.2</td>
<td>0.180</td>
<td>0.285</td>
<td>0.091</td>
<td>0.162</td>
</tr>
<tr>
<td>MinerU 2</td>
<td>0.139</td>
<td>0.240</td>
<td><strong>0.047</strong></td>
<td>0.109</td>
<td>0.297</td>
<td>0.536</td>
<td><strong>82.5</strong></td>
<td>79.0</td>
<td>0.141</td>
<td>0.195</td>
<td><strong>0.069</strong></td>
<td>0.118</td>
</tr>
<tr>
<td>OCRFlux</td>
	
<td>0.195</td>
<td>0.281</td>
<td>0.064</td>
<td>0.183</td>
<td>0.379</td>
<td>0.613</td>
<td>71.6</td>
<td>81.3</td>
<td>0.253</td>
<td>0.139</td>
<td>0.086</td>
<td>0.187</td>


</tr>
<tr>
<td rowspan="3"><strong>General<br>VLMs</strong></td>
<td>GPT4o</td>
<td>0.233</td>
<td>0.399</td>
<td>0.144</td>
<td>0.409</td>
<td>0.425</td>
<td>0.606</td>
<td>72.0</td>
<td>62.9</td>
<td>0.234</td>
<td>0.329</td>
<td>0.128</td>
<td>0.251</td>
</tr>
<tr>
<td>Qwen2.5-VL-7B</td>
<td>0.312</td>
<td>0.406</td>
<td>0.157</td>
<td>0.228</td>
<td>0.351</td>
<td>0.574</td>
<td>76.4</td>
<td>72.2</td>
<td>0.588</td>
<td>0.619</td>
<td>0.149</td>
<td>0.203</td>
</tr>
<tr>
<td>InternVL3-8B</td>
<td>0.314</td>
<td>0.383</td>
<td>0.134</td>
<td>0.218</td>
<td>0.417</td>
<td>0.563</td>
<td>66.1</td>
<td>73.1</td>
<td>0.586</td>
<td>0.564</td>
<td>0.118</td>
<td>0.186</td>
</tr>
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


### 2. The end-to-end text recognition performance across 9 PDF page types.

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
<td rowspan="3"><strong>Pipeline<br>Tools</strong></td>
<td>MinerU</td>
<td>0.055</td>
<td>0.124</td>
<td><u>0.033</u></td>
<td>0.102</td>
<td>0.159</td>
<td><strong>0.072</strong></td>
<td><u>0.025</u></td>
<td>0.984</td>
<td>0.171</td>
<td>0.206</td>
</tr>
<tr>
<td>Marker</td>
<td>0.074</td>
<td>0.340</td>
<td>0.089</td>
<td>0.319</td>
<td>0.452</td>
<td>0.153</td>
<td>0.059</td>
<td>0.651</td>
<td>0.192</td>
<td>0.274</td>
</tr>
<tr>
<td>Mathpix</td>
<td>0.131</td>
<td>0.220</td>
<td>0.202</td>
<td>0.216</td>
<td>0.278</td>
<td>0.147</td>
<td>0.091</td>
<td>0.634</td>
<td>0.690</td>
<td>0.300</td>
</tr>
<tr>
<td rowspan="4"><strong>Expert<br>VLMs</strong></td>
<td>GOT-OCR</td>
<td>0.111</td>
<td>0.222</td>
<td>0.067</td>
<td>0.132</td>
<td>0.204</td>
<td>0.198</td>
<td>0.179</td>
<td>0.388</td>
<td>0.771</td>
<td>0.267</td>
</tr>
<tr>
<td>Nougat</td>
<td>0.734</td>
<td>0.958</td>
<td>1.000</td>
<td>0.820</td>
<td>0.930</td>
<td>0.830</td>
<td>0.214</td>
<td>0.991</td>
<td>0.871</td>
<td>0.806</td>
</tr>
<tr>
<td>Dolphin</td>
<td>0.091</td>
<td>0.131</td>
<td>0.057</td>
<td>0.146</td>
<td>0.231</td>
<td>0.121</td>
<td>0.074</td>
<td>0.363</td>
<td>0.307</td>
<td>0.177</td>
</tr>
<tr>
<td>OCRFlux</td>
<td>0.068</td>
<td>0.125</td>
<td>0.092</td>
<td>0.102</td>
<td>0.119</td>
<td>0.083</td>
<td>0.047</td>
<td>0.223</td>
<td>0.536</td>
<td>0.149</td>
</tr>
<tr>
<td rowspan="3"><strong>General<br>VLMs</strong></td>
<td>GPT4o</td>
<td>0.157</td>
<td>0.163</td>
<td>0.348</td>
<td>0.187</td>
<td>0.281</td>
<td>0.173</td>
<td>0.146</td>
<td>0.607</td>
<td>0.751</td>
<td>0.316</td>
</tr>
<tr>
<td>Qwen2.5-VL-7B</td>
<td>0.148</td>
<td><strong>0.053</strong></td>
<td>0.111</td>
<td>0.137</td>
<td>0.189</td>
<td>0.117</td>
<td>0.134</td>
<td>0.204</td>
<td>0.706</td>
<td>0.205</td>
</tr>
<tr>
<td>InternVL3-8B</td>
<td>0.163</td>
<td><u>0.056</u></td>
<td>0.107</td>
<td>0.109</td>
<td>0.129</td>
<td>0.100</td>
<td>0.159</td>
<td><strong>0.150</strong></td>
<td>0.681</td>
<td>0.188</td>
</tr>
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
<td>