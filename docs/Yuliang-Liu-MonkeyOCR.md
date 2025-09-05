# MonkeyOCR: Effortless Document Parsing with AI

**Unlock the power of intelligent document processing with MonkeyOCR, a cutting-edge solution built on a Structure-Recognition-Relation (SRR) triplet paradigm!**  [Go to the original repository](https://github.com/Yuliang-Liu/MonkeyOCR)

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

MonkeyOCR leverages the Structure-Recognition-Relation (SRR) triplet paradigm, streamlining document parsing by simplifying modular approaches and avoiding the inefficiencies of large multimodal models.  It excels at extracting information from documents, including text, tables, and formulas, with impressive speed and accuracy.

## Key Features:

*   **Superior Performance:** MonkeyOCR-pro-1.2B outperforms MonkeyOCR-3B, delivers speed improvements, and achieves top results on various benchmarks.
    *   MonkeyOCR-pro-1.2B surpasses MonkeyOCR-3B by 7.4% on Chinese documents.
    *   Approximately 36% speed improvement over MonkeyOCR-pro-3B, with ~1.6% performance drop.
    *   Outperforms Nanonets-OCR-3B by 7.3% on olmOCR-Bench.
    *   Achieves the best overall performance on OmniDocBench, outperforming even closed-source and extra-large open-source VLMs.

*   **Fast Inference:**  Benefit from rapid processing speeds on various GPUs.  See the tables below for example pages/second speeds.

*   **Comprehensive Support:**  Compatible with a variety of hardware configurations.  The model has been successfully tested on GPUs like the 3090, 4090, A6000, H800, and A100, as well as the 4060 (quantized 3B model and 1.2B model) and is community-tested on many more.

*   **User-Friendly:**  Easy to install and use, with a straightforward command-line interface and a convenient Gradio demo.
*   **Flexible Output:** MonkeyOCR generates markdown files, layout results (PDF), and intermediate block results (JSON).

## Performance Benchmarks
### Comparing MonkeyOCR with closed-source and extra large open-source VLMs.
<a href="https://zimgs.com/i/EKhkhY"><img src="https://v1.ax1x.com/2025/07/15/EKhkhY.png" alt="EKhkhY.png" border="0" /></a>

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
MonkeyOCR has been tested on various GPU including: 3090, 4090, A6000, H800, A100, and 4060.
For more details, see the [issue](https://github.com/Yuliang-Liu/MonkeyOCR/issues/90), [H200](https://github.com/Yuliang-Liu/MonkeyOCR/issues/151), [L20](https://github.com/Yuliang-Liu/MonkeyOCR/issues/133), [V100](https://github.com/Yuliang-Liu/MonkeyOCR/issues/144), [2080 Ti](https://github.com/Yuliang-Liu/MonkeyOCR/pull/1) and [npu](https://github.com/Yuliang-Liu/MonkeyOCR/pull/226/files).

## News

*   **2025.07.10:** 🚀 We release [MonkeyOCR-pro-1.2B](https://huggingface.co/echo840/MonkeyOCR-pro-1.2B), a faster and leaner version model that surpasses the previous 3B version in accuracy, speed, and efficiency.
*   **2025.06.12:** 🚀 The model's trending on [Hugging Face](https://huggingface.co/models?sort=trending). Thanks for the love!
*   **2025.06.05:** 🚀 We release [MonkeyOCR](https://huggingface.co/echo840/MonkeyOCR), an English and Chinese documents parsing model.

## Quick Start:

### Local Installation
1.  **Install MonkeyOCR:** Follow the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) to set up your environment.
2.  **Download Model Weights:** Get your model from Hugging Face or ModelScope.
    ```bash
    # Hugging Face:
    pip install huggingface_hub
    python tools/download_model.py -n MonkeyOCR-pro-3B  # or MonkeyOCR

    # ModelScope:
    pip install modelscope
    python tools/download_model.py -t modelscope -n MonkeyOCR-pro-3B  # or MonkeyOCR
    ```
3.  **Inference:** Process PDFs or images with these commands:
    ```bash
    # Single file processing
    python parse.py input.pdf
    python parse.py image.jpg

    # Parse files in a dir with specific group page num
    python parse.py input_path -g 20

    # Single-task recognition
    python parse.py input.pdf -t text/formula/table

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
      python parse.py /path/to/folder -g 10 -m            # Group files and merge text blocks in output
    ```

    </details>

    <details>
    <summary><b>Output Results</b></summary>

    MonkeyOCR mainly generates three types of output files:

    1.  **Processed Markdown File** (`your.md`): The final parsed document content in markdown format, containing text, formulas, tables, and other structured elements.
    2.  **Layout Results** (`your_layout.pdf`): The layout results drawed on origin PDF.
    3.  **Intermediate Block Results** (`your_middle.json`): A JSON file containing detailed information about all detected blocks, including:

        *   Block coordinates and positions
        *   Block content and type information
        *   Relationship information between blocks

    These files provide both the final formatted output and detailed intermediate results for further analysis or processing.

    </details>

4.  **Gradio Demo:**
    ```bash
    python demo/demo_gradio.py
    ```
    Access the demo at http://localhost:7860.

5.  **FastAPI:**
    ```bash
    uvicorn api.main:app --port 8000
    ```
    Explore the API at http://localhost:8000/docs.
    > [!TIP]
    > To improve API concurrency performance, consider configuring the inference backend as `vllm_async`.

## Docker Deployment
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
    > If your GPU is from the 20/30/40-series, V100, L20/L40 or similar, please build the patched Docker image for LMDeploy compatibility:
    >
    > ```bash
    > docker compose build monkeyocr-fix
    > ```
    >
    > Otherwise, you may encounter the following error: `triton.runtime.errors.OutOfResources: out of resource: shared memory`

4.  Run the container with the Gradio demo (accessible on port 7860):
    ```bash
    docker compose up monkeyocr-demo
    ```
    Alternatively, start an interactive development environment:
    ```bash
    docker compose run --rm monkeyocr-dev
    ```

5.  Run the FastAPI service (accessible on port 7861):
    ```bash
    docker compose up monkeyocr-api
    ```
    Access the API documentation at http://localhost:7861/docs.

## Windows Support
For detailed information, please consult the [windows support guide](docs/windows_support.md).

## Quantization
This model can be quantized using AWQ.  Find instructions in the [quantization guide](docs/Quantization.md).

## Benchmark Results:
Here are the evaluation results of our model on OmniDocBench, and the performance on olmOCR-bench.

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
<td>PP-StructureV3</td>
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
<td>0.097</td><td>0.293</td>
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
<td><strong>MonkeyOCR-pro-3B <a href="https://huggingface.co/echo840/MonkeyOCR-pro-3B">[Weight]</a></strong></td>
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
<td><strong>