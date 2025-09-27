html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MonkeyOCR: Advanced Document Parsing with AI</title>
    <meta name="description" content="MonkeyOCR uses a Structure-Recognition-Relation (SRR) triplet paradigm to accurately parse documents. Get state-of-the-art results for text, tables, and formulas, including Chinese and English support.">
    <meta name="keywords" content="OCR, document parsing, AI, text recognition, table extraction, formula recognition, Chinese OCR, English OCR, MonkeyOCR">
    <style>
        body { font-family: sans-serif; }
        h1, h2, h3 { color: #333; }
        ul { list-style-type: disc; }
        a { color: #007bff; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .key-feature { font-weight: bold; }
        table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>

<div align="center">
    <h1>MonkeyOCR: AI-Powered Document Parsing</h1>
    <p><em>Parse documents effortlessly with MonkeyOCR's cutting-edge structure recognition and content extraction.</em></p>
    <p>
        <a href="https://arxiv.org/abs/2506.05218"><img src="https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv" alt="arXiv"></a>
        <a href="https://huggingface.co/echo840/MonkeyOCR-pro-3B"><img src="https://img.shields.io/badge/HuggingFace-black.svg?logo=HuggingFace" alt="HuggingFace"></a>
        <a href="https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue"><img src="https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues" alt="GitHub Issues"></a>
        <a href="https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed"><img src="https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues" alt="GitHub Closed Issues"></a>
        <a href="https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt"><img src="https://img.shields.io/badge/License-Apache%202.0-yellow" alt="License"></a>
        <a href="https://github.com/Yuliang-Liu/MonkeyOCR"><img src="https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views" alt="GitHub Views"></a>
    </p>
    <p>
        <a href="https://arxiv.org/abs/2506.05218"><img src="https://img.shields.io/badge/Arxiv-b31b1b.svg?logo=arXiv" alt="arXiv"></a>
        <a href="README.md"><img src="https://img.shields.io/badge/Code-Available-white" alt="Code"></a>
        <a href="https://huggingface.co/echo840/MonkeyOCR"><img src="https://img.shields.io/badge/Model%20Weight-gray" alt="Model Weight"></a>
        <a href="https://modelscope.cn/models/l1731396519/MonkeyOCR"><img src="https://img.shields.io/badge/ModelScope-green" alt="ModelScope"></a>
        <a href="https://openbayes.com/console/public/tutorials/91ESrGvEvBq"><img src="https://img.shields.io/badge/Openbayes-yellow" alt="Public Courses"></a>
        <a href="http://vlrlabmonkey.xyz:7685/"><img src="https://img.shields.io/badge/Demo-blue" alt="Demo"></a>
    </p>
</div>

<h2>Key Features</h2>
<ul>
    <li><span class="key-feature">State-of-the-Art Performance:</span> MonkeyOCR excels in parsing both English and Chinese documents, delivering superior results compared to other models.</li>
    <li><span class="key-feature">Structure-Recognition-Relation Triplet Paradigm:</span>  Employs a novel SRR approach for efficient and accurate document understanding.</li>
    <li><span class="key-feature">High Accuracy and Speed:</span>  Offers impressive performance in text recognition, table extraction, and formula identification, with notable speed improvements.</li>
    <li><span class="key-feature">Multiple Model Versions:</span> MonkeyOCR-pro-1.2B surpasses the 3B version in speed and accuracy.</li>
    <li><span class="key-feature">Supports Diverse Document Types:</span> Handles a wide range of document formats, including books, reports, academic papers, and financial documents.</li>
    <li><span class="key-feature">Easy to Use:</span> Quick start guide for local installation, model download and deployment.</li>
</ul>

<h2>Introduction</h2>
<p>MonkeyOCR utilizes a Structure-Recognition-Relation (SRR) triplet paradigm, simplifying the document parsing pipeline. It is designed to efficiently process full-page documents without the overhead of large multimodal models.  For more details, visit the <a href="https://github.com/Yuliang-Liu/MonkeyOCR">original repository</a>.</p>

<h3>Key Achievements:</h3>
<ul>
    <li>MonkeyOCR-pro-1.2B outperforms MonkeyOCR-3B on Chinese documents by 7.4%.</li>
    <li>MonkeyOCR-pro-1.2B delivers approximately a 36% speed improvement over MonkeyOCR-pro-3B.</li>
    <li>On olmOCR-Bench, MonkeyOCR-pro-1.2B outperforms Nanonets-OCR-3B by 7.3%.</li>
    <li>On OmniDocBench, MonkeyOCR-pro-3B achieves the best overall performance.</li>
</ul>

<h3>Performance Comparison</h3>
<p>See how MonkeyOCR compares to other advanced models:</p>
<img src="https://v1.ax1x.com/2025/07/15/EKhkhY.png" alt="Performance Comparison" border="0" />

<h2>Inference Speed</h2>

<h3>Inference Speed (Pages/s) on Different GPUs</h3>
<p>
  Download PDF page counts from: <a href="https://drive.google.com/drive/folders/1geumlJmVY7UUKdr8324sYZ0FHSAElh7m?usp=sharing">PDF</a>
</p>

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

<h3>VLM OCR Speed (Pages/s) on Different GPUs</h3>
<p>
  Download PDF page counts from: <a href="https://drive.google.com/drive/folders/1geumlJmVY7UUKdr8324sYZ0FHSAElh7m?usp=sharing">PDF</a>
</p>

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

<h2>Supported Hardware</h2>
<p>MonkeyOCR has been tested on GPUs like 3090, 4090, A6000, H800, and A100. Community contributions have extended support to various other GPUs and NPUs. See below</p>
<p>
    <ul>
        <li><a href="https://github.com/Yuliang-Liu/MonkeyOCR/issues/90">50-series GPUs</a></li>
        <li><a href="https://github.com/Yuliang-Liu/MonkeyOCR/issues/151">H200</a></li>
        <li><a href="https://github.com/Yuliang-Liu/MonkeyOCR/issues/133">L20</a></li>
        <li><a href="https://github.com/Yuliang-Liu/MonkeyOCR/issues/144">V100</a></li>
        <li><a href="https://github.com/Yuliang-Liu/MonkeyOCR/pull/1">2080 Ti</a></li>
        <li><a href="https://github.com/Yuliang-Liu/MonkeyOCR/pull/226/files">npu</a></li>
    </ul>
</p>

<h2>Recent Updates</h2>
<ul>
    <li><code>2025.07.10</code> 🚀 MonkeyOCR-pro-1.2B released - a faster and more efficient model.</li>
    <li><code>2025.06.12</code> 🚀 Trending on Hugging Face!</li>
    <li><code>2025.06.05</code> 🚀 MonkeyOCR released, supporting English and Chinese document parsing.</li>
</ul>

<h2>Quick Start</h2>
<h3>1. Local Installation</h3>
<h4>1.1. Install MonkeyOCR</h4>
<p>Follow the <a href="https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support">installation guide</a> to set up your environment.</p>
<h4>1.2. Download Model Weights</h4>
<p>Download the model from Hugging Face:</p>
<pre><code>pip install huggingface_hub
python tools/download_model.py -n MonkeyOCR-pro-3B  # or MonkeyOCR</code></pre>
<p>Alternatively, download from ModelScope:</p>
<pre><code>pip install modelscope
python tools/download_model.py -t modelscope -n MonkeyOCR-pro-3B  # or MonkeyOCR</code></pre>
<h4>1.3. Inference</h4>
<p>Use these commands to parse files or directories containing PDFs or images:</p>
<pre><code># End-to-end parsing
python parse.py input_path
# Parse files in a dir with specific group page num
python parse.py input_path -g 20
# Single-task recognition (outputs markdown only)
python parse.py input_path -t text/formula/table
# Parse PDFs in input_path and split results by pages
python parse.py input_path -s
# Specify output directory and model config file
python parse.py input_path -o ./output -c config.yaml
</code></pre>

<details>
    <summary><b>More usage examples</b></summary>
    <pre><code># Single file processing
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
    </code></pre>
</details>

<details>
    <summary><b>Output Results</b></summary>
    <p>MonkeyOCR generates three output file types:</p>
    <ol>
        <li><b>Processed Markdown File</b> (<code>your.md</code>): The final parsed content in markdown format.</li>
        <li><b>Layout Results</b> (<code>your_layout.pdf</code>): Layout results on the origin PDF.</li>
        <li><b>Intermediate Block Results</b> (<code>your_middle.json</code>): JSON with detailed block information.</li>
    </ol>
</details>

<h3>1.4. Gradio Demo</h3>
<pre><code>python demo/demo_gradio.py
</code></pre>
<p>Access the demo at <a href="http://localhost:7860">http://localhost:7860</a> after running.</p>

<h3>1.5. Fast API</h3>
<pre><code>uvicorn api.main:app --port 8000
</code></pre>
<p>Access API documentation at <a href="http://localhost:8000/docs">http://localhost:8000/docs</a>.</p>
<blockquote>
    <p><b>TIP:</b> Improve API concurrency using <code>vllm_async</code> for the inference backend.</p>
</blockquote>

<h2>2. Docker Deployment</h2>
<ol>
    <li>Navigate to the <code>docker</code> directory:
        <pre><code>cd docker</code></pre>
    </li>
    <li>
        <b>Prerequisite:</b> Ensure NVIDIA GPU support (via <code>nvidia-docker2</code>). If not enabled, run:
        <pre><code>bash env.sh</code></pre>
    </li>
    <li>Build the Docker image:
        <pre><code>docker compose build monkeyocr</code></pre>
        <blockquote>
            <p><b>IMPORTANT:</b>  For 20/30/40-series, V100, L20/L40, or similar GPUs, build the patched image:
            <pre><code>docker compose build monkeyocr-fix</code></pre></p>
        </blockquote>
    </li>
    <li>Run the container with the Gradio demo:
        <pre><code>docker compose up monkeyocr-demo</code></pre>
        <p>or start an interactive development environment:
        <pre><code>docker compose run --rm monkeyocr-dev</code></pre></p>
    </li>
    <li>Run the FastAPI service:
        <pre><code>docker compose up monkeyocr-api</code></pre>
        <p>Access API documentation at <a href="http://localhost:7861/docs">http://localhost:7861/docs</a>.</p>
</ol>

<h2>3. Windows Support</h2>
<p>See the <a href="docs/windows_support.md">windows support guide</a>.</p>

<h2>4. Quantization</h2>
<p>This model can be quantized using AWQ. Refer to the <a href="docs/Quantization.md">quantization guide</a>.</p>

<h2>Benchmark Results</h2>
<p>Evaluation results on OmniDocBench are shown below. MonkeyOCR-3B uses DocLayoutYOLO for structure detection; MonkeyOCR-3B* uses our trained structure detection model for improved Chinese performance.</p>

<h3>1. End-to-End Evaluation</h3>
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
<td>0.097</td>
<td>0.293</td>
<td>0.455</td>
<td>0.655</td>
<td>68.1</td>
<td>61.3</td>
<td>0.608</td>
<td>0.652</td>
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

<h3>2. Text Recognition Performance</h3>
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
<td>0.274