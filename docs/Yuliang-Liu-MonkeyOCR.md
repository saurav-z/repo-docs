# MonkeyOCR: Unlock Advanced Document Parsing with Unmatched Accuracy and Speed

**MonkeyOCR revolutionizes document processing using a Structure-Recognition-Relation (SRR) triplet paradigm, delivering state-of-the-art results for complex documents.** [Explore the original repo](https://github.com/Yuliang-Liu/MonkeyOCR).

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

## Key Features:

*   **Superior Accuracy:** MonkeyOCR-pro-1.2B achieves remarkable performance, outperforming other models on multiple benchmarks.
*   **Blazing Speed:** Achieve up to 36% speed improvement compared to the previous model version.
*   **SRR Triplet Paradigm:** Simplifies document parsing, avoiding the inefficiencies of traditional methods.
*   **Versatile Support:** Processes English and Chinese documents, formulas, tables, and more.
*   **Easy Deployment:** Supports local installation, Gradio demo, FastAPI service, and Docker deployment.

## What's New:

*   **MonkeyOCR-pro-1.2B Release:** A faster and more efficient model, outperforming its predecessor in accuracy and speed.
*   **Trending on Hugging Face:**  Gaining popularity within the community.

## Performance Highlights:

*   MonkeyOCR-pro-1.2B surpasses MonkeyOCR-3B by 7.4% on Chinese documents.
*   On olmOCR-Bench, MonkeyOCR-pro-1.2B outperforms Nanonets-OCR-3B by 7.3%.
*   MonkeyOCR-pro-3B achieves the best overall performance on OmniDocBench, outperforming closed-source and extra-large open-source VLMs.

## Quickstart:

### 1. Installation:

Follow the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) for your environment.

### 2. Download Model Weights:

```bash
pip install huggingface_hub
python tools/download_model.py -n MonkeyOCR  # or MonkeyOCR-pro-1.2B
```

Or from ModelScope:
```bash
pip install modelscope
python tools/download_model.py -t modelscope -n MonkeyOCR  # or MonkeyOCR-pro-1.2B
```

### 3. Inference:

Use the `parse.py` script:

```bash
python parse.py input_path
```

See [More usage examples](https://github.com/Yuliang-Liu/MonkeyOCR#quick-start) for details.

### 4. Gradio Demo:

Run the demo:

```bash
python demo/demo_gradio.py
```

Access the demo at http://localhost:7860.

### 5. FastAPI Service:

Start the API:

```bash
uvicorn api.main:app --port 8000
```

API documentation is available at http://localhost:8000/docs.

## Benchmarks:

### OmniDocBench Results:

MonkeyOCR showcases exceptional performance on various tasks, outperforming other models.

<details>
  <summary>See Details</summary>
  
  **End-to-end evaluation results:**
  <!-- Table will be added here -->
  <table>
    <thead>
		<tr align='center'>
    		<th>Model<br>Type</th>
        	<th>Methods</th>
        	<th colspan="2">Overall<sup>Edit</sup>↓</th>
        	<th colspan="2">Text<sup>Edit</sup>↓</th>
        	<th colspan="2">Formula<sup>Edit</sup>↓</th>
        	<th colspan="2">Table<sup>TEDS</sup>↑</th>
        	<th colspan="2">Table<sup>Edit</sup>↓</th>
        	<th colspan="2">Read Order<sup>Edit</sup>↓</th>
    	</tr>
    </thead>
    <tbody>
    	<tr align='center'>
    		<td rowspan='8'>Pipeline<br>Tools</td>
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
    	<tr align='center'>
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
    	<tr align='center'>
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
    	<tr align='center'>
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
    	<tr align='center'>
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
    	<tr align='center'>
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
    	<tr align='center'>
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
    	<tr align='center'>
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
    	<tr align='center'>
    		<td rowspan='8'>Expert<br>VLMs</td>
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
    	<tr align='center'>
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
    	<tr align='center'>
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
    	<tr align='center'>
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
    	<tr align='center'>
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
    	<tr align='center'>
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
    	<tr align='center'>
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
    	<tr align='center'>
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
    	<tr align='center'>
    		<td rowspan='3'>General<br>VLMs</td>
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
    	<tr align='center'>
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
    	<tr align='center'>
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
    	<tr align='center'>
    		<td rowspan='4'>Mix</td>
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
    	<tr align='center'>
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
    	<tr align='center'>
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
    	<tr align='center'>
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
</details>

**End-to-end text recognition performance:**
<details>
  <summary>See Details</summary>
   <!-- Table will be added here -->
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
</details>

**olmOCR-Bench Results:**
<details>
  <summary>See Details</summary>

  <!-- Table will be added here -->
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
    <td>GOT OCR</td>
    <td>52.7</td>
    <td>52.0</td>
    <td>0.2</td>
    <td>22.1</td>
    <td>93.6</td>
    <td>42.0</td>
    <td>29.9</td>
    <td>94.0</td>
    <td>48.3 ± 1.1</td>
    </tr>
    <tr>
    <td>Marker</td>
    <td>76.0</td>
    <td>57.9</td>
    <td>57.6</td>
    <td>27.8</td>
    <td>84.9</td>
    <td>72.9</td>
    <td>84.6</td>
    <td><strong>99.1</strong></td>
    <td>70.1 ± 1.1</td>
    </tr>
    <tr>
    <td>MinerU</td>
    <td>75.4</td>
    <td>47.4</td>
    <td>60.9</td>
    <td>17.3</td>
    <td><strong>96.6</strong></td>
    <td>59.0</td>
    <td>39.1</td>
    <td>96.6</td>
    <td>61.5 ± 1.1</td>
    </tr>
    <tr>
    <td>Mistral OCR</td>
    <td>77.2</td>
    <td>67.5</td>
    <td>60.6</td>
    <td>29.3</td>
    <td>93.6</td>
    <td>71.3</td>
    <td>77.1</td>
    <td>99.4</td>
    <td>72.0 ± 1.1</td>
    </tr>
    <tr>
    <td>Nanonets OCR</td>
    <td>67.0</td>
    <td>68.6</td>
    <td><strong>77.7</strong></td>
    <td>39.5</td>
    <td>40.7</td>
    <td>69.9</td>
    <td>53.4</td>
    <td>99.3</td>
    <td>64.5 ± 1.1</td>
    </tr>
    <tr>
    <td>GPT-4o<br>(No Anchor)</td>
    <td>51.5</td>
    <td><strong>75.5</strong></td>
    <td>69.1</td>
    <td>40.9</td>
    <td>94.2</td>
    <td>68.9</td>
    <td>54.1</td>
    <td>96.7</td>
    <td>68.9 ± 1.1</td>
    </tr>
    <tr>
    <td>GPT-4o<br>(Anchored)</td>
    <td>53.5</td>
    <td>74.5</td>
    <td>70.0</td>
    <td>40.7</td>
    <td>93.8</td>
    <td>69.3</td>
    <td>60.6</td>
    <td>96.8</td>
    <td>69.9 ± 1.1</td>
    </tr>
    <tr>
    <td>Gemini Flash 2<br>(No Anchor)</td>
    <td>32.1</td>
    <td>56.3</td>
    <td>61.4</td>
    <td>27.8</td>
    <td>48.0</td>
    <td>58.7</td>
    <td><strong>84.4</strong></td>
    <td>94.0</td>
    <td>57.8 ± 1.1</td>
    </tr>
    <tr>
    <td>Gemini Flash 2<br>(Anchored)</td>
    <td>54.5</td>
    <td>56.1</td>
    <td>72.1</td>
    <td>34.2</td>
    <td>64.7</td>
    <td>61.5</td>
    <td>71.5</td>
    <td>95.6</td>
    <td>63.8 ± 1.2</td>
    </tr>
    <tr>
    <td>Qwen 2 VL<br>(No Anchor)</td>
    <td>19.7</td>
    <td>31.7</td>
    <td>24.2</td>
    <td>17.1</td>
    <td>88.9</td>
    <td>8.3</td>
    <td>6.8</td>
    <td>55.5</td>
    <td>31.5 ± 0.9</td>
    </tr>
    <tr>
    <td>Qwen 2.5 VL<br>(No Anchor)</td>
    <td>63.1</td>
    <td>65.7</td>
    <td>67.3</td>
    <td>38.6</td>
    <td>73.6</td>
    <td>68.3</td>
    <td>49.1</td>
    <td>98.3</td>
    <td>65.5 ± 1.2</td>
    </tr>
    <tr>
    <td>olmOCR v0.1.75<br>(No Anchor)</td>
    <td>71.5</td>
    <td>71.4</td>
    <td>71.4</td>
    <td><strong>42.8</strong></td>
    <td>94.1</td>
    <td>77.7</td>
    <td>71.0</td>
    <td>97.8</td>
    <td>74.7 ± 1.1</td>
    </tr>
    <tr>
    <td>olmOCR v0.1.75<br>(Anchored)</td>
    <td>74.9</td>
    <td>71.2</td>
    <td>71.0</td>
    <td>42.2</td>
    <td>94.5</td>
    <td><strong>78.3</strong></td>
    <td>73.3</td>
    <td>98.3</td>
    <td>75.5 ± 1.0</td>
    </tr>
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
</details>

## Supported Hardware:

Tested on various GPUs (3090, 4090, A6000, H800, A100, 4060 with 8GB VRAM). Community contributions have extended support to additional GPUs