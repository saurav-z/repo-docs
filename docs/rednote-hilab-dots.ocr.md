<div align="center">

<p align="center">
    <img src="https://raw.githubusercontent.com/rednote-hilab/dots.ocr/master/assets/logo.png" width="300" alt="dots.ocr Logo"/>
<p>

<h1 align="center">
dots.ocr: Revolutionizing Document Parsing with a Single Vision-Language Model
</h1>

[![Blog](https://img.shields.io/badge/Blog-View_on_GitHub-333.svg?logo=github)](https://github.com/rednote-hilab/dots.ocr/blob/master/assets/blog.md)
[![HuggingFace](https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace)](https://huggingface.co/rednote-hilab/dots.ocr)

<div align="center">
  <a href="https://dotsocr.xiaohongshu.com" target="_blank" rel="noopener noreferrer">🖥️ Live Demo</a> |
  <a href="https://raw.githubusercontent.com/rednote-hilab/dots.ocr/master/assets/wechat.jpg" target="_blank" rel="noopener noreferrer">💬 WeChat</a> |
  <a href="https://www.xiaohongshu.com/user/profile/683ffe42000000001d021a4c" target="_blank" rel="noopener noreferrer">📕 rednote</a>
</div>

</div>

## **Unlock the Power of Dots.OCR: Your All-in-One Document Parser**

dots.ocr is a cutting-edge, multilingual document parser that leverages a single vision-language model to detect layout, recognize content, and maintain accurate reading order, all while delivering impressive performance. [Explore the original repository](https://github.com/rednote-hilab/dots.ocr) to learn more.

**Key Features:**

*   🚀 **State-of-the-Art Performance:** Achieves SOTA results on OmniDocBench for text, tables, and reading order. Formula recognition is comparable to much larger models.
*   🌍 **Multilingual Support:** Excels in parsing documents across numerous languages, including low-resource languages.
*   ⚙️ **Simplified Architecture:**  A unified vision-language model eliminates the need for complex, multi-model pipelines, streamlining your workflow.
*   ⚡ **Fast & Efficient:** Built on a compact 1.7B LLM, dots.ocr provides rapid inference speeds.

## Performance Highlights

**dots.ocr consistently outperforms traditional methods and even large-scale models in key areas, delivering superior accuracy and efficiency.**

### Performance Comparison: dots.ocr vs. Competing Models

<img src="https://raw.githubusercontent.com/rednote-hilab/dots.ocr/master/assets/chart.png" border="0" alt="Performance Comparison Chart" />

>   **Note:** The EN and ZH metrics represent end-to-end evaluation results from OmniDocBench, while the Multilingual metric is from dots.ocr-bench.

##  Key Benchmark Results

### 1. OmniDocBench
**dots.ocr demonstrates superior performance across multiple document parsing tasks.**  See the detailed benchmark results below.

#### The end-to-end evaluation results of different tasks.

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
<td>0.454</td>
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
<td>0.206</td>
<td>0.058</td>
<td>0.088</td>
<td>0.295</td>
<td>0.535</td>
<td>-</td>
<td>-</td>
<td>0.159</td>
<td>0.109</td>
<td>0.069</td>
<td>0.091</td>
</tr>
<tr>
<td rowspan="9"><strong>Expert<br>VLMs</strong></td>
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
<td>0.047</td>
<td>0.109</td>
<td>0.297</td>
<td>0.536</td>
<td>82.5</td>
<td>79.0</td>
<td>0.141</td>
<td>0.195</td>
<td>0.069<</td>
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
<td>MonkeyOCR-pro-3B</td>
<td>0.138</td>
<td>0.206</td>
<td>0.067</td>
<td>0.107</td>
<td><strong>0.246</strong></td>
<td>0.421</td>
<td>81.5</td>
<td>87.5</td>
<td>0.139</td>
<td>0.111</td>
<td>0.100</td>
<td>0.185</td>
</tr>
<tr>
<td rowspan="5"><strong>General<br>VLMs</strong></td>
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
      <td>Qwen2-VL-72B</td>
      <td>0.252</td>
      <td>0.327</td>
      <td>0.096</td>
      <td>0.218</td>
      <td>0.404</td>
      <td>0.487</td>
      <td>76.8</td>
      <td>76.4</td>
      <td>0.387</td>
      <td>0.408</td>
      <td>0.119</td>
      <td>0.193</td>
    </tr>
    <tr>
      <td>Qwen2.5-VL-72B</td>
      <td>0.214</td>
      <td>0.261</td>
      <td>0.092</td>
      <td>0.18</td>
      <td>0.315</td>
      <td>0.434</td>
      <td>82.9</td>
      <td>83.9</td>
      <td>0.341</td>
      <td>0.262</td>
      <td>0.106</td>
      <td>0.168</td>
    </tr>
    <tr>
      <td>Gemini2.5-Pro</td>
      <td>0.148</td>
      <td>0.212</td>
      <td>0.055</td>
      <td>0.168</td>
      <td>0.356</td>
      <td>0.439</td>
      <td>85.8</td>
      <td>86.4</td>
      <td>0.13</td>
      <td>0.119</td>
      <td>0.049</td>
      <td>0.121</td>
    </tr>
    <tr>
      <td>doubao-1-5-thinking-vision-pro-250428</td>
      <td>0.140</td>
      <td>0.162</td>
      <td>0.043</td>
      <td>0.085</td>
      <td>0.295</td>
      <td><strong>0.384</strong></td>
      <td>83.3</td>
      <td><strong>89.3</strong></td>
      <td>0.165</td>
      <td><strong>0.085</strong></td>
      <td>0.058</td>
      <td>0.094</td>
    </tr>
<tr>
<td rowspan="1"><strong>Expert VLMs</strong></td>
<td><strong>dots.ocr</strong></td>
<td><strong>0.125</strong></td>
<td><strong>0.160</strong></td>
<td><strong>0.032</strong></td>
<td><strong>0.066</strong></td>
<td>0.329</td>
<td>0.416</td>
<td><strong>88.6</strong></td>
<td>89.0</td>
<td><strong>0.099</strong></td>
<td>0.092</td>
<td><strong>0.040</strong></td>
<td><strong>0.067</strong></td>
</tr>
<tr>
</tbody>
</table>

#### The end-to-end text recognition performance across 9 PDF page types.

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
<td rowspan="5"><strong>Expert<br>VLMs</strong></td>
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
<td>MonkeyOCR-pro-3B</td>
<td>0.084</td>
<td>0.129</td>
<td>0.060</td>
<td>0.090</td>
<td>0.107</td>
<td>0.073</td>
<td>0.050</td>
<td>0.171</td>
<td>0.107</td>
<td>0.100</td>
</tr>
<tr>
<td rowspan="4"><strong>General<br>VLMs</strong></td>
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
<td>0.053</td>
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
<td>0.056</td>
<td>0.107</td>
<td>0.109</td>
<td>0.129</td>
<td>0.100</td>
<td>0.159</td>
<td>0.150</td>
<td>0.681</td>
<td>0.188</td>
</tr>
<tr>
<td>doubao-1-5-thinking-vision-pro-250428</td>
<td>0.048</td>
<td>0.048</td>
<td>0.024</td>
<td><strong>0.062</strong></td>
<td>0.085</td>
<td>0.051</td>
<td>0.039</td>
<td><strong>0.096</strong></td>
<td>0.181</td>
<td>0.073</td>
</tr>
<tr>
<td rowspan="1"><strong>Expert VLMs</strong></td>
<td><strong>dots.ocr</strong></td>
<td><strong>0.031</strong></td>
<td><strong>0.047</strong></td>
<td><strong>0.011</strong></td>
<td>0.082</td>
<td><strong>0.079</strong></td>
<td><strong>0.028</strong></td>
<td><strong>0.029</strong></td>
<td>0.109</td>
<td><strong>0.056</strong></td>
<td><strong>0.055</strong></td>
</tr>

</tbody>
</table>

>   **Notes:** Metrics are sourced from MonkeyOCR, OmniDocBench, and internal evaluations.  We've removed page header and footer cells.

### 2.  dots.ocr-bench

**Our in-house benchmark, dots.ocr-bench, showcases the robust parsing capabilities of dots.ocr, which includes 1493 PDF images across 100 languages.**

#### The end-to-end evaluation results of different tasks.

<table>
<thead>
<tr>
<th rowspan="1"><strong>Methods</strong></th>
<th colspan="1"><strong>Overall<sup>Edit</sup>↓</strong></th>
<th colspan="1"><strong>Text<sup>Edit</sup>↓</strong></th>
<th colspan="1"><strong>Formula<sup>Edit</sup>↓</strong></th>
<th colspan="1"><strong>Table<sup>TEDS</sup>↑</strong></th>
<th colspan="1"><strong>Table<sup>Edit</sup>↓</strong></th>
<th colspan="1"><strong>Read Order<sup>Edit</sup>↓</strong></th>
</tr>
</thead>
<tbody>
<td>MonkeyOCR-3B</td>
<td>0.483</td>
<td>0.445</td>
<td>0.627</td>
<td>50.93</td>
<td>0.452</td>
<td>0.409</td>
</tr>
<tr>
<td>doubao-1-5-thinking-vision-pro-250428</td>
<td>0.291</td>
<td>0.226</td>
<td>0.440</td>
<td>71.2</td>
<td>0.260</td>
<td>0.238</td>
</tr>
<tr>
<td>doubao-1-6</td>
<td>0.299</td>
<td>0.270</td>
<td>0.417</td>
<td>71.0</td>
<td>0.258</td>
<td>0.253</td>
</tr>
<tr>
<td>Gemini2.5-Pro</td>
<td>0.251</td>
<td>0.163</td>
<td>0.402</td>
<td>77.1</td>
<td>0.236</td>
<td>0.202</td>
</tr>
<tr>
<td><strong>dots.ocr</strong> </td>
<td><strong>0.177</strong></td>
<td><strong>0.075</strong></td>
<td><strong>0.297</strong></td>
<td><strong>79.2</strong></td>
<td><strong>0.186</strong></td>
<td><strong>0.152</strong></td>
</tr>
</tbody>
</table>

>   **Notes:** We use the same metric calculation pipeline as OmniDocBench and have removed the page header and footer cells.

#### Layout Detection

<table>
<thead>
<tr>
<th rowspan="2"><strong>Method</strong></th>
<th colspan="5" style="text-align: center;"><strong>F1@IoU=.50:.05:.95↑</strong></th>
<th colspan="5" style="text-align: center;"><strong>F1@IoU=.50↑</strong></th>
</tr>
<tr>
<th>Overall</th>
<th>Text</th>
<th>Formula</th>
<th>Table</th>
<th>Picture</th>
<th>Overall</th>
<th>Text</th>
<th>Formula</th>
<th>Table</th>
<th>Picture</th>
</tr>
</thead>
<tbody>
<td>DocLayout-YOLO-DocStructBench</td>
<td>0.733</td>
<td>0.694</td>
<td>0.480</td>
<td>0.803</td>
<td>0.619</td>
<td>0.806</td>
<td>0.779</td>
<td>0.620</td>
<td>0.858</td>
<td>0.678</td>
</tr>
<tr>
<td>dots.ocr-parse all</td>
<td>0.831</td>
<td>0.801</td>
<td>0.654</td>
<td>0.838</td>
<td>0.748</td>
<td>0.922</td>
<td>0.909</td>
<td>0.770</td>
<td>0.888</td>
<td>0.831</td>
</tr>
<tr>
<td> <strong>dots.ocr-detection only</strong> </td>
<td><strong>0.845</strong></td>
<td><strong>0.816</strong></td>
<td><strong>0.716</strong></td>
<td><strong>0.875</strong></td>
<td><strong>0.765</strong></td>
<td><strong>0.930</strong></td>
<td><strong>0.917</strong></td>
<td><strong>0.832</strong></td>
<td><strong>0.918</strong></td>
<td><strong>0.843</strong></td>
</tr>
</tbody>
</table>

>   **Notes:** prompt\_layout\_all\_en for parse all and prompt\_layout\_only\_en for detection only. Refer to [prompts](https://github.com/rednote-hilab/dots.ocr/blob/master/dots_ocr/utils/prompts.py) for details.

### 3. olmOCR-bench.

**dots.ocr shows a leading performance against many powerful baselines.**

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
<td>99.1</td>
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
<td>77.7</td>
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
<td>78.3</td>
<td>73.3</td>
<td>98.3</td>
<td>75.5 ± 1.0</td>
</tr>
<tr>
<td>MonkeyOCR-pro-3B</td>
<td><strong>83.8</strong></td>
<td>68.8</td>
<td>74.6</td>
<td>36.1</td>
<td>91.2</td>
<td>76.6</td>
<td>80.1</td>
<td>95.3</td>
<td>75.8 ± 1.0</td>
</tr>
<tr>
<td><strong>dots.ocr</strong></td>
<td>82.1</td>
<td>64.2</td>
<td><strong>88.3</strong></td>
<td>40.9</td>
<td>94.1</td>
<td><strong>82.4</strong></td>
<td>81.2</td>
<td><strong>99.5</strong></td>
<td><strong>79.1 ± 1.0</strong></td>
</tr>
</tbody>
</table>

>   **Note:** The metrics are from MonkeyOCR, olmocr, and our internal evaluations. We've removed page header and footer cells.

## Quick Start

Follow these steps to get started with dots.ocr:

### 1. Installation

```bash
conda create -n dots_ocr python=3.12
conda activate dots_ocr

git clone https://github.com/rednote-hilab/dots.ocr.git
cd dots.ocr

# Install pytorch, see https://pytorch.org/get-started/previous-versions/ for your cuda version
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

**Troubleshooting:**

If you encounter installation issues, try our Docker Image, available at [https://hub.docker.com/r/rednotehilab/dots.ocr](https://hub.docker.com/r/rednotehilab/dots.ocr) for an easier setup:

```bash
git clone https://github.com/rednote-hilab/dots.ocr.git
cd dots.ocr
pip install -e .
```

### 2. Download Model Weights

>   **Note:**  For the model save path, please use a directory name without periods (e.g., `DotsOCR` instead of `dots.ocr`) until our integration with Transformers is complete.

```bash
python3 tools/download_model.py

# with modelscope
python3 tools/download_model.py --type modelscope
```

### 3. Deployment

#### vLLM Inference (Recommended)

For optimal performance, we recommend using vLLM for deployment and inference. Our evaluations were conducted with vLLM version 0.9.1.

The [Docker Image](https://hub.docker.com/r/rednotehilab/dots.ocr) is based on the official vllm image. You can also follow [Dockerfile](https://github.com/rednote-hilab/dots.ocr/blob/master/docker/Dockerfile) to build the deployment environment by yourself.

```bash
# You need to register model to vllm at first
python3 tools/download_model.py
export hf_model_path=./weights/DotsOCR  # Path to your downloaded model weights, Please use a directory name without periods (e.g., `DotsOCR` instead of `dots.ocr`) for the model save path. This is a temporary workaround pending our integration with Transformers.
export PYTHONPATH=$(dirname "$hf_model_path"):$PYTHONPATH
sed -i '/^from vllm\.entrypoints\.cli\.main import main$/a\
from DotsOCR import modeling_dots_ocr_vllm' `which vllm`  # If you downloaded model weights by yourself, please replace `DotsOCR` by your model saved directory name, and remember to use a directory name without periods (e.g., `DotsOCR` instead of `dots.ocr`)

# launch vllm server
CUDA_VISIBLE_DEVICES=0 vllm serve ${hf_model_path} --tensor-parallel-size 1 --gpu-memory-utilization 0.95  --chat-template-content-format string --served-model-name model --trust-remote-code

# If you get a ModuleNotFoundError: No module named 'DotsOCR', please check the note above on the saved model directory name.

# vllm api demo
python3 ./demo/demo_vllm.py --prompt_mode prompt_layout_all_en
```

#### Hugging Face Inference

```bash
python3 demo/demo_hf.py
```

<details>
<summary><b>Hugging Face inference details