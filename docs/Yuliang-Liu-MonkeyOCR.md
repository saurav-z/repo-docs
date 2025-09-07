html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MonkeyOCR: State-of-the-Art Document Parsing with Structure-Recognition-Relation Triplet Paradigm</title>
    <meta name="description" content="MonkeyOCR is a cutting-edge document parsing solution leveraging a Structure-Recognition-Relation (SRR) triplet paradigm, outperforming existing methods with speed and accuracy.">
    <meta name="keywords" content="OCR, document parsing, structure recognition, relation extraction, deep learning, AI, text recognition, table extraction, formula extraction, PDF parsing, image parsing">
    <!-- Add more SEO-friendly meta tags as needed -->
</head>
<body>

<div align="center">
    <h1>MonkeyOCR: Revolutionizing Document Parsing with SRR</h1>
    <p><em>Unlock unparalleled document processing capabilities with MonkeyOCR, a cutting-edge solution built on a Structure-Recognition-Relation (SRR) triplet paradigm.</em></p>

    <p>
        <a href="https://arxiv.org/abs/2506.05218"><img src="https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv" alt="arXiv"></a>
        <a href="https://huggingface.co/echo840/MonkeyOCR"><img src="https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace" alt="HuggingFace Weights"></a>
        <a href="https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue"><img src="https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues" alt="GitHub issues"></a>
        <a href="https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed"><img src="https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues" alt="GitHub closed issues"></a>
        <a href="https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt"><img src="https://img.shields.io/badge/License-Apache%202.0-yellow" alt="License"></a>
        <a href="https://github.com/Yuliang-Liu/MonkeyOCR"><img src="https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views" alt="GitHub views"></a>
    </p>
</div>

<hr>

<h2>Key Features of MonkeyOCR</h2>
<ul>
    <li><strong>Superior Performance:</strong> MonkeyOCR-pro-1.2B achieves state-of-the-art results, outperforming leading closed-source and open-source models, especially on Chinese documents.</li>
    <li><strong>Fast and Efficient:</strong> Experience significant speed improvements with MonkeyOCR-pro-1.2B (up to 36% faster) while maintaining exceptional accuracy.</li>
    <li><strong>SRR Triplet Paradigm:</strong> Our innovative Structure-Recognition-Relation approach simplifies document parsing, enhancing efficiency.</li>
    <li><strong>Comprehensive Support:</strong> Parses text, formulas, and tables with high accuracy.</li>
    <li><strong>Versatile Compatibility:</strong> Works seamlessly with various GPUs (including 3090, 4090, A6000, H800, A100, and even the 4060 with 8GB VRAM) and supports multiple file formats (PDF, images).</li>
    <li><strong>Easy Deployment:</strong>  Offers multiple deployment options: local installation, Gradio demo, and FastAPI.</li>
    <li><strong>Community Supported:</strong> The model is compatible with various hardware, and offers a strong open-source community.</li>
</ul>

<hr>

<h2><a href="https://github.com/Yuliang-Liu/MonkeyOCR">Get Started with MonkeyOCR</a></h2>

<hr>

<h2>Performance Highlights</h2>

<h3>Comparison with Large Language Models</h3>
<img src="https://v1.ax1x.com/2025/07/15/EKhkhY.png" alt="Performance Comparison" border="0">
<br>
<h3>Inference Speed</h3>
<p>See the tables below for pages per second performance on different GPUs:</p>

<h4>Inference Speed (Pages/s) on Different GPUs</h4>

<details>
    <summary>MonkeyOCR-pro-3B</summary>
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
        </tbody>
    </table>
</details>

<details>
    <summary>MonkeyOCR-pro-1.2B</summary>
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
</details>

<h4>VLM OCR Speed (Pages/s) on Different GPUs</h4>

<details>
    <summary>MonkeyOCR-pro-3B</summary>
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
        </tbody>
    </table>
</details>

<details>
    <summary>MonkeyOCR-pro-1.2B</summary>
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
</details>

<hr>

<h2>Quick Start</h2>

<h3>1. Installation</h3>
<p>Follow the <a href="https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support">installation guide</a>.</p>

<h3>2. Download Model Weights</h3>
<p>Download from Hugging Face or ModelScope:</p>
<pre>
pip install huggingface_hub
python tools/download_model.py -n MonkeyOCR-pro-3B  # or MonkeyOCR
</pre>
<pre>
pip install modelscope
python tools/download_model.py -t modelscope -n MonkeyOCR-pro-3B  # or MonkeyOCR
</pre>

<h3>3. Inference</h3>

<pre>
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
</pre>

<details>
    <summary>More Usage Examples</summary>
    <pre>
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
    </pre>
</details>

<details>
    <summary>Output Results</summary>
    <p>MonkeyOCR generates:</p>
    <ul>
        <li>Processed Markdown File (your.md)</li>
        <li>Layout Results (your_layout.pdf)</li>
        <li>Intermediate Block Results (your_middle.json)</li>
    </ul>
</details>

<h3>4. Gradio Demo</h3>
<pre>
python demo/demo_gradio.py
</pre>
<p>Access the demo at http://localhost:7860.</p>

<h3>5. Fast API</h3>
<pre>
uvicorn api.main:app --port 8000
</pre>
<p>API documentation: http://localhost:8000/docs. Consider using vllm_async for API concurrency.</p>

<hr>

<h2>Docker Deployment</h2>
<p>See the detailed instructions on the <a href="https://github.com/Yuliang-Liu/MonkeyOCR">main GitHub repository</a>.</p>

<h2>Windows Support</h2>
<p>See the <a href="docs/windows_support.md">Windows support guide</a>.</p>

<h2>Quantization</h2>
<p>Quantize the model using AWQ via the <a href="docs/Quantization.md">quantization guide</a>.</p>

<hr>

<h2>Benchmark Results</h2>

<p>Comprehensive benchmark results on OmniDocBench and olmOCR-bench, and text recognition performance across 9 PDF types, are detailed in the main repository.</p>

<p><strong>OmniDocBench Results:</strong></p>
<details>
    <summary>The end-to-end evaluation results of different tasks.</summary>
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
            <tr><td>... (Table content - Refer to the original README) ...</td></tr>
        </tbody>
    </table>
</details>

<details>
    <summary>The end-to-end text recognition performance across 9 PDF page types.</summary>
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
            <tr><td>... (Table content - Refer to the original README) ...</td></tr>
        </tbody>
    </table>
</details>

<details>
    <summary>The evaluation results of olmOCR-bench.</summary>
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
            <tr><td>... (Table content - Refer to the original README) ...</td></tr>
        </tbody>
    </table>
</details>

<hr>

<h2>Visualization Demo</h2>
<p>Experience MonkeyOCR firsthand with our interactive demo: <a href="http://vlrlabmonkey.xyz:7685">http://vlrlabmonkey.xyz:7685</a></p>
<p><em>Upload a PDF or image and let MonkeyOCR parse it!</em></p>
<p align="center">
    <img src="asserts/Visualization.GIF?raw=true" width="600" alt="MonkeyOCR Demo">
</p>

<hr>

<h2>Citing MonkeyOCR</h2>
<p>If you use MonkeyOCR in your research, please cite us:</p>
<pre>
@misc{li2025monkeyocrdocumentparsingstructurerecognitionrelation,
      title={MonkeyOCR: Document Parsing with a Structure-Recognition-Relation Triplet Paradigm},
      author={Zhang Li and Yuliang Liu and Qiang Liu and Zhiyin Ma and Ziyang Zhang and Shuo Zhang and Zidun Guo and Jiarui Zhang and Xinyu Wang and Xiang Bai},
      year={2025},
      eprint={2506.05218},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.05218},
}
</pre>

<hr>

<h2>Acknowledgments</h2>
<p>We thank the contributors and resources listed in the original README (MinerU, DocLayout-YOLO, etc.) for their valuable contributions.</p>

<h2>Limitations</h2>
<p>MonkeyOCR currently has limitations. For details, consult the original README.</p>

<h2>Copyright</h2>
<p>MonkeyOCR is intended for academic and non-commercial use. Contact us (xbai@hust.edu.cn or ylliu@hust.edu.cn) for inquiries regarding faster or stronger models.</p>

</body>
</html>