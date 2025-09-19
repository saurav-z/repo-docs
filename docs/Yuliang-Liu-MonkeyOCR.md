html
<!DOCTYPE html>
<html>
<head>
    <title>MonkeyOCR: Advanced Document Parsing with Structure Recognition - AI-Powered OCR</title>
    <meta name="description" content="MonkeyOCR is a cutting-edge document parsing system utilizing a Structure-Recognition-Relation (SRR) triplet paradigm.  Extract text, formulas, and tables from your documents with speed and accuracy.">
    <meta name="keywords" content="OCR, document parsing, AI, structure recognition, text extraction, formula extraction, table extraction, open source, MonkeyOCR">
    <style>
        /* Basic Styling - Customize as needed */
        body { font-family: sans-serif; line-height: 1.6; }
        h1, h2 { color: #333; }
        ul { list-style-type: disc; margin-left: 20px; }
        a { color: #007bff; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .key-feature { margin-bottom: 10px; }
        .table-responsive { overflow-x: auto; } /* For horizontal scrolling on tables */
    </style>
</head>
<body>

<div align="center">
    <h1 align="center">MonkeyOCR: Effortless Document Parsing with AI-Powered OCR</h1>
</div>

<p><b>Unlock unparalleled document understanding with MonkeyOCR, a revolutionary OCR solution leveraging a Structure-Recognition-Relation (SRR) triplet paradigm.</b> Visit the <a href="https://github.com/Yuliang-Liu/MonkeyOCR">original repository</a> for more details!</p>

<!-- Badges (Re-add these from the original, keeping it concise) -->
<div align="center">
    <a href="https://arxiv.org/abs/2506.05218"><img src="https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv" alt="arXiv"></a>
    <a href="https://huggingface.co/echo840/MonkeyOCR-pro-3B"><img src="https://img.shields.io/badge/HuggingFace-black.svg?logo=HuggingFace" alt="Hugging Face"></a>
    <a href="https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue"><img src="https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues" alt="GitHub Issues"></a>
    <a href="https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed"><img src="https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues" alt="GitHub Closed Issues"></a>
    <a href="https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt"><img src="https://img.shields.io/badge/License-Apache%202.0-yellow" alt="License"></a>
    <a href="https://github.com/Yuliang-Liu/MonkeyOCR"><img src="https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views" alt="GitHub Views"></a>
</div>

<h2>Key Features of MonkeyOCR</h2>
<ul>
    <li class="key-feature"><b>SRR Paradigm:</b> Simplifies document parsing with a Structure-Recognition-Relation triplet paradigm, avoiding the inefficiencies of large multimodal models.</li>
    <li class="key-feature"><b>Superior Performance:</b> MonkeyOCR-pro-1.2B outperforms comparable models on key benchmarks, including olmOCR-Bench and OmniDocBench, even outperforming closed-source and extra-large open-source VLMs.</li>
    <li class="key-feature"><b>Speed and Efficiency:</b> Achieve significant speed improvements (up to 36% faster) with MonkeyOCR-pro-1.2B compared to previous versions, without sacrificing accuracy.</li>
    <li class="key-feature"><b>Versatile Deployment:</b> Supports various hardware configurations, including NVIDIA GPUs (3090, 4090, A6000, H800, A100, and more) and is compatible with Docker and Windows environments.</li>
    <li class="key-feature"><b>User-Friendly:</b> Provides a Gradio demo and a FastAPI service for easy interaction and integration.</li>
</ul>

<h2>Benchmark Results</h2>

<p>MonkeyOCR demonstrates excellent performance across diverse document types, including books, slides, financial reports, textbooks, and academic papers.</p>

<h3>End-to-End Evaluation on OmniDocBench</h3>
<!-- Replace with a more concise summary or a link to the full table on the repo -->
<p>MonkeyOCR excels in end-to-end document parsing, showcasing strong performance across various tasks such as text, formula, and table extraction. Results indicate strong overall performance compared to other methods including VLMs.</p>
<p><b>View complete benchmark details on the <a href="https://github.com/Yuliang-Liu/MonkeyOCR">GitHub repository</a>.</b></p>

<h3>Text Recognition Performance Across PDF Page Types</h3>
<!-- Replace with a more concise summary or a link to the full table on the repo -->
<p>MonkeyOCR excels in text recognition. For detailed breakdown by PDF type please view the repository.</p>
<p><b>View complete benchmark details on the <a href="https://github.com/Yuliang-Liu/MonkeyOCR">GitHub repository</a>.</b></p>

<h3>olmOCR-Bench Results</h3>
<!-- Replace with a more concise summary or a link to the full table on the repo -->
<p>MonkeyOCR shines on the olmOCR-Bench, outperforming many competing OCR models.</p>
<p><b>View complete benchmark details on the <a href="https://github.com/Yuliang-Liu/MonkeyOCR">GitHub repository</a>.</b></p>

<h2>Getting Started</h2>

<p>Quickly integrate MonkeyOCR into your workflow.  See the <a href="https://github.com/Yuliang-Liu/MonkeyOCR#quick-start">Quick Start Guide</a> on the GitHub repository for detailed instructions, including installation, model downloads, and inference commands.</p>

<h2>Demo & Visualization</h2>

<p><b>Experience MonkeyOCR firsthand!</b>  Try our user-friendly demo at <a href="http://vlrlabmonkey.xyz:7685">http://vlrlabmonkey.xyz:7685</a>.  Simply upload your documents and let MonkeyOCR do the work.</p>

<h2>Additional Resources</h2>
<ul>
  <li><a href="https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support">Installation Guide</a></li>
  <li><a href="https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/windows_support.md">Windows Support</a></li>
  <li><a href="https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/Quantization.md">Quantization Guide</a></li>
</ul>

<h2>Citing MonkeyOCR</h2>

<p>If you use MonkeyOCR in your research, please cite the following:</p>
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

<h2>Acknowledgments</h2>
<p> [MinerU](https://github.com/opendatalab/MinerU), [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO), [PyMuPDF](https://github.com/pymupdf/PyMuPDF), [layoutreader](https://github.com/ppaanngggg/layoutreader), [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [LMDeploy](https://github.com/InternLM/lmdeploy), [PP-StructureV3](https://github.com/PaddlePaddle/PaddleOCR), [PP-DocLayout_plus-L](https://huggingface.co/PaddlePaddle/PP-DocLayout_plus-L), and [InternVL3](https://github.com/OpenGVLab/InternVL) for providing base code and models, as well as their contributions to this field. We also thank [M6Doc](https://github.com/HCIILAB/M6Doc), [DocLayNet](https://github.com/DS4SD/DocLayNet), [CDLA](https://github.com/buptlihang/CDLA), [D4LA](https://github.com/AlibabaResearch/AdvancedLiterateMachinery), [DocGenome](https://github.com/Alpha-Innovator/DocGenome), [PubTabNet](https://github.com/ibm-aur-nlp/PubTabNet), and [UniMER-1M](https://github.com/opendatalab/UniMERNet) for providing valuable datasets. We also thank everyone who contributed to this open-source effort.</p>

<h2>Limitations</h2>
<p>Currently, MonkeyOCR do not yet fully support for photographed text, handwritten content, Traditional Chinese characters, or multilingual text. We plan to consider adding support for these features in future public releases. Additionally, our model is deployed on a single GPU, so if too many users upload files at the same time, issues like “This application is currently busy” may occur. The processing time shown on the demo page does not reflect computation time alone—it also includes result uploading and other overhead. During periods of high traffic, this time may be longer. The inference speeds of MonkeyOCR, MinerU, and Qwen2.5 VL-7B were measured on an H800 GPU.</p>

<h2>Copyright</h2>
<p>Please don’t hesitate to share your valuable feedback — it’s a key motivation that drives us to continuously improve our framework. Note: Our model is intended for academic research and non-commercial use only. If you are interested in faster (smaller) or stronger one, please contact us at xbai@hust.edu.cn or ylliu@hust.edu.cn.</p>

</body>
</html>
```

Key improvements and explanations:

*   **Title and Meta Description:**  Added `title`, `description`, and `keywords` meta tags for SEO. The title includes the primary keyword "MonkeyOCR" and related terms. The description is a concise summary.  Keywords are added to help search engines understand the content.
*   **One-Sentence Hook:** The introductory paragraph is modified to be more engaging and benefit-driven.
*   **Clear Headings:**  Uses `<h2>` headings for sections to improve readability and SEO.  Organized the content more logically.
*   **Key Features (Bulleted):** Uses a bulleted list to clearly highlight the main advantages of MonkeyOCR. This is user-friendly and also helps with SEO (search engines like lists).  Each bullet point is concise and keyword-rich.
*   **Concise Benchmark Summaries:** Instead of pasting the entire table, provides a brief description of the key findings and a link back to the repository for details. This keeps the README cleaner and avoids redundancy, while still providing key information.
*   **Streamlined "Getting Started":**  Simplified the "Getting Started" section by focusing on the key steps.
*   **Clear Calls to Action:** Encourages users to try the demo and visit the repository.
*   **Organized Content:**  The content is broken down logically, making it easier for users to scan and find the information they need.
*   **HTML Structure:** Uses basic HTML for better formatting (headings, lists, etc.) and SEO.
*   **Revised limitations Section:** Added a revised limitation sections

This improved README is significantly better for SEO, easier to read, and provides a more compelling overview of MonkeyOCR's capabilities. It directs users to key information and encourages engagement.