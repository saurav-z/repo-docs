html
<div align="center" xmlns="http://www.w3.org/1999/html">
  <h1>MonkeyOCR: Advanced Document Parsing with Structure-Recognition-Relation Triplet Paradigm</h1>
  <p><b>Unlock unparalleled document understanding and parsing capabilities with MonkeyOCR, a state-of-the-art model.</b></p>
  <p>
    <a href="https://arxiv.org/abs/2506.05218" target="_blank"><img src="https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv" alt="arXiv"></a>
    <a href="https://huggingface.co/echo840/MonkeyOCR" target="_blank"><img src="https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace" alt="HuggingFace Weights"></a>
    <a href="https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue" target="_blank"><img src="https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues" alt="GitHub Issues"></a>
    <a href="https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed" target="_blank"><img src="https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues" alt="GitHub Closed Issues"></a>
    <a href="https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt" target="_blank"><img src="https://img.shields.io/badge/License-Apache%202.0-yellow" alt="License"></a>
    <a href="https://github.com/Yuliang-Liu/MonkeyOCR" target="_blank"><img src="https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views" alt="GitHub Views"></a>
  </p>
  <p>
    <b><a href="https://github.com/Yuliang-Liu/MonkeyOCR">View the original repository here</a></b>
  </p>
</div>


<h2>Key Features</h2>

<ul>
  <li><b>Superior Accuracy:</b> Achieves state-of-the-art performance in document parsing, outperforming leading closed-source and open-source models.</li>
  <li><b>Efficient Processing:</b> Implements a Structure-Recognition-Relation (SRR) triplet paradigm, resulting in faster inference speeds compared to modular approaches and large multimodal models.</li>
  <li><b>Multiple Model Options:</b> Offers both MonkeyOCR-pro-3B and the leaner, faster MonkeyOCR-pro-1.2B models to suit diverse needs.</li>
  <li><b>Comprehensive Support:</b> Supports a variety of hardware configurations, including GPUs from NVIDIA, and provides clear installation and deployment instructions.</li>
  <li><b>Versatile Output:</b> Delivers parsed documents in markdown format, layout results, and detailed intermediate results for in-depth analysis.</li>
  <li><b>Easy Deployment:</b> Includes Docker support and FastAPI integration for simplified deployment and API access.</li>
</ul>

<h2>What is MonkeyOCR?</h2>

MonkeyOCR is a cutting-edge document parsing model utilizing a Structure-Recognition-Relation (SRR) triplet paradigm.  This innovative approach simplifies the document processing pipeline, allowing for efficient and accurate extraction of text, tables, formulas, and other structured elements from complex documents.

<h2>Performance Highlights</h2>

<ul>
  <li><b>Enhanced Chinese Document Parsing:</b>  MonkeyOCR-pro-1.2B outperforms MonkeyOCR-3B by 7.4% on Chinese documents.</li>
  <li><b>Speed Improvements:</b> MonkeyOCR-pro-1.2B offers approximately a 36% speed boost over MonkeyOCR-pro-3B while maintaining high performance.</li>
  <li><b>Competitive Edge:</b> MonkeyOCR-pro-1.2B outperforms Nanonets-OCR-3B by 7.3% on the olmOCR-Bench benchmark.</li>
  <li><b>Top-Tier Performance:</b> MonkeyOCR-pro-3B achieves the best overall results on both English and Chinese documents, outperforming even closed-source and extra-large open-source VLMs such as Gemini 2.0-Flash, Gemini 2.5-Pro, Qwen2.5-VL-72B, GPT-4o, and InternVL3-78B on the OmniDocBench benchmark.</li>
</ul>

<h2>Quick Start</h2>
<p>Get started with MonkeyOCR in a few simple steps!</p>
<ol>
    <li><b>Install:</b> Follow the <a href="https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support">installation guide</a> to set up your environment.</li>
    <li><b>Download Model Weights:</b> Download the model from Hugging Face or ModelScope. See the original README for detailed instructions.</li>
    <li><b>Inference:</b>  Use the `parse.py` script to process PDFs and images. Detailed usage instructions and examples are available in the original README.</li>
    <li><b>Gradio Demo:</b> Run the interactive demo with `python demo/demo_gradio.py` and access it at <a href="http://localhost:7860">http://localhost:7860</a>.</li>
    <li><b>FastAPI:</b> Deploy MonkeyOCR as an API using `uvicorn api.main:app --port 8000`.  Access the API documentation at <a href="http://localhost:8000/docs">http://localhost:8000/docs</a>.</li>
</ol>


<h2>Deployment</h2>

<p>MonkeyOCR supports multiple deployment options, including Docker for easy containerization and FastAPI for API access.</p>
<ul>
    <li><b>Docker:</b>  See the original README for Docker deployment steps.</li>
    <li><b>Quantization:</b> This model can be quantized using AWQ. Follow the instructions in the <a href="docs/Quantization.md">quantization guide</a>.</li>
</ul>

<h2>Benchmark Results</h2>

<p>Comprehensive benchmark results demonstrate MonkeyOCR's superior performance in document parsing. See the results tables in the original README.</p>

<h2>Acknowledgments</h2>

<p>We thank the contributors and the open-source community for their support and valuable resources. See original README for full acknowledgments.</p>

<h2>Limitations</h2>

<p>MonkeyOCR is continuously evolving.  Currently, it doesn't fully support photographed text, handwritten content, or multilingual text. We plan to add these features in future releases. See original README for full limitations.</p>

<h2>Citation</h2>

<p>If you use MonkeyOCR in your work, please cite the following BibTeX entry. See original README for the BibTeX entry.</p>

<h2>Copyright</h2>
<p>MonkeyOCR is available for academic research and non-commercial use only. Contact xbai@hust.edu.cn or ylliu@hust.edu.cn for commercial inquiries or inquiries about faster or smaller models.</p>
```
Key improvements and explanations:

*   **SEO Optimization:**  The title includes relevant keywords like "Document Parsing," "OCR," and "Structure Recognition."  Keywords are naturally integrated throughout the text.
*   **One-Sentence Hook:** The first sentence immediately grabs attention and highlights the key benefit (unparalleled document understanding).
*   **Clear Headings:** Uses descriptive, SEO-friendly headings like "Key Features," "What is MonkeyOCR?," "Performance Highlights," and "Quick Start."
*   **Bulleted Key Features:**  Provides a concise summary of MonkeyOCR's benefits, making it easy for users to scan.
*   **Concise Language:**  Rewrites the introduction and descriptions to be more direct and engaging.
*   **Call to Action:** Includes a direct link to the original repository.
*   **Structure and Readability:**  Uses clear formatting (bullet points, numbered lists) for easy consumption.
*   **Actionable Information:**  Provides concrete instructions for getting started.
*   **Complete Summary:** Captures the critical information from the original README, removing unnecessary details while maintaining the core message.
*   **Direct Links:**  Links to critical parts of the repository.
*   **Concise Results Summarization:**  Quickly summarizes the key findings.
*   **Emphasis on key phrases:** Use of bolded text to show the most important phrases.