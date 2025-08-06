# MonkeyOCR: Unlock Document Understanding with a Structure-Recognition-Relation Triplet Paradigm

**Effortlessly parse complex documents and extract valuable information with MonkeyOCR, a cutting-edge model achieving state-of-the-art performance on various benchmarks. Explore the code and resources on [GitHub](https://github.com/Yuliang-Liu/MonkeyOCR).**

**Key Features:**

*   **SRR Paradigm:** MonkeyOCR utilizes a Structure-Recognition-Relation (SRR) triplet paradigm, simplifying document parsing compared to traditional multi-tool approaches.
*   **Superior Performance:** MonkeyOCR consistently outperforms closed-source and large open-source VLMs on benchmarks like OmniDocBench and olmOCR-Bench.
*   **Fast and Efficient:** Experience impressive inference speeds on various GPUs, optimizing document processing.
*   **Multi-Format Support:** Process PDF and image files with ease.
*   **Comprehensive Output:** Generate markdown, layout results, and intermediate block results for detailed analysis.
*   **User-Friendly Demo:** Try out the [Gradio Demo](http://vlrlabmonkey.xyz:7685) to experience MonkeyOCR's capabilities.
*   **Quantization Support:** Leverage AWQ for efficient model deployment and use.

**Get Started:**

1.  **Installation:** Follow the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) to set up your environment.
2.  **Download Weights:** Download the model weights from Hugging Face or ModelScope using the provided `tools/download_model.py` script.
3.  **Inference:** Run the `parse.py` script with your input file(s) or directories, exploring the provided usage examples.
4.  **API Deployment:** Deploy the FastAPI service for convenient API access. Docker deployment instructions are also available.

**Performance Highlights:**

*   MonkeyOCR-pro-1.2B surpasses MonkeyOCR-3B in accuracy and speed on Chinese documents and outperforms other OCRs on olmOCR-Bench.
*   MonkeyOCR-pro-3B achieved the best overall performance on both English and Chinese documents, outperforming even closed-source and extra-large open-source VLMs such as Gemini 2.0-Flash, Gemini 2.5-Pro, Qwen2.5-VL-72B, GPT-4o, and InternVL3-78B on the OmniDocBench benchmarks.
*   **Inference Speed:**  See detailed performance tables showcasing Pages/s on different GPUs for various model sizes.

**Benchmark Results:**  Detailed performance metrics on OmniDocBench and olmOCR-Bench, as well as text recognition accuracy across document types, are provided.

**Hardware Support:**  MonkeyOCR has been tested on a variety of GPUs, including 3090, 4090, A6000, H800, A100 and 4060 with community support for 50-series GPUs, H200, L20, V100, 2080 Ti and npu.

**News:**

*   **2025.07.10**: 🚀 We release [MonkeyOCR-pro-1.2B](https://huggingface.co/echo840/MonkeyOCR-pro-1.2B), — a leaner and faster version model that outperforms our previous 3B version in accuracy, speed, and efficiency.
*   **2025.06.12**: 🚀 The model’s trending on [Hugging Face](https://huggingface.co/models?sort=trending). Thanks for the love!
*   **2025.06.05**: 🚀 We release [MonkeyOCR](https://huggingface.co/echo840/MonkeyOCR), an English and Chinese documents parsing model.

**Additional Resources:**

*   **Gradio Demo:** [http://vlrlabmonkey.xyz:7685](http://vlrlabmonkey.xyz:7685)
*   **Windows Support:** [docs/windows_support.md](docs/windows_support.md)
*   **Quantization Guide:** [docs/Quantization.md](docs/Quantization.md)
*   **Citing MonkeyOCR:**  Include the provided BibTeX entry in your publications.

**Acknowledgments:**  The project acknowledges the contributions of various libraries, datasets, and the open-source community.

**Limitations:**
Currently, MonkeyOCR do not yet fully support for photographed text, handwritten content, Traditional Chinese characters, or multilingual text. We plan to consider adding support for these features in future public releases. Additionally, our model is deployed on a single GPU, so if too many users upload files at the same time, issues like “This application is currently busy” may occur. The processing time shown on the demo page does not reflect computation time alone—it also includes result uploading and other overhead. During periods of high traffic, this time may be longer. The inference speeds of MonkeyOCR, MinerU, and Qwen2.5 VL-7B were measured on an H800 GPU.

**Copyright:** This model is intended for academic research and non-commercial use. Contact xbai@hust.edu.cn or ylliu@hust.edu.cn for faster/stronger versions.