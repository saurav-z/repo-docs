<div align="center">
  <h1>MonkeyOCR: Revolutionizing Document Parsing with SRR Triplet Paradigm</h1>
  <p><em>Unlock unparalleled document understanding with MonkeyOCR, a cutting-edge model designed for superior structure recognition and relation extraction.</em></p>
  
  [![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
  [![HuggingFace Weights](https://img.shields.io/badge/HuggingFace%20Weights-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR)
  [![GitHub Issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
  [![GitHub Closed Issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
  [![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
  [![GitHub Views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)
</div>

**[View the original repository on GitHub](https://github.com/Yuliang-Liu/MonkeyOCR)**

MonkeyOCR employs a Structure-Recognition-Relation (SRR) triplet paradigm, offering a streamlined and efficient approach to document parsing. It excels in understanding and extracting information from complex documents, outperforming many existing methods.

**Key Features:**

*   **Superior Performance:** MonkeyOCR achieves significant improvements across various document types, including formulas and tables, outperforming pipeline-based methods and end-to-end models.
*   **Fast Processing:** Processes multi-page documents at an impressive speed, surpassing competitors in efficiency.
*   **Comprehensive Output:** Generates processed Markdown files, layout results, and detailed intermediate block results for in-depth analysis.
*   **Easy Deployment:** Supports local installation with comprehensive documentation, along with Docker and FastAPI deployment options.
*   **Integration with Hugging Face and ModelScope:**  Seamlessly download and utilize the model weights through Hugging Face and ModelScope.
*   **Demo Available:** Interact with the model directly via a user-friendly Gradio demo.

**Highlighted Advantages:**

*   **Enhanced Accuracy:** Achieves an average improvement of 5.1% across various document types, with significant gains in formula and table processing.
*   **Competitive Speed:** Faster document parsing than models like MinerU and Qwen2.5 VL-7B.
*   **Strong Comparative Results**:  MonkeyOCR-3B leads in many benchmarks compared to various expert VLMs.

**Getting Started:**

Follow these steps to get MonkeyOCR running:

1.  **Installation:** Set up your environment using the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda.md#install-with-cuda-support).
2.  **Download Model Weights:** Choose from Hugging Face or ModelScope to download model weights:
    ```bash
    # Hugging Face
    pip install huggingface_hub
    python tools/download_model.py
    ```
    ```bash
    # ModelScope
    pip install modelscope
    python tools/download_model.py -t modelscope
    ```
3.  **Inference:** Parse documents using simple commands:

    ```bash
    # End-to-end parsing
    python parse.py input_path

    # Single-task recognition (outputs markdown only)
    python parse.py input_path -t text/formula/table

    # Specify output directory and model config file
    python parse.py input_path -o ./output -c config.yaml

    # Parse images in input_path(a dir) in groups with specific group size
    python parse.py input_path -g 20

    # Parse a PDF and split results by pages
    python parse.py your.pdf -s
    ```

4.  **Explore Outputs:** Analyze the processed Markdown (`.md`), layout results (`_layout.pdf`), and intermediate block results (`_middle.json`).
5.  **Run the Demo:** Launch the Gradio demo for a hands-on experience:
    ```bash
    python demo/demo_gradio.py
    ```

**[Explore the Gradio Demo](http://vlrlabmonkey.xyz:7685)**

**Docker Deployment:**
Effortlessly deploy with Docker Compose. Follow the steps in the original README to set up your Docker environment and get started with the demo or FastAPI.

**Benchmark Results:**

[Refer to the original README for detailed end-to-end evaluation and text recognition performance on OmniDocBench.]

**Quantization:**
Utilize AWQ for model quantization. Find the instructions within the [Quantization guide](docs/Quantization.md).

**Citing MonkeyOCR:**
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

**Acknowledgments:**

[Refer to the original README for a comprehensive list of acknowledgements.]

**Alternative Models to Explore:**

[Refer to the original README for the models to explore.]

**Copyright:**

[Refer to the original README for copyright.]