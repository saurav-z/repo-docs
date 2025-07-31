# MonkeyOCR: Effortless Document Parsing with AI (Get Started!)

**MonkeyOCR is an innovative document parsing solution leveraging a Structure-Recognition-Relation (SRR) triplet paradigm, simplifying document processing and achieving state-of-the-art results. Explore the project on [GitHub](https://github.com/Yuliang-Liu/MonkeyOCR) to unlock the power of efficient document understanding!**

*   **Advanced Document Parsing:** MonkeyOCR excels at extracting text, formulas, tables, and more from diverse document types.
*   **High Performance:**  Achieve optimal parsing speeds, including impressive performance across different GPUs, as shown in benchmark results.
*   **Multi-Model Support:**  Choose from optimized MonkeyOCR-pro models and explore faster parsing with the 1.2B model, and other model options.
*   **User-Friendly Deployment:**  Easy setup with local installation, Hugging Face model downloads, and Docker support for convenient deployment.
*   **Real-World Application:**  Transform financial reports, academic papers, and more into structured, accessible formats.

## Key Features

*   **SRR Paradigm:**  Employs a Structure-Recognition-Relation triplet paradigm for efficient and modular document parsing.
*   **Superior Accuracy:**  MonkeyOCR-pro-1.2B surpasses the performance of the 3B version while offering a faster speed.
*   **High-Speed Inference:**  Demonstrates impressive inference speeds across various GPUs (3090, A6000, H800, 4090) and document sizes.
*   **Benchmark Results:**  Achieves state-of-the-art performance on OmniDocBench and competitive results on other benchmarks like olmOCR-Bench.
*   **Flexible Deployment:**  Supports local installation, Docker containers, and FastAPI for versatile use.
*   **Demo and Visualization:** Offers a user-friendly Gradio demo ([http://vlrlabmonkey.xyz:7685](http://vlrlabmonkey.xyz:7685)) for easy testing and understanding.
*   **Quantization support** offers AWQ quantization capabilities.

## Quick Start

### 1.  Install MonkeyOCR

   Follow the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) to set up your environment.

### 2. Download Model Weights

   Choose from the following options:

   *   **Hugging Face:**
       ```bash
       pip install huggingface_hub
       python tools/download_model.py -n MonkeyOCR  # or MonkeyOCR-pro-1.2B
       ```
   *   **ModelScope:**
       ```bash
       pip install modelscope
       python tools/download_model.py -t modelscope -n MonkeyOCR  # or MonkeyOCR-pro-1.2B
       ```

### 3. Inference

   Use the `parse.py` script to process PDF or image files:

   ```bash
   # Example usage
   python parse.py input.pdf  # Parse a single PDF
   ```

   Refer to the detailed usage examples in the original README for more options (splitting pages, single-task recognition, directory processing, etc.).

### 4. Run the Gradio Demo

   ```bash
   python demo/demo_gradio.py
   ```
   Access the demo at http://localhost:7860

### 5. Start the FastAPI Service

   ```bash
   uvicorn api.main:app --port 8000
   ```

   Access the API documentation at http://localhost:8000/docs.  For improved performance, consider using `lmdeploy_queue` or `vllm_queue` as the inference backend.

## Docker Deployment

1.  Navigate to the `docker` directory:

    ```bash
    cd docker
    ```

2.  Build the Docker image (and nvidia-docker2 setup if needed)

    ```bash
    bash env.sh
    docker compose build monkeyocr
    ```

    For 20/30/40 series GPUs please use the monkeyocr-fix build.

3.  Run the Gradio demo (port 7860):

    ```bash
    docker compose up monkeyocr-demo
    ```

    Or run a development environment:

    ```bash
    docker compose run --rm monkeyocr-dev
    ```

4. Run the FastAPI service (port 7861):

    ```bash
    docker compose up monkeyocr-api
    ```

## Windows Support

See the [windows support guide](docs/windows_support.md) for details.

## Quantization

This model can be quantized using AWQ. Follow the instructions in the [quantization guide](docs/Quantization.md).

## Benchmark Results

See the original README for detailed benchmark results.

##  Example Visualizations

*   **PDF and image parsing**
    <p align="center">
      <img src="asserts/Visualization.GIF?raw=true" width="600"/>
    </p>

*   **Example for formula document**
    <img src="https://v1.ax1x.com/2025/06/10/7jVLgB.jpg" alt="7jVLgB.jpg" border="0" />

*   **Example for table document**
    <img src="https://v1.ax1x.com/2025/06/11/7jcOaa.png" alt="7jcOaa.png" border="0" />

*   **Example for newspaper**
    <img src="https://v1.ax1x.com/2025/06/11/7jcP5V.png" alt="7jcP5V.png" border="0" />

*   **Example for financial report**
    <img src="https://v1.ax1x.com/2025/06/11/7jc10I.png" alt="7jc10I.png" border="0" />
    <img src="https://v1.ax1x.com/2025/06/11/7jcRCL.png" alt="7jcRCL.png" border="0" />


##  Citing MonkeyOCR

```bibtex
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

## Acknowledgments

Thank you to the referenced projects and contributors!

## Limitations

MonkeyOCR is continuously being improved and currently has limitations. Please refer to the original README.

## Copyright

Please share your feedback!  Our model is for academic research and non-commercial use. Contact xbai@hust.edu.cn or ylliu@hust.edu.cn for inquiries.