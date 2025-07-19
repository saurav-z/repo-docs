# MonkeyOCR: Extract Insights from Documents with State-of-the-Art Accuracy 🚀

[Original Repo](https://github.com/Yuliang-Liu/MonkeyOCR)

MonkeyOCR is a cutting-edge document parsing system utilizing a Structure-Recognition-Relation (SRR) triplet paradigm, delivering superior performance and efficiency for comprehensive document analysis.

**Key Features:**

*   **Advanced Document Understanding:** Analyzes diverse document types including financial reports, academic papers, and more.
*   **Superior Performance:** Outperforms leading models like Gemini 2.5-Pro and GPT-4o on the OmniDocBench benchmark.
*   **High Speed:** Offers significant speed improvements compared to previous versions, with up to 2.4x faster processing.
*   **Versatile Outputs:** Generates Markdown files, layout results (PDF), and detailed intermediate results (JSON).
*   **Easy Deployment:** Supports local installation, Hugging Face weights, and Docker deployment.

**Model Highlights:**

*   **MonkeyOCR-pro-1.2B:** A leaner and faster model that surpasses the 3B version in accuracy, speed, and efficiency.
*   **State-of-the-Art Results:** Achieves the best overall performance on both English and Chinese documents on the OmniDocBench.

**Key Improvements and Results:**

*   MonkeyOCR-pro-1.2B surpasses MonkeyOCR-3B by 7.4% on Chinese documents.
*   MonkeyOCR-pro-1.2B delivers approximately a 36% speed improvement over MonkeyOCR-pro-3B, with approximately 1.6% drop in performance.
*   On olmOCR-Bench, MonkeyOCR-pro-1.2B outperforms Nanonets-OCR-3B by 7.3%.
*   On OmniDocBench, MonkeyOCR-pro-3B achieves the best overall performance on both English and Chinese documents, outperforming even closed-source and extra-large open-source VLMs such as Gemini 2.0-Flash, Gemini 2.5-Pro, Qwen2.5-VL-72B, GPT-4o, and InternVL3-78B.

**Model Weights and Resources:**

*   **MonkeyOCR-pro-1.2B Weights:** [Hugging Face](https://huggingface.co/echo840/MonkeyOCR-pro-1.2B)
*   **MonkeyOCR-pro-3B Demo:** [Demo](http://vlrlabmonkey.xyz:7685/)
*   **Weights:** [Hugging Face](https://huggingface.co/echo840/MonkeyOCR/blob/main/Structure/layout_zh.pt)
*   **Weights:** [Hugging Face](https://huggingface.co/echo840/MonkeyOCR/blob/main/Structure/doclayout_yolo_docstructbench_imgsz1280_2501.pt)

**Performance Charts (Example):**

*   **Inference Speed (Pages/s):**
    <details>
    <summary>Expand Table</summary>

    | Model             | GPU   | 50 Pages | 100 Pages | 300 Pages | 500 Pages | 1000 Pages |
    | :---------------- | :---- | :------- | :-------- | :-------- | :-------- | :--------- |
    | MonkeyOCR-pro-3B  | 3090  | 0.705    | 0.680     | 0.711     | 0.700     | 0.724      |
    | MonkeyOCR-pro-3B  | A6000 | 0.885    | 0.860     | 0.915     | 0.892     | 0.934      |
    | MonkeyOCR-pro-3B  | H800  | 1.371    | 1.135     | 1.339     | 1.433     | 1.509      |
    | MonkeyOCR-pro-3B  | 4090  | 1.321    | 1.300     | 1.384     | 1.343     | 1.410      |
    | MonkeyOCR-pro-1.2B | 3090  | 0.919    | 1.086     | 1.166     | 1.182     | 1.199      |
    | MonkeyOCR-pro-1.2B | A6000 | 1.177    | 1.361     | 1.506     | 1.525     | 1.569      |
    | MonkeyOCR-pro-1.2B | H800  | 1.466    | 1.719     | 1.763     | 1.875     | 1.650      |
    | MonkeyOCR-pro-1.2B | 4090  | 1.759    | 1.987     | 2.260     | 2.345     | 2.415      |

    </details>
*   **End-to-End Evaluation Results:**
    <details>
    <summary>Expand Table</summary>

    | **Model Type**              | **Methods**               | **Overall**<br/>*EN* | **Overall**<br/>*ZH* |
    | :-------------------------- | :------------------------ | :------------------- | :------------------- |
    | **Pipeline Tools**          | MonkeyOCR-3B*             | 0.154               | 0.277               |
    | **Mix**                     | MonkeyOCR-pro-3B [Demo]  | **0.138**           | **0.206**           |
    | **Mix**                     | MonkeyOCR-pro-1.2B      | 0.153               | 0.223               |

    </details>

*   **Benchmark Results (olmOCR-bench):**
    <details>
    <summary>Expand Table</summary>

    | Model                   | Overall  |
    | :---------------------- | :------- |
    | MonkeyOCR-pro-3B [Demo] | **75.8** |
    | MonkeyOCR-pro-1.2B      | 71.8     |
    </details>

**Quick Start:**

1.  **Install:** Follow the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support).
2.  **Download Model Weights:**  Use the provided `download_model.py` script or download from Hugging Face.
3.  **Inference:** Run the `parse.py` script with your input PDF or image.
4.  **Gradio Demo:** Launch the interactive demo with `python demo/demo_gradio.py`.
5.  **FastAPI:** Start the API service with `uvicorn api.main:app --port 8000`.
6.  **Docker Deployment:** Use the provided Docker compose files for easy deployment.
7.  **Quantization:** Quantize the model using AWQ (see [quantization guide](docs/Quantization.md)).

**Example Outputs:**

*   Processed Markdown File (`your.md`)
*   Layout Results (`your_layout.pdf`)
*   Intermediate Block Results (`your_middle.json`)

**See more information about usage example, docker deployment and Windows support. Refer to [Quick Start](https://github.com/Yuliang-Liu/MonkeyOCR#quick-start)**

**Visualization Demo:**

Experience MonkeyOCR firsthand through our intuitive demo. Visit [demo](http://vlrlabmonkey.xyz:7685/) and upload a PDF or image to see it in action.

<img src="asserts/Visualization.GIF?raw=true" width="600"/>

**Citing MonkeyOCR:**  Please use the provided BibTeX entry if you utilize MonkeyOCR in your research.

**Acknowledgments:**  Special thanks to the individuals and organizations listed in the acknowledgments section for their contributions.

**Limitations:**  MonkeyOCR has certain limitations regarding specific document types and language support, which are planned for future improvements.

**Copyright:**  The model is intended for academic research and non-commercial use only.