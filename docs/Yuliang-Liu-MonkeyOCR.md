# MonkeyOCR: Unlock Advanced Document Parsing with AI

**MonkeyOCR is a cutting-edge document parsing model that leverages a Structure-Recognition-Relation (SRR) triplet paradigm to extract and understand the structure and content of your documents.**  [View the original repository](https://github.com/Yuliang-Liu/MonkeyOCR)

**Key Features:**

*   **Superior Accuracy:** Achieves state-of-the-art results on both English and Chinese documents, including significant improvements on complex formats like formulas and tables.
*   **Efficient Processing:** Offers fast multi-page document parsing, outperforming other leading models.
*   **Flexible Output:** Generates markdown output, layout results, and detailed intermediate block results for versatile use.
*   **Easy to Use:**  Includes a Gradio demo and FastAPI service for easy integration and testing.
*   **Modular Design:** Leverages an innovative Structure-Recognition-Relation (SRR) triplet paradigm, enabling efficient document processing.

**Key Benefits:**

*   **Improved Productivity:** Quickly and accurately extract information from documents.
*   **Enhanced Data Analysis:** Gain deeper insights from structured document data.
*   **Versatile Applications:** Ideal for a wide range of document types, from academic papers to financial reports.

**Quick Start:**

1.  **Installation:** Follow the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda.md#install-with-cuda-support) to set up your environment.
2.  **Model Download:** Download the model weights from [Hugging Face](https://huggingface.co/echo840/MonkeyOCR) or [ModelScope](https://modelscope.cn/models/l1731396519/MonkeyOCR). Use the `tools/download_model.py` script.
3.  **Inference:** Use the `parse.py` script with various options for file parsing, task selection, and output customization. See the Quick Start section in the original README for details and examples.
4.  **Demo:**  Run the Gradio demo with `python demo/demo_gradio.py` to experience MonkeyOCR's capabilities.
5.  **API:** Run the FastAPI service with `uvicorn api.main:app --port 8000` for programmatic access.

**Benchmark Results:**

MonkeyOCR demonstrates impressive performance across various document types and tasks.

*   **End-to-end Evaluation:** MonkeyOCR-3B and MonkeyOCR-3B* models consistently outperform other methods in overall document parsing, as shown in the table.
*   **Text Recognition Performance:** MonkeyOCR delivers superior text recognition accuracy across diverse document types, as demonstrated in the detailed table.

*(Refer to the original README for the detailed tables and performance metrics.)*

**Explore Further:**

*   **Gradio Demo:**  [http://vlrlabmonkey.xyz:7685](http://vlrlabmonkey.xyz:7685)
*   **Windows Support:**  Check the [Windows Support](docs/windows_support.md) Guide for details.
*   **Quantization:** Explore [Quantization guide](docs/Quantization.md)
*   **Docker Deployment:** Use the provided Docker files for easy deployment.

**Citation:**

If you use MonkeyOCR in your research, please cite the following:

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

(Include the acknowledgments section from the original README).

**Alternative Models to Explore:**

(Include the alternative models section from the original README).

**Copyright:**

(Include the copyright section from the original README).