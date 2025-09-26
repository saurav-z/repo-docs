# MonkeyOCR: Advanced Document Parsing with Superior Accuracy and Speed

**Unleash the power of AI to accurately parse and understand your documents with MonkeyOCR, a state-of-the-art solution built on a novel Structure-Recognition-Relation (SRR) triplet paradigm.** ([Original Repo](https://github.com/Yuliang-Liu/MonkeyOCR))

[![arXiv](https://img.shields.io/badge/Arxiv-MonkeyOCR-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-black.svg?logo=HuggingFace)](https://huggingface.co/echo840/MonkeyOCR-pro-3B)
[![GitHub issues](https://img.shields.io/github/issues/Yuliang-Liu/MonkeyOCR?color=critical&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Yuliang-Liu/MonkeyOCR?color=success&label=Issues)](https://github.com/Yuliang-Liu/MonkeyOCR/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/LICENSE.txt)
[![GitHub views](https://komarev.com/ghpvc/?username=Yuliang-Liu&repo=MonkeyOCR&color=brightgreen&label=Views)](https://github.com/Yuliang-Liu/MonkeyOCR)

> **MonkeyOCR: Document Parsing with a Structure-Recognition-Relation Triplet Paradigm**<br>
> Zhang Li, Yuliang Liu, Qiang Liu, Zhiyin Ma, Ziyang Zhang, Shuo Zhang, Zidun Guo, Jiarui Zhang, Xinyu Wang, Xiang Bai <br>
[![arXiv](https://img.shields.io/badge/Arxiv-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05218) 
[![Source_code](https://img.shields.io/badge/Code-Available-white)](README.md)
[![Model Weight](https://img.shields.io/badge/HuggingFace-gray)](https://huggingface.co/echo840/MonkeyOCR)
[![Model Weight](https://img.shields.io/badge/ModelScope-green)](https://modelscope.cn/models/l1731396519/MonkeyOCR)
[![Public Courses](https://img.shields.io/badge/Openbayes-yellow)](https://openbayes.com/console/public/tutorials/91ESrGvEvBq)
[![Demo](https://img.shields.io/badge/Demo-blue)](http://vlrlabmonkey.xyz:7685/)


## Key Features

*   **Superior Accuracy:** MonkeyOCR-pro-1.2B outperforms other models in Chinese and English documents.
*   **Exceptional Speed:** Achieve up to 36% speed improvement with MonkeyOCR-pro-1.2B.
*   **Competitive Performance:**  MonkeyOCR-pro-1.2B excels over Nanonets-OCR-3B and even surpasses closed-source and large open-source VLMs like Gemini 2.0-Flash, Gemini 2.5-Pro, Qwen2.5-VL-72B, GPT-4o, and InternVL3-78B.
*   **SRR Paradigm:** A unique Structure-Recognition-Relation triplet paradigm that simplifies the document parsing pipeline.
*   **Versatile Deployment:**  Supports a variety of hardware, including various GPUs and Docker.

## Performance Highlights

MonkeyOCR demonstrates strong performance across several benchmarks, including:

*   **Superior Results on OmniDocBench**:  MonkeyOCR-pro-3B achieves the best overall performance on both English and Chinese documents.
*   **olmOCR-Bench**: MonkeyOCR-pro-1.2B outperforms Nanonets-OCR-3B by 7.3%.
*   **Speed and Efficiency:** The tables below provide a detailed look at the model's performance in terms of speed (pages/second) across different GPUs and document page counts.

### Inference Speed (Pages/s) on Different GPUs

<details>
<summary>Inference Speed Tables</summary>

**End-to-End Parsing**

| Model              | GPU   | 50 Pages | 100 Pages | 300 Pages | 500 Pages | 1000 Pages |
| ------------------ | ----- | -------- | --------- | --------- | --------- | ---------- |
| MonkeyOCR-pro-3B   | 3090  | 0.492    | 0.484     | 0.497     | 0.492     | 0.496      |
|                    | A6000 | 0.585    | 0.587     | 0.609     | 0.598     | 0.608      |
|                    | H800  | 0.923    | 0.768     | 0.897     | 0.930     | 0.891      |
|                    | 4090  | 0.972    | 0.969     | 1.006     | 0.986     | 1.006      |
| MonkeyOCR-pro-1.2B | 3090  | 0.615    | 0.660     | 0.677     | 0.687     | 0.683      |
|                    | A6000 | 0.709    | 0.786     | 0.825     | 0.829     | 0.825      |
|                    | H800  | 0.965    | 1.082     | 1.101     | 1.145     | 1.015      |
|                    | 4090  | 1.194    | 1.314     | 1.436     | 1.442     | 1.434      |

**VLM OCR Speed**

| Model              | GPU   | 50 Pages | 100 Pages | 300 Pages | 500 Pages | 1000 Pages |
| ------------------ | ----- | -------- | --------- | --------- | --------- | ---------- |
| MonkeyOCR-pro-3B   | 3090  | 0.705    | 0.680     | 0.711     | 0.700     | 0.724      |
|                    | A6000 | 0.885    | 0.860     | 0.915     | 0.892     | 0.934      |
|                    | H800  | 1.371    | 1.135     | 1.339     | 1.433     | 1.509      |
|                    | 4090  | 1.321    | 1.300     | 1.384     | 1.343     | 1.410      |
| MonkeyOCR-pro-1.2B | 3090  | 0.919    | 1.086     | 1.166     | 1.182     | 1.199      |
|                    | A6000 | 1.177    | 1.361     | 1.506     | 1.525     | 1.569      |
|                    | H800  | 1.466    | 1.719     | 1.763     | 1.875     | 1.650      |
|                    | 4090  | 1.759    | 1.987     | 2.260     | 2.345     | 2.415      |

</details>

## Quickstart

Follow these steps to get started with MonkeyOCR:

1.  **Installation:** Follow the [installation guide](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda_pp.md#install-with-cuda-support) to set up your environment.
2.  **Download Model Weights:**
    *   Use `pip install huggingface_hub` and `python tools/download_model.py -n MonkeyOCR-pro-3B`
    *   Or use `pip install modelscope` and `python tools/download_model.py -t modelscope -n MonkeyOCR-pro-3B`
3.  **Inference:**  Run the following command to parse documents:

    ```bash
    python parse.py input_path  # Replace input_path with your file/directory.
    ```
4.  **Gradio Demo:** Run the demo with `python demo/demo_gradio.py` and access it at `http://localhost:7860`.
5.  **FastAPI:**  Start the API with `uvicorn api.main:app --port 8000` and access documentation at `http://localhost:8000/docs`.

## Detailed Information

*   **Supported Hardware:**  The model has been tested on a variety of GPUs including 3090, 4090, A6000, H800, A100, and 4060.
*   **News:**
    *   `2025.07.10`: Released [MonkeyOCR-pro-1.2B](https://huggingface.co/echo840/MonkeyOCR-pro-1.2B).
    *   `2025.06.12`: Trending on [Hugging Face](https://huggingface.co/models?sort=trending).
    *   `2025.06.05`: Released [MonkeyOCR](https://huggingface.co/echo840/MonkeyOCR).
*   **Docker Deployment:** Available, see instructions in the original README.
*   **Windows Support:** Details available in the [windows support guide](docs/windows_support.md).
*   **Quantization:**  This model can be quantized using AWQ (see [quantization guide](docs/Quantization.md)).
*   **Benchmark Results:**  See the original README for detailed end-to-end evaluation results.

##  Visualization Demo

Experience MonkeyOCR's capabilities firsthand: http://vlrlabmonkey.xyz:7685

>  Upload, parse, and see the results!

## Citation

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

## Acknowledgments

(List of acknowledgements from the original README)

## Limitations

(List of limitations from the original README)