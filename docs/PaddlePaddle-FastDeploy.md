<p align="center">
  <a href="https://github.com/PaddlePaddle/FastDeploy/releases"><img src="https://github.com/user-attachments/assets/42b0039f-39e3-4279-afda-6d1865dfbffb" width="500"></a>
</p>

<p align="center">
    <a href=""><img src="https://img.shields.io/badge/python-3.10-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux-pink.svg"></a>
    <a href="https://github.com/PaddlePaddle/FastDeploy/graphs/contributors"><img src="https://img.shields.io/github/contributors/PaddlePaddle/FastDeploy?color=9ea"></a>
    <a href="https://github.com/PaddlePaddle/FastDeploy/commits"><img src="https://img.shields.io/github/commit-activity/m/PaddlePaddle/FastDeploy?color=3af"></a>
    <a href="https://github.com/PaddlePaddle/FastDeploy/issues"><img src="https://img.shields.io/github/issues/PaddlePaddle/FastDeploy?color=9cc"></a>
    <a href="https://github.com/PaddlePaddle/FastDeploy/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/FastDeploy?color=ccf"></a>
</p>

<p align="center">
     <a href="https://trendshift.io/repositories/4046" target="_blank"><img src="https://trendshift.io/api/badge/repositories/4046" alt="PaddlePaddle%2FFastDeploy | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a></br>
    <a href="https://paddlepaddle.github.io/FastDeploy/get_started/installation/nvidia_gpu/"><b> Installation </b></a>
    |
    <a href="https://paddlepaddle.github.io/FastDeploy/get_started/quick_start"><b> Quick Start </b></a>
    |
    <a href="https://paddlepaddle.github.io/FastDeploy/supported_models/"><b> Supported Models </b></a>
</p>

---

# FastDeploy: Deploy LLMs and VLMs with Speed and Efficiency

**FastDeploy** is a powerful toolkit based on PaddlePaddle, designed to provide production-ready deployment solutions for Large Language Models (LLMs) and Visual Language Models (VLMs).  [See the original repo](https://github.com/PaddlePaddle/FastDeploy).

## Key Features

*   🚀 **Optimized Performance**:  Achieve high throughput and low latency with advanced acceleration techniques.
*   🧠 **Load-Balanced PD Disaggregation**: Utilize an industrial-grade solution with context caching and dynamic instance role switching.
*   🔗 **Unified KV Cache Transmission**: Benefit from a high-performance transport library with intelligent NVLink/RDMA selection.
*   🔌 **vLLM Compatibility**: Seamlessly integrate with the vLLM interface for easy deployment.
*   ⚙️ **Comprehensive Quantization Support**: Leverage various quantization formats, including W8A16, W8A8, W4A16, W4A8, W2A16, and FP8.
*   ⏩ **Advanced Acceleration**: Utilize speculative decoding, Multi-Token Prediction (MTP), and Chunked Prefill for faster inference.
*   💻 **Multi-Hardware Support**: Deploy on NVIDIA GPUs, Kunlunxin XPUs, Hygon DCUs, Ascend NPUs, Iluvatar GPUs, Enflame GCUs, MetaX GPUs, and more.

## News

*   **[2025-08] 🔥 Released FastDeploy v2.1:** Introduction of a new KV Cache scheduling strategy, expanding support for PD separation and CUDA Graph. Includes enhancements for Kunlun and Hygon platforms, with optimizations to service and inference engines.
*   **[2025-07] The FastDeploy 2.0 Inference Deployment Challenge is now live!** Complete the inference deployment task for the ERNIE 4.5 series open-source models to win prizes! 🎁
*   **[2025-06] 🔥 Released FastDeploy v2.0:**  Added support for ERNIE 4.5 inference and deployment, including an industrial-grade PD disaggregation with context caching for optimized MoE model inference.

## Requirements

*   OS: Linux
*   Python: 3.10 ~ 3.12

## Installation

FastDeploy supports various hardware platforms for inference deployment. Follow the instructions for your specific hardware:

*   [NVIDIA GPU](https://paddlepaddle.github.io/FastDeploy/get_started/installation/nvidia_gpu/)
*   [Kunlunxin XPU](https://paddlepaddle.github.io/FastDeploy/get_started/installation/kunlunxin_xpu/)
*   [Iluvatar GPU](https://paddlepaddle.github.io/FastDeploy/get_started/installation/iluvatar_gpu/)
*   [Enflame GCU](https://paddlepaddle.github.io/FastDeploy/get_started/installation/Enflame_gcu/)
*   [Hygon DCU](https://paddlepaddle.github.io/FastDeploy/get_started/installation/hygon_dcu/)

**Note:**  Support for Ascend NPU and MetaX GPU is currently in development.

## Get Started

Explore the following documentation to learn how to use FastDeploy:

*   [10-Minutes Quick Deployment](https://paddlepaddle.github.io/FastDeploy/get_started/quick_start.md)
*   [ERNIE-4.5 Large Language Model Deployment](https://paddlepaddle.github.io/FastDeploy/get_started/ernie-4.5.md)
*   [ERNIE-4.5-VL Multimodal Model Deployment](https://paddlepaddle.github.io/FastDeploy/get_started/ernie-4.5-vl.md)
*   [Offline Inference Development](https://paddlepaddle.github.io/FastDeploy/docs/offline_inference.md)
*   [Online Service Deployment](https://paddlepaddle.github.io/FastDeploy/docs/online_serving/README.md)
*   [Full Supported Models List](https://paddlepaddle.github.io/FastDeploy/docs/supported_models.md)
*   [Best Practices](https://paddlepaddle.github.io/FastDeploy/docs/best_practices/README.md)

## Supported Models

| Model                      | Data Type                                    | PD Disaggregation | Chunked Prefill | Prefix Caching | MTP | CUDA Graph | Maximum Context Length |
| :------------------------- | :------------------------------------------- | :---------------- | :-------------- | :------------- | :-- | :--------- | :--------------------- |
| ERNIE-4.5-300B-A47B        | BF16/WINT4/WINT8/W4A8C8/WINT2/FP8            | ✅                | ✅              | ✅             | ✅  | ✅         | 128K                   |
| ERNIE-4.5-300B-A47B-Base   | BF16/WINT4/WINT8                               | ✅                | ✅              | ✅             | ❌   | ✅         | 128K                   |
| ERNIE-4.5-VL-424B-A47B     | BF16/WINT4/WINT8                               | WIP               | ✅              | WIP            | ❌   | WIP        | 128K                   |
| ERNIE-4.5-VL-28B-A3B       | BF16/WINT4/WINT8                               | ❌                 | ✅              | WIP            | ❌   | WIP        | 128K                   |
| ERNIE-4.5-21B-A3B          | BF16/WINT4/WINT8/FP8                          | ❌                 | ✅              | ✅             | ✅  | ✅         | 128K                   |
| ERNIE-4.5-21B-A3B-Base     | BF16/WINT4/WINT8/FP8                          | ✅                | ✅              | ✅             | ❌   | ✅         | 128K                   |
| ERNIE-4.5-0.3B             | BF16/WINT8/FP8                                | ✅                | ✅              | ✅             | ❌   | ✅         | 128K                   |

## Advanced Usage

*   [Quantization](https://paddlepaddle.github.io/FastDeploy/docs/quantization/README.md)
*   [PD Disaggregation Deployment](https://paddlepaddle.github.io/FastDeploy/docs/features/disaggregated.md)
*   [Speculative Decoding](https://paddlepaddle.github.io/FastDeploy/docs/features/speculative_decoding.md)
*   [Prefix Caching](https://paddlepaddle.github.io/FastDeploy/docs/features/prefix_caching.md)
*   [Chunked Prefill](https://paddlepaddle.github.io/FastDeploy/docs/features/chunked_prefill.md)

## Acknowledgement

FastDeploy is licensed under the [Apache-2.0 open-source license](./LICENSE). We gratefully acknowledge the use of code portions from [vLLM](https://github.com/vllm-project/vllm) to maintain interface compatibility.
```
Key improvements and explanations:

*   **SEO Optimization:**  The use of the keywords "LLMs", "VLMs", "Inference", and "Deployment" throughout the text.  The headings are clear and use relevant terms.
*   **Hook:**  The first sentence immediately grabs attention, clearly stating what FastDeploy *is*.
*   **Key Features (Bulleted):** Highlights the most important aspects with clear language.
*   **Clear Headings:** Organizes the content logically.
*   **Concise Language:**  Replaced some wordier phrases with more direct ones.
*   **Links:** Included links back to the original repo and other important pages, using descriptive anchor text.
*   **Updated News Section:** Expanded on the news section to be more informative, and included dates for SEO purposes.
*   **Complete and Informative:** Included all relevant sections from the original, providing a comprehensive overview.
*   **Markdown Formatting:** Properly formatted using Markdown for readability on GitHub.
*   **Focus on Benefits:**  The "Key Features" section focuses on the *benefits* of using FastDeploy (e.g., "Optimized Performance," "Seamless Integration").