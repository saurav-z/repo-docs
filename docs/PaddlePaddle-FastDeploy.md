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

**FastDeploy provides a production-ready toolkit for deploying large language models and visual language models based on PaddlePaddle, significantly boosting performance.**  ([Back to Original Repo](https://github.com/PaddlePaddle/FastDeploy))

## Key Features

*   🚀 **Optimized Performance:** Achieve superior inference speeds and resource utilization with core acceleration technologies.
*   🔄 **Load-Balanced PD Disaggregation:** Industrial-grade solution featuring context caching and dynamic instance role switching.
*   🤝 **OpenAI API Compatibility:** Seamless integration with OpenAI API Server and vLLM for easy deployment.
*   🧮 **Comprehensive Quantization Support:** Utilize various quantization formats, including W8A16, W8A8, and FP8, for model optimization.
*   ⏩ **Advanced Acceleration Techniques:** Benefit from speculative decoding, Multi-Token Prediction (MTP), and Chunked Prefill for enhanced performance.
*   🖥️ **Multi-Hardware Support:** Deploy models across diverse hardware platforms, including NVIDIA GPUs, Kunlunxin XPU, Hygon DCU, and more.

## News

*   **[2025-09] 🔥 FastDeploy v2.2:** Compatibility with HuggingFace models, performance optimizations, and support for baidu/ERNIE-21B-A3B-Thinking.
*   **[2025-08] 🔥 Released FastDeploy v2.1:** New KV Cache scheduling strategy, expanded PD separation and CUDA Graph support, and hardware enhancements for Kunlun and Hygon.
*   **[2025-07] FastDeploy 2.0 Inference Deployment Challenge:** Win prizes by deploying ERNIE 4.5 models. [Sign up here](https://www.wjx.top/vm/meSsp3L.aspx#) and [event details](https://github.com/PaddlePaddle/FastDeploy/discussions/2728).
*   **[2025-06] 🔥 Released FastDeploy v2.0:** Supports inference and deployment for ERNIE 4.5 and features industrial-grade PD disaggregation.

## About

FastDeploy is an inference and deployment toolkit based on PaddlePaddle, designed for LLMs and VLMs. It is built with core acceleration technologies to provide production-ready, out-of-the-box deployment solutions.

## Requirements

*   **OS:** Linux
*   **Python:** 3.10 ~ 3.12

## Installation

FastDeploy supports various hardware platforms. Detailed installation instructions are available for:

*   [NVIDIA GPU](https://paddlepaddle.github.io/FastDeploy/get_started/installation/nvidia_gpu/)
*   [Kunlunxin XPU](https://paddlepaddle.github.io/FastDeploy/get_started/installation/kunlunxin_xpu/)
*   [Iluvatar GPU](https://paddlepaddle.github.io/FastDeploy/get_started/installation/iluvatar_gpu/)
*   [Enflame GCU](https://paddlepaddle.github.io/FastDeploy/get_started/installation/Enflame_gcu/)
*   [Hygon DCU](https://paddlepaddle.github.io/FastDeploy/get_started/installation/hygon_dcu/)
*   [MetaX GPU](https://paddlepaddle.github.io/FastDeploy/get_started/installation/metax_gpu/)

**Note:** Support for additional hardware platforms, including Ascend NPU, is currently under development.

## Get Started

Explore FastDeploy with these resources:

*   [10-Minutes Quick Deployment](https://paddlepaddle.github.io/FastDeploy/get_started/quick_start)
*   [ERNIE-4.5 Large Language Model Deployment](https://paddlepaddle.github.io/FastDeploy/get_started/ernie-4.5.md)
*   [ERNIE-4.5-VL Multimodal Model Deployment](https://paddlepaddle.github.io/FastDeploy/get_started/ernie-4.5-vl.md)
*   [Offline Inference Development](https://paddlepaddle.github.io/FastDeploy/offline_inference.md)
*   [Online Service Deployment](https://paddlepaddle.github.io/FastDeploy/online_serving/README.md)
*   [Best Practices](https://paddlepaddle.github.io/FastDeploy/best_practices/README.md)

## Supported Models

*   [Full Supported Models List](https://paddlepaddle.github.io/FastDeploy/supported_models/)

## Advanced Usage

*   [Quantization](https://paddlepaddle.github.io/FastDeploy/quantization/README.md)
*   [PD Disaggregation Deployment](https://paddlepaddle.github.io/FastDeploy/features/disaggregated.md)
*   [Speculative Decoding](https://paddlepaddle.github.io/FastDeploy/features/speculative_decoding.md)
*   [Prefix Caching](https://paddlepaddle.github.io/FastDeploy/features/prefix_caching.md)
*   [Chunked Prefill](https://paddlepaddle.github.io/FastDeploy/features/chunked_prefill.md)

## Acknowledgement

FastDeploy is licensed under the [Apache-2.0 open-source license](./LICENSE). We acknowledge and appreciate the use of code portions from [vLLM](https://github.com/vllm-project/vllm) for interface compatibility.
```
Key improvements and optimizations:

*   **SEO-Friendly Title and Hook:**  The title is optimized with keywords. The one-sentence hook highlights the main benefit (speed and efficiency).
*   **Clear Headings:**  Uses descriptive headings to organize the information.
*   **Bulleted Key Features:**  Provides a concise overview of FastDeploy's capabilities.
*   **Concise Language:** Streamlines the information for readability.
*   **Links Back to Original Repo:**  Includes a clear link to the original repo, and links within the README itself are maintained for user convenience.
*   **Keyword Optimization:** Incorporates relevant keywords such as "LLMs," "VLMs," "inference," "deployment," "PaddlePaddle," "NVIDIA GPU," and hardware names to improve search engine visibility.
*   **Updated Information:** Incorporates the latest release information and links to relevant resources.
*   **Improved Formatting:** Uses bolding and other formatting to emphasize key information.
*   **Clear Calls to Action:** Guides users to installation, quick start guides, and the list of supported models.