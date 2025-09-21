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

# FastDeploy: Deploy LLMs and VLMs with Speed and Efficiency

**FastDeploy** is a powerful toolkit built on PaddlePaddle, designed to streamline the inference and deployment of Large Language Models (LLMs) and Visual Language Models (VLMs), providing production-ready solutions for optimized performance. Explore the original repository on [GitHub](https://github.com/PaddlePaddle/FastDeploy) for more details.

## Key Features

*   **Optimized Resource Utilization:** Achieve maximum efficiency with our industrial-grade load-balanced PD Disaggregation, featuring context caching and dynamic instance role switching.
*   **High-Performance KV Cache Transmission:** Benefit from a lightweight, high-performance transport library with intelligent NVLink/RDMA selection.
*   **OpenAI API & vLLM Compatibility:** Deploy with ease using our out-of-the-box vLLM interface compatibility.
*   **Extensive Quantization Support:** Utilize a wide range of quantization formats, including W8A16, W8A8, W4A16, W4A8, W2A16, and FP8, for enhanced performance.
*   **Advanced Acceleration Techniques:** Leverage techniques like speculative decoding, Multi-Token Prediction (MTP), and Chunked Prefill.
*   **Multi-Hardware Support:** Compatible with NVIDIA GPUs, Kunlunxin XPUs, Hygon DCUs, Ascend NPUs, Iluvatar GPUs, Enflame GCUs, MetaX GPUs, and more.

## What's New

*   **[2025-09] 🔥 FastDeploy v2.2:** Now offers compatibility with HuggingFace ecosystem models, further optimized performance, and supports baidu/ERNIE-4.5-21B-A3B-Thinking!
*   **[2025-08] 🔥 FastDeploy v2.1:** Introduced a new KV Cache scheduling strategy, expanded PD separation and CUDA Graph support, and enhanced hardware support for Kunlun and Hygon.
*   **[2025-07] Inference Deployment Challenge:** Participate in the FastDeploy 2.0 Inference Deployment Challenge for the ERNIE 4.5 series models. [Sign up here](https://www.wjx.top/vm/meSsp3L.aspx#)

## Requirements

*   **Operating System:** Linux
*   **Python:** 3.10 ~ 3.12

## Installation

FastDeploy supports various hardware platforms. Detailed installation instructions are available for:

*   [NVIDIA GPU](https://paddlepaddle.github.io/FastDeploy/get_started/installation/nvidia_gpu.md)
*   [Kunlunxin XPU](https://paddlepaddle.github.io/FastDeploy/get_started/installation/kunlunxin_xpu.md)
*   [Iluvatar GPU](https://paddlepaddle.github.io/FastDeploy/get_started/installation/iluvatar_gpu.md)
*   [Enflame GCU](https://paddlepaddle.github.io/FastDeploy/get_started/installation/Enflame_gcu.md)
*   [Hygon DCU](https://paddlepaddle.github.io/FastDeploy/get_started/installation/hygon_dcu.md)
*   [MetaX GPU](https://paddlepaddle.github.io/FastDeploy/get_started/installation/metax_gpu.md)

**Note:** Support for additional hardware platforms, including Ascend NPU, is under development.

## Get Started

Quickly learn how to use FastDeploy with these resources:

*   [10-Minutes Quick Deployment](https://paddlepaddle.github.io/FastDeploy/get_started/quick_start.md)
*   [ERNIE-4.5 Large Language Model Deployment](https://paddlepaddle.github.io/FastDeploy/get_started/ernie-4.5.md)
*   [ERNIE-4.5-VL Multimodal Model Deployment](https://paddlepaddle.github.io/FastDeploy/get_started/ernie-4.5-vl.md)
*   [Offline Inference Development](https://paddlepaddle.github.io/FastDeploy/offline_inference.md)
*   [Online Service Deployment](https://paddlepaddle.github.io/FastDeploy/online_serving/README.md)
*   [Best Practices](https://paddlepaddle.github.io/FastDeploy/best_practices/README.md)

## Supported Models

*   [Full Supported Models List](https://paddlepaddle.github.io/FastDeploy/supported_models.md)

## Advanced Usage

*   [Quantization](https://paddlepaddle.github.io/FastDeploy/quantization/README.md)
*   [PD Disaggregation Deployment](https://paddlepaddle.github.io/FastDeploy/features/disaggregated.md)
*   [Speculative Decoding](https://paddlepaddle.github.io/FastDeploy/features/speculative_decoding.md)
*   [Prefix Caching](https://paddlepaddle.github.io/FastDeploy/features/prefix_caching.md)
*   [Chunked Prefill](https://paddlepaddle.github.io/FastDeploy/features/chunked_prefill.md)

## Acknowledgement

FastDeploy is licensed under the [Apache-2.0 open-source license](./LICENSE). We gratefully acknowledge the portions of [vLLM](https://github.com/vllm-project/vllm) code that have been referenced and incorporated.