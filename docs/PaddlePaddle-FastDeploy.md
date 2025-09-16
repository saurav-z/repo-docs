# FastDeploy: Deploy LLMs and VLMs with Speed and Efficiency

**FastDeploy empowers you to deploy large language models (LLMs) and visual language models (VLMs) quickly and efficiently.**  [Get Started with FastDeploy](https://github.com/PaddlePaddle/FastDeploy)

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

## Key Features

FastDeploy provides a robust and efficient platform for deploying your AI models:

*   **Optimized Performance:** Leverages core acceleration technologies for production-ready deployment.
*   **PD Disaggregation**: Offers an industrial-grade solution with context caching and dynamic instance role switching.
*   **Unified KV Cache Transmission:** Utilizes a lightweight, high-performance transport library with intelligent NVLink/RDMA selection.
*   **OpenAI API & vLLM Compatibility:** Supports one-command deployment and vLLM interface compatibility.
*   **Comprehensive Quantization Support:** Offers various quantization formats, including W8A16, W8A8, W4A16, W4A8, W2A16, and FP8.
*   **Advanced Acceleration Techniques:** Includes speculative decoding, Multi-Token Prediction (MTP), and Chunked Prefill.
*   **Multi-Hardware Support:**  Compatible with NVIDIA GPUs, Kunlunxin XPUs, Hygon DCUs, Ascend NPUs, Iluvatar GPUs, Enflame GCUs, MetaX GPU etc.

## News

*   **[2025-09] 🔥 FastDeploy v2.2 Released:** Compatibility with HuggingFace models, further performance optimization, and support for baidu/ERNIE-21B-A3B-Thinking!
*   **[2025-08] 🔥 FastDeploy v2.1 Released:** Introduced a new KV Cache scheduling strategy, expanded PD separation and CUDA Graph support. Enhanced hardware support for Kunlun and Hygon, along with comprehensive optimizations.
*   **[2025-07] Inference Deployment Challenge:** Participate in the FastDeploy 2.0 Challenge for ERNIE 4.5 models! 🎁 [Sign up](https://www.wjx.top/vm/meSsp3L.aspx#) | [Event Details](https://github.com/PaddlePaddle/FastDeploy/discussions/2728)
*   **[2025-06] 🔥 FastDeploy v2.0 Released:** Support for ERNIE 4.5 inference and deployment.  Open-sourced industrial-grade PD disaggregation.

## About

FastDeploy is designed as an inference and deployment toolkit built upon PaddlePaddle. It offers a streamlined and efficient solution for deploying LLMs and VLMs. It's designed for production environments.

## Requirements

*   **Operating System:** Linux
*   **Python:** 3.10 ~ 3.12

## Installation

Install FastDeploy for your hardware:

*   [NVIDIA GPU](./docs/get_started/installation/nvidia_gpu.md)
*   [Kunlunxin XPU](./docs/get_started/installation/kunlunxin_xpu.md)
*   [Iluvatar GPU](./docs/get_started/installation/iluvatar_gpu.md)
*   [Enflame GCU](./docs/get_started/installation/Enflame_gcu.md)
*   [Hygon DCU](./docs/get_started/installation/hygon_dcu.md)
*   [MetaX GPU](./docs/get_started/installation/metax_gpu.md)

**Note:**  Ongoing development includes Ascend NPU support.

## Get Started

*   [10-Minutes Quick Deployment](./docs/get_started/quick_start.md)
*   [ERNIE-4.5 Large Language Model Deployment](./docs/get_started/ernie-4.5.md)
*   [ERNIE-4.5-VL Multimodal Model Deployment](./docs/get_started/ernie-4.5-vl.md)
*   [Offline Inference Development](./docs/offline_inference.md)
*   [Online Service Deployment](./docs/online_serving/README.md)
*   [Best Practices](./docs/best_practices/README.md)

## Supported Models

*   [Full Supported Models List](./docs/supported_models.md)

## Advanced Usage

*   [Quantization](./docs/quantization/README.md)
*   [PD Disaggregation Deployment](./docs/features/disaggregated.md)
*   [Speculative Decoding](./docs/features/speculative_decoding.md)
*   [Prefix Caching](./docs/features/prefix_caching.md)
*   [Chunked Prefill](./docs/features/chunked_prefill.md)

## Acknowledgement

FastDeploy is licensed under the [Apache-2.0 open-source license](./LICENSE).  We express our gratitude to [vLLM](https://github.com/vllm-project/vllm) for code contributions used to maintain interface compatibility.