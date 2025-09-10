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

# FastDeploy: Deploy LLMs and VLMs with Lightning Speed and Efficiency

**FastDeploy** is a powerful, open-source toolkit designed for fast and efficient inference and deployment of large language models (LLMs) and visual language models (VLMs).  Get started at the [FastDeploy GitHub repository](https://github.com/PaddlePaddle/FastDeploy)!

## Key Features

*   **Production-Ready Deployment:** Ready-to-use solutions for LLM and VLM deployment.
*   **Optimized Performance:**
    *   🚀 **Load-Balanced PD Disaggregation:**  Maximizes resource utilization with context caching and dynamic instance role switching.
    *   🔄 **Unified KV Cache Transmission:** Efficient NVLink/RDMA-aware transport library.
    *   ⏩ **Advanced Acceleration Techniques:** Supports speculative decoding, Multi-Token Prediction (MTP) and Chunked Prefill.
*   **Wide Compatibility:**
    *   🤝 **OpenAI API Server and vLLM Compatibility:** Deploy with a vLLM interface.
    *   🧮 **Comprehensive Quantization:** Supports a variety of quantization formats including W8A16, W8A8, W4A16, W4A8, W2A16, and FP8.
*   **Multi-Hardware Support:**  Runs on NVIDIA GPUs, Kunlunxin XPUs, Hygon DCUs, Ascend NPUs, Iluvatar GPUs, Enflame GCUs, MetaX GPUs and more.

## News

*   **[2025-09] 🔥 FastDeploy v2.2 is newly released!** Offering HuggingFace ecosystem compatibility, enhanced performance, and support for baidu/ERNIE-21B-A3B-Thinking!
*   **[2025-08] 🔥 Released FastDeploy v2.1:** New KV Cache scheduling, expanded PD separation and CUDA Graph support, and optimized hardware support for platforms like Kunlun and Hygon.
*   **[2025-07]** FastDeploy 2.0 Inference Deployment Challenge is live! Win prizes by deploying ERNIE 4.5 series models. [Sign up](https://www.wjx.top/vm/meSsp3L.aspx#) & [event details](https://github.com/PaddlePaddle/FastDeploy/discussions/2728).
*   **[2025-06] 🔥 Released FastDeploy v2.0:** Supports inference and deployment for ERNIE 4.5, featuring an industrial-grade PD disaggregation solution.

## Requirements

*   OS: Linux
*   Python: 3.10 ~ 3.12

## Installation

Install FastDeploy on various hardware platforms:

*   [NVIDIA GPU](./docs/get_started/installation/nvidia_gpu.md)
*   [Kunlunxin XPU](./docs/get_started/installation/kunlunxin_xpu.md)
*   [Iluvatar GPU](./docs/get_started/installation/iluvatar_gpu.md)
*   [Enflame GCU](./docs/get_started/installation/Enflame_gcu.md)
*   [Hygon DCU](./docs/get_started/installation/hygon_dcu.md)
*   [MetaX GPU](./docs/get_started/installation/metax_gpu.md)

**Note:**  Support for Ascend NPU and other hardware is under development.

## Get Started

Quickly deploy your models with these guides:

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

FastDeploy is licensed under the [Apache-2.0 open-source license](./LICENSE).  We are grateful to the [vLLM](https://github.com/vllm-project/vllm) project, whose code was referenced and incorporated for interface compatibility.