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

# FastDeploy: Supercharge Your LLM and VLM Deployments with Production-Ready Solutions

**FastDeploy** is an inference and deployment toolkit from PaddlePaddle designed to streamline the deployment of Large Language Models (LLMs) and Visual Language Models (VLMs). Get started with FastDeploy on [GitHub](https://github.com/PaddlePaddle/FastDeploy) today!

## Key Features

*   **Accelerated Inference:** Experience significant performance gains with cutting-edge acceleration techniques.
*   **Production-Ready Deployment:** Benefit from out-of-the-box solutions optimized for real-world use cases.
*   **Model Compatibility:** Supports a wide range of models, including those from the Hugging Face ecosystem.
*   **Hardware Agnostic:** Compatible with NVIDIA GPUs, Kunlunxin XPUs, Iluvatar GPUs, Enflame GCUs, Hygon DCUs, Ascend NPUs, MetaX GPUs, and Intel Gaudi.
*   **Advanced Optimization:** Explore features like load-balanced PD disaggregation, unified KV cache transmission, quantization support, speculative decoding, and multi-token prediction.
*   **OpenAI API and vLLM Compatibility:** Deploy your models with ease using OpenAI API server and vLLM compatibility.

## News

*   **[2025-09] 🔥 FastDeploy v2.2 Released!** Includes HuggingFace ecosystem compatibility, performance optimizations, and support for [baidu/ERNIE-21B-A3B-Thinking](https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-Thinking)!
*   **[2025-08] 🔥 FastDeploy v2.1 Released:** Introducing a new KV Cache scheduling strategy and expanded support for PD separation and CUDA Graph. Enhanced hardware support for Kunlun and Hygon and performance improvements.
*   **[2025-07] Inference Deployment Challenge:** Join the FastDeploy 2.0 Inference Deployment Challenge! Win prizes by completing tasks related to ERNIE 4.5 series open-source models! 📌[Sign up here](https://www.wjx.top/vm/meSsp3L.aspx#) 📌[Event details](https://github.com/PaddlePaddle/FastDeploy/discussions/2728)
*   **[2025-06] 🔥 FastDeploy v2.0 Released:** Supports inference and deployment for ERNIE 4.5 and industrial-grade PD disaggregation.

## About

FastDeploy provides a complete toolkit for deploying LLMs and VLMs, delivering:

*   **Load-Balanced PD Disaggregation:** Industrial-grade solution with context caching and dynamic instance role switching.
*   **Unified KV Cache Transmission:** Lightweight, high-performance transport with intelligent NVLink/RDMA selection.
*   **OpenAI API Server and vLLM Compatibility:** One-command deployment with [vLLM](https://github.com/vllm-project/vllm/) interface compatibility.
*   **Comprehensive Quantization Format Support:** W8A16, W8A8, W4A16, W4A8, W2A16, FP8, and more.
*   **Advanced Acceleration Techniques:** Speculative decoding, Multi-Token Prediction (MTP) and Chunked Prefill.
*   **Multi-Hardware Support:** NVIDIA GPU, Kunlunxin XPU, Hygon DCU, Ascend NPU, Iluvatar GPU, Enflame GCU, MetaX GPU, Intel Gaudi etc.

## Requirements

*   **Operating System:** Linux
*   **Python:** 3.10 ~ 3.12

## Installation

Install FastDeploy for your preferred hardware:

*   [NVIDIA GPU](./docs/get_started/installation/nvidia_gpu.md)
*   [Kunlunxin XPU](./docs/get_started/installation/kunlunxin_xpu.md)
*   [Iluvatar GPU](./docs/get_started/installation/iluvatar_gpu.md)
*   [Enflame GCU](./docs/get_started/installation/Enflame_gcu.md)
*   [Hygon DCU](./docs/get_started/installation/hygon_dcu.md)
*   [MetaX GPU](./docs/get_started/installation/metax_gpu.md)
*   [Intel Gaudi](./docs/get_started/installation/intel_gaudi.md)

**Note:** Ascend NPU support is under development.

## Get Started

Explore the following resources to quickly get started with FastDeploy:

*   [10-Minutes Quick Deployment](./docs/get_started/quick_start.md)
*   [ERNIE-4.5 Large Language Model Deployment](./docs/get_started/ernie-4.5.md)
*   [ERNIE-4.5-VL Multimodal Model Deployment](./docs/get_started/ernie-4.5-vl.md)
*   [Offline Inference Development](./docs/offline_inference.md)
*   [Online Service Deployment](./docs/online_serving/README.md)
*   [Best Practices](./docs/best_practices/README.md)

## Supported Models

Find a comprehensive list of supported models and learn about model conversion and usage:

*   [Full Supported Models List](./docs/supported_models.md)

## Advanced Usage

Leverage advanced features for optimized deployment:

*   [Quantization](./docs/quantization/README.md)
*   [PD Disaggregation Deployment](./docs/features/disaggregated.md)
*   [Speculative Decoding](./docs/features/speculative_decoding.md)
*   [Prefix Caching](./docs/features/prefix_caching.md)
*   [Chunked Prefill](./docs/features/chunked_prefill.md)

## Acknowledgement

FastDeploy is licensed under the [Apache-2.0 open-source license](./LICENSE). We acknowledge and express our gratitude for the reference and incorporation of code from [vLLM](https://github.com/vllm-project/vllm) to maintain interface compatibility.