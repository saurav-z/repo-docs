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

--------------------------------------------------------------------------------
# FastDeploy: Supercharge Your LLM and VLM Deployment with PaddlePaddle

**FastDeploy** empowers developers to effortlessly deploy and accelerate large language models (LLMs) and visual language models (VLMs) on various hardware platforms. Learn more about this powerful toolkit on the [FastDeploy GitHub repo](https://github.com/PaddlePaddle/FastDeploy).

## Key Features

*   **Production-Ready Deployment Solutions:** Get out-of-the-box solutions for efficient LLM/VLM deployment.
*   **Load-Balanced PD Disaggregation:** Industrial-grade solution with context caching and dynamic instance role switching for optimal resource utilization and SLO compliance.
*   **Unified KV Cache Transmission:** Lightweight, high-performance transport library with intelligent NVLink/RDMA selection.
*   **OpenAI API and vLLM Compatibility:** Deploy with a single command and enjoy compatibility with the [vLLM](https://github.com/vllm-project/vllm/) interface.
*   **Comprehensive Quantization Support:** Supports W8A16, W8A8, W4A16, W4A8, W2A16, FP8, and more.
*   **Advanced Acceleration Techniques:** Leverage speculative decoding, Multi-Token Prediction (MTP), and Chunked Prefill for enhanced performance.
*   **Multi-Hardware Support:** Compatible with NVIDIA GPUs, Kunlunxin XPU, Hygon DCU, Ascend NPU, Iluvatar GPU, Enflame GCU, MetaX GPU, Intel Gaudi and more.

## News

*   **[2025-09] 🔥 FastDeploy v2.2 Released!** Introducing HuggingFace ecosystem model compatibility, performance optimizations, and support for [baidu/ERNIE-21B-A3B-Thinking](https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-Thinking)!
*   **[2025-08] 🔥 FastDeploy v2.1 Released:** Features a new KV Cache scheduling strategy, expanded PD separation, CUDA Graph support across models, and enhanced hardware support.
*   **[2025-07] FastDeploy 2.0 Inference Deployment Challenge:** Win prizes by deploying ERNIE 4.5 series open-source models! [Sign up here](https://www.wjx.top/vm/meSsp3L.aspx#) and view [event details](https://github.com/PaddlePaddle/FastDeploy/discussions/2728).
*   **[2025-06] 🔥 FastDeploy v2.0 Released:** Supports ERNIE 4.5 inference and deployment with industrial-grade PD disaggregation.

## Requirements

*   **Operating System:** Linux
*   **Python:** 3.10 ~ 3.12

## Installation

Install FastDeploy on your preferred hardware:

*   [NVIDIA GPU](./docs/get_started/installation/nvidia_gpu.md)
*   [Kunlunxin XPU](./docs/get_started/installation/kunlunxin_xpu.md)
*   [Iluvatar GPU](./docs/get_started/installation/iluvatar_gpu.md)
*   [Enflame GCU](./docs/get_started/installation/Enflame_gcu.md)
*   [Hygon DCU](./docs/get_started/installation/hygon_dcu.md)
*   [MetaX GPU](./docs/get_started/installation/metax_gpu.md)
*   [Intel Gaudi](./docs/get_started/installation/intel_gaudi.md)

**Note:** Expanding hardware support is an ongoing effort, including Ascend NPU.

## Get Started

Explore FastDeploy with these resources:

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

FastDeploy uses code from [vLLM](https://github.com/vllm-project/vllm), licensed under the [Apache-2.0 open-source license](./LICENSE).