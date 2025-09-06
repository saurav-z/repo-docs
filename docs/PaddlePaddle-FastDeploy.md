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

# FastDeploy: The Ultimate Toolkit for LLM and VLM Deployment

**Accelerate your large language models (LLMs) and visual language models (VLMs) with FastDeploy, the production-ready deployment toolkit based on PaddlePaddle.**  ([See the original repo](https://github.com/PaddlePaddle/FastDeploy))

## Key Features

*   **Production-Ready Deployment:** Out-of-the-box solutions for LLM/VLM deployment.
*   **Load-Balanced PD Disaggregation:** Industrial-grade solution featuring context caching and dynamic instance role switching to optimizes resource utilization.
*   **Unified KV Cache Transmission:** Lightweight high-performance transport library with intelligent NVLink/RDMA selection.
*   **OpenAI API & vLLM Compatibility:** Deploy with a vLLM interface.
*   **Comprehensive Quantization Support:** W8A16, W8A8, W4A16, W4A8, W2A16, FP8, and more.
*   **Advanced Acceleration Techniques:** Speculative decoding, Multi-Token Prediction (MTP) and Chunked Prefill.
*   **Multi-Hardware Support:** NVIDIA GPU, Kunlunxin XPU, Hygon DCU, Ascend NPU, Iluvatar GPU, Enflame GCU, MetaX GPU etc.

## What's New

*   **[2025-08] 🔥 Released FastDeploy v2.1:** New KV Cache scheduling strategy, expanded PD separation and CUDA Graph support, enhanced hardware support, and performance optimizations.
*   **[2025-07] FastDeploy 2.0 Inference Deployment Challenge:** Win prizes by completing tasks with ERNIE 4.5 models. [Sign up here](https://www.wjx.top/vm/meSsp3L.aspx#) and [Event details](https://github.com/PaddlePaddle/FastDeploy/discussions/2728)
*   **[2025-06] 🔥 Released FastDeploy v2.0:** Supports inference and deployment for ERNIE 4.5 and includes industrial-grade PD disaggregation.

## Requirements

*   **Operating System:** Linux
*   **Python:** 3.10 ~ 3.12

## Installation

FastDeploy supports deployment on various hardware platforms. Choose your platform for detailed instructions:

*   [NVIDIA GPU](https://paddlepaddle.github.io/FastDeploy/get_started/installation/nvidia_gpu/)
*   [Kunlunxin XPU](https://paddlepaddle.github.io/FastDeploy/get_started/installation/kunlunxin_xpu/)
*   [Iluvatar GPU](https://paddlepaddle.github.io/FastDeploy/get_started/installation/iluvatar_gpu/)
*   [Enflame GCU](https://paddlepaddle.github.io/FastDeploy/get_started/installation/Enflame_gcu/)
*   [Hygon DCU](https://paddlepaddle.github.io/FastDeploy/get_started/installation/hygon_dcu/)

**Note:**  Additional hardware support is under development (Ascend NPU, MetaX GPU).

## Get Started

*   [10-Minutes Quick Deployment](https://paddlepaddle.github.io/FastDeploy/get_started/quick_start/)
*   [ERNIE-4.5 Large Language Model Deployment](https://paddlepaddle.github.io/FastDeploy/get_started/ernie-4.5.md)
*   [ERNIE-4.5-VL Multimodal Model Deployment](https://paddlepaddle.github.io/FastDeploy/get_started/ernie-4.5-vl.md)
*   [Offline Inference Development](https://paddlepaddle.github.io/FastDeploy/get_started/offline_inference/)
*   [Online Service Deployment](https://paddlepaddle.github.io/FastDeploy/get_started/online_serving/README.md)
*   [Full Supported Models List](https://paddlepaddle.github.io/FastDeploy/get_started/supported_models/)
*   [Best Practices](https://paddlepaddle.github.io/FastDeploy/get_started/best_practices/README.md)

## Supported Models

| Model                      | Data Type                                   | PD Disaggregation | Chunked Prefill | Prefix Caching | MTP   | CUDA Graph | Maximum Context Length |
| :------------------------- | :------------------------------------------ | :----------------- | :-------------- | :------------- | :---- | :--------- | :--------------------- |
| ERNIE-4.5-300B-A47B        | BF16/WINT4/WINT8/W4A8C8/WINT2/FP8           | ✅                 | ✅              | ✅             | ✅    | ✅         | 128K                   |
| ERNIE-4.5-300B-A47B-Base   | BF16/WINT4/WINT8                            | ✅                 | ✅              | ✅             | ❌     | ✅         | 128K                   |
| ERNIE-4.5-VL-424B-A47B     | BF16/WINT4/WINT8                            | WIP                | ✅              | WIP            | ❌      | WIP        | 128K                   |
| ERNIE-4.5-VL-28B-A3B       | BF16/WINT4/WINT8                            | ❌                  | ✅              | WIP            | ❌      | WIP        | 128K                   |
| ERNIE-4.5-21B-A3B          | BF16/WINT4/WINT8/FP8                         | ❌                  | ✅              | ✅             | ✅    | ✅         | 128K                   |
| ERNIE-4.5-21B-A3B-Base     | BF16/WINT4/WINT8/FP8                         | ✅                 | ✅              | ✅             | ❌     | ✅         | 128K                   |
| ERNIE-4.5-0.3B             | BF16/WINT8/FP8                               | ✅                 | ✅              | ✅             | ❌      | ✅         | 128K                   |

## Advanced Usage

*   [Quantization](https://paddlepaddle.github.io/FastDeploy/get_started/quantization/README.md)
*   [PD Disaggregation Deployment](https://paddlepaddle.github.io/FastDeploy/get_started/features/disaggregated.md)
*   [Speculative Decoding](https://paddlepaddle.github.io/FastDeploy/get_started/features/speculative_decoding.md)
*   [Prefix Caching](https://paddlepaddle.github.io/FastDeploy/get_started/features/prefix_caching.md)
*   [Chunked Prefill](https://paddlepaddle.github.io/FastDeploy/get_started/features/chunked_prefill.md)

## Acknowledgement

FastDeploy is licensed under the [Apache-2.0 open-source license](./LICENSE). Parts of the [vLLM](https://github.com/vllm-project/vllm) code were used.