<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h1 align="center">vLLM: Serve LLMs Easily, Quickly, and Affordably</h1>

<p align="center">
  <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
    <a href="https://github.com/vllm-project/vllm"><b>GitHub</b></a>
</p>

---

## About vLLM

vLLM is a cutting-edge open-source library designed to streamline the serving of Large Language Models (LLMs), offering significant improvements in speed, cost-efficiency, and ease of use; [check out the original repo](https://github.com/vllm-project/vllm) for more details. Developed at UC Berkeley's Sky Computing Lab and supported by a vibrant community, vLLM empowers everyone to harness the power of LLMs.

## Key Features

*   **Blazing Fast Inference:**
    *   State-of-the-art serving throughput for rapid responses.
    *   Efficient **PagedAttention** for optimized memory management.
    *   Continuous batching of incoming requests for enhanced efficiency.
    *   Fast model execution via CUDA/HIP graph.
    *   Optimized CUDA kernels, including integration with FlashAttention and FlashInfer.
    *   Speculative decoding for accelerated generation.
    *   Chunked prefill for faster initial processing.
*   **Broad Model and Hardware Support:**
    *   Seamless integration with popular Hugging Face models.
    *   Support for various decoding algorithms, including parallel sampling and beam search.
    *   Tensor, pipeline, data, and expert parallelism for distributed inference.
    *   Streaming outputs for real-time feedback.
    *   OpenAI-compatible API server for easy integration.
    *   Support for NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, TPU, and AWS Neuron.
    *   Prefix caching support for improved performance.
    *   Multi-LoRA support for flexible model customization.
*   **Quantization Support:**
    *   GPTQ, AWQ, AutoRound, INT4, INT8, and FP8 for model optimization.
*   **Extensive Model Compatibility:**
    *   Supports a wide range of models, including Transformer-like LLMs (e.g., Llama), Mixture-of-Expert LLMs (e.g., Mixtral), Embedding Models (e.g., E5-Mistral), and Multi-modal LLMs (e.g., LLaVA).

## Getting Started

Install vLLM with pip:

```bash
pip install vllm
```

Explore the documentation to get started:
* [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
* [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
* [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

vLLM welcomes contributions. Learn how to get involved: [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html).

## Sponsors

vLLM is a community project. Our compute resources for development and testing are supported by the following organizations. Thank you for your support!

<!-- Note: Please sort them in alphabetical order. -->
<!-- Note: Please keep these consistent with docs/community/sponsors.md -->
Cash Donations:
- a16z
- Dropbox
- Sequoia Capital
- Skywork AI
- ZhenFund

Compute Resources:
- AMD
- Anyscale
- AWS
- Crusoe Cloud
- Databricks
- DeepInfra
- Google Cloud
- Intel
- Lambda Lab
- Nebius
- Novita AI
- NVIDIA
- Replicate
- Roblox
- RunPod
- Trainy
- UC Berkeley
- UC San Diego

Slack Sponsor: Anyscale

We also have an official fundraising venue through [OpenCollective](https://opencollective.com/vllm). We plan to use the fund to support the development, maintenance, and adoption of vLLM.

## Citation

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

*   For technical questions and feature requests, use GitHub [Issues](https://github.com/vllm-project/vllm/issues) or [Discussions](https://github.com/vllm-project/vllm/discussions).
*   Discuss with fellow users: [vLLM Forum](https://discuss.vllm.ai).
*   Coordinate contributions and development: [Slack](https://slack.vllm.ai).
*   For security disclosures: GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories).
*   For collaborations and partnerships: [vllm-questions@lists.berkeley.edu](mailto:vllm-questions@lists.berkeley.edu).

## Media Kit

Access vLLM's logo and media assets: [Media Kit](https://github.com/vllm-project/media-kit).