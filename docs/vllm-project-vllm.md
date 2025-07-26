<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h1 align="center">vLLM: High-Throughput LLM Serving for Everyone</h1>

<p align="center">
  <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
  <a href="https://github.com/vllm-project/vllm"><b>GitHub Repo</b></a>
</p>

---

## About vLLM

vLLM revolutionizes LLM inference by providing a fast, easy-to-use, and cost-effective solution for serving large language models.  Originally developed at UC Berkeley's Sky Computing Lab, vLLM is a community-driven project continuously improving and innovating.

## Key Features

*   **Blazing Fast Performance:**
    *   State-of-the-art serving throughput for optimal LLM performance.
    *   **PagedAttention** for efficient memory management.
    *   Continuous batching of incoming requests.
    *   Optimized CUDA/HIP graph execution.
    *   Support for various quantization techniques: GPTQ, AWQ, AutoRound, INT4, INT8, and FP8.
    *   Optimized CUDA kernels, including integration with FlashAttention and FlashInfer.
    *   Speculative decoding for faster generation.
    *   Chunked prefill for efficient processing.

*   **Easy to Use and Flexible:**
    *   Seamless integration with popular Hugging Face models.
    *   High-throughput serving with parallel sampling, beam search, and more.
    *   Support for tensor, pipeline, data, and expert parallelism for distributed inference.
    *   Real-time streaming outputs.
    *   OpenAI-compatible API server.
    *   Supports NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, TPU, and AWS Neuron.
    *   Prefix caching support.
    *   Multi-LoRA support.

*   **Wide Model Compatibility:**
    *   Supports most popular open-source models on Hugging Face, including:
        *   Transformer-like LLMs (e.g., Llama)
        *   Mixture-of-Expert LLMs (e.g., Mixtral, Deepseek-V2 and V3)
        *   Embedding Models (e.g., E5-Mistral)
        *   Multi-modal LLMs (e.g., LLaVA)
    *   [Full List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM quickly with pip:

```bash
pip install vllm
```

Explore the following resources to get started:

*   [Installation Guide](https://docs.vllm.ai/en/latest/getting_started/installation.html)
*   [Quickstart Guide](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
*   [Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome contributions! See the [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) guide to get involved.

## Sponsors

vLLM is supported by the following organizations.  Thank you for your support!

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

If you use vLLM in your research, please cite our paper:

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

*   **For technical questions and feature requests:**  Use GitHub [Issues](https://github.com/vllm-project/vllm/issues) or [Discussions](https://github.com/vllm-project/vllm/discussions)
*   **For discussions with fellow users:** Use the [vLLM Forum](https://discuss.vllm.ai)
*   **For coordinating contributions and development:** Use [Slack](https://slack.vllm.ai)
*   **For security disclosures:**  Use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
*   **For collaborations and partnerships:** Contact us at [vllm-questions@lists.berkeley.edu](mailto:vllm-questions@lists.berkeley.edu)

## Media Kit

Refer to the [vLLM media kit repo](https://github.com/vllm-project/media-kit) to get the logo.