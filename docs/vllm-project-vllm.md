<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h1 align="center">vLLM: Revolutionizing LLM Inference with Speed, Efficiency, and Ease</h1>

<p align="center">
  <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |  <a href="https://github.com/vllm-project/vllm"><b>GitHub Repo</b></a>
</p>

---

## About vLLM

**vLLM is the go-to library for fast, cost-effective, and user-friendly Large Language Model (LLM) inference and serving, making it accessible to everyone.** Originally developed at UC Berkeley's Sky Computing Lab, vLLM is now a thriving community-driven project.

## Key Features

*   **Blazing Fast Performance:**
    *   State-of-the-art serving throughput.
    *   Efficient memory management via **PagedAttention**.
    *   Continuous batching for optimal resource utilization.
    *   CUDA/HIP graph for rapid model execution.
    *   Optimized CUDA kernels including FlashAttention and FlashInfer integration.
    *   Speculative decoding for further speedups.
    *   Chunked prefill for streamlined processing.
*   **Flexible & Easy to Use:**
    *   Seamless integration with Hugging Face models.
    *   High-throughput serving with diverse decoding algorithms (parallel sampling, beam search, and more).
    *   Support for tensor, pipeline, data and expert parallelism, facilitating distributed inference.
    *   Streaming output for real-time results.
    *   OpenAI-compatible API server.
    *   Support for NVIDIA, AMD, Intel, PowerPC, TPU and AWS Neuron.
    *   Prefix caching support.
    *   Multi-LoRA support.
*   **Broad Model Compatibility:**
    *   Supports most popular open-source models on Hugging Face, including:
        *   Transformer-based LLMs (e.g., Llama)
        *   Mixture-of-Experts LLMs (e.g., Mixtral)
        *   Embedding Models (e.g., E5-Mistral)
        *   Multi-modal LLMs (e.g., LLaVA)
    *   Full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).
*   **Advanced Quantization Support:**  Optimize model size and speed with:
    *   GPTQ
    *   AWQ
    *   AutoRound
    *   INT4, INT8, and FP8

## Getting Started

Install vLLM easily with pip:

```bash
pip install vllm
```

Or build from source:  See [installation instructions](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source)

Explore our documentation to get started:
*   [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
*   [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
*   [Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome contributions!  See [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for details.

## Sponsors

vLLM is a community-driven project supported by the following organizations:

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

*   For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues) or [Discussions](https://github.com/vllm-project/vllm/discussions)
*   For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
*   For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
*   For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
*   For collaborations and partnerships, please contact us at [vllm-questions@lists.berkeley.edu](mailto:vllm-questions@lists.berkeley.edu)

## Media Kit

For vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit).