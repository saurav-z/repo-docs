<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h1 align="center">vLLM: Revolutionizing Large Language Model Serving</h1>

<p align="center">
  <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a>
</p>

---

vLLM provides an open-source inference and serving engine designed to make running large language models (LLMs) easy, fast, and cost-effective for everyone.  Explore the power of LLMs with vLLM - [visit the original repository](https://github.com/vllm-project/vllm) to get started!

## Key Features

*   **Blazing Fast Performance:**
    *   State-of-the-art serving throughput for rapid response times.
    *   **PagedAttention:** Efficient memory management for attention key and value.
    *   Continuous batching to maximize GPU utilization.
    *   Optimized CUDA/HIP kernels for rapid model execution.
    *   Support for Quantization techniques: GPTQ, AWQ, AutoRound, INT4, INT8, and FP8
    *   Integration with FlashAttention and FlashInfer for optimized performance.
    *   Speculative decoding for improved efficiency.
    *   Chunked prefill for further acceleration.
*   **User-Friendly and Flexible:**
    *   Seamless integration with popular Hugging Face models.
    *   Diverse decoding algorithms: Parallel sampling, beam search, and more.
    *   Tensor and pipeline parallelism support for distributed inference.
    *   Real-time streaming outputs for a responsive user experience.
    *   OpenAI-compatible API server for easy integration.
    *   Broad hardware compatibility: NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, TPU, and AWS Neuron.
    *   Prefix caching support for enhanced efficiency.
    *   Multi-LoRA support for advanced customization.
*   **Extensive Model Support:**
    *   Supports a wide range of open-source models including Transformer-like LLMs (e.g., Llama), Mixture-of-Expert LLMs (e.g., Mixtral), Embedding Models (e.g., E5-Mistral), and Multi-modal LLMs (e.g., LLaVA).
    *   Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM using pip:

```bash
pip install vllm
```

Or build from [source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source).

Explore the comprehensive [documentation](https://docs.vllm.ai/en/latest/) for installation, quickstart guides, and a list of supported models.

## Contributing

We encourage contributions! Check out the [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) guidelines.

## Sponsors

We are grateful for the support of our sponsors:

**Cash Donations:**
*   a16z
*   Dropbox
*   Sequoia Capital
*   Skywork AI
*   ZhenFund

**Compute Resources:**
*   AMD
*   Anyscale
*   AWS
*   Crusoe Cloud
*   Databricks
*   DeepInfra
*   Google Cloud
*   Intel
*   Lambda Lab
*   Nebius
*   Novita AI
*   NVIDIA
*   Replicate
*   Roblox
*   RunPod
*   Trainy
*   UC Berkeley
*   UC San Diego

**Slack Sponsor:** Anyscale

Support vLLM's development, maintenance, and adoption through our [OpenCollective](https://opencollective.com/vllm).

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

*   For technical questions and feature requests, use GitHub [Issues](https://github.com/vllm-project/vllm/issues) or [Discussions](https://github.com/vllm-project/vllm/discussions).
*   Discuss with fellow users on the [vLLM Forum](https://discuss.vllm.ai).
*   Coordinate contributions and development on [Slack](https://slack.vllm.ai).
*   Report security disclosures via GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories).
*   For collaborations and partnerships, contact us at [vllm-questions@lists.berkeley.edu](mailto:vllm-questions@lists.berkeley.edu).

## Media Kit

Access vLLM's logo and other media assets in our [media kit repo](https://github.com/vllm-project/media-kit).