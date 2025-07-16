<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

## vLLM: Serve Large Language Models (LLMs) Easily, Quickly, and Affordably

[vLLM](https://github.com/vllm-project/vllm) revolutionizes LLM inference by providing a fast, user-friendly, and cost-effective solution for serving your models.

### Key Features:

*   **Blazing Fast Performance:**
    *   State-of-the-art throughput for LLM serving.
    *   **PagedAttention:** Efficiently manages attention key and value memory.
    *   Continuous batching for efficient request processing.
    *   CUDA/HIP graph for fast model execution.
    *   Optimized CUDA kernels with FlashAttention and FlashInfer integration.
    *   Speculative decoding and Chunked prefill.
*   **Broad Model Support:**
    *   Seamless integration with popular Hugging Face models.
    *   Supports Transformer-like LLMs (e.g., Llama), Mixture-of-Expert LLMs (e.g., Mixtral), Embedding Models (e.g., E5-Mistral), and Multi-modal LLMs (e.g., LLaVA).
    *   Full list of [supported models](https://docs.vllm.ai/en/latest/models/supported_models.html).
*   **Flexible and Easy to Use:**
    *   High-throughput serving with various decoding algorithms (parallel sampling, beam search, etc.).
    *   Tensor, pipeline, data, and expert parallelism for distributed inference.
    *   Streaming output support.
    *   OpenAI-compatible API server.
    *   Supports NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, TPU, and AWS Neuron.
    *   Prefix caching support and Multi-LoRA support.
*   **Quantization Support:**
    *   GPTQ, AWQ, AutoRound, INT4, INT8, and FP8 quantization.

### Getting Started

Install vLLM with pip:

```bash
pip install vllm
```

Explore the documentation for in-depth information:

*   [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
*   [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
*   [Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

### Contributing

vLLM thrives on community contributions. Learn how to get involved in [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html).

### Sponsors

vLLM is proudly supported by a community of organizations:

*   **Cash Donations:** a16z, Dropbox, Sequoia Capital, Skywork AI, ZhenFund
*   **Compute Resources:** AMD, Anyscale, AWS, Crusoe Cloud, Databricks, DeepInfra, Google Cloud, Intel, Lambda Lab, Nebius, Novita AI, NVIDIA, Replicate, Roblox, RunPod, Trainy, UC Berkeley, UC San Diego
*   **Slack Sponsor:** Anyscale

We also have an official fundraising venue through [OpenCollective](https://opencollective.com/vllm).

### Citation

If you use vLLM in your research, please cite our paper:

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

### Contact Us

*   **Technical Questions and Feature Requests:** [GitHub Issues](https://github.com/vllm-project/vllm/issues) or [Discussions](https://github.com/vllm-project/vllm/discussions)
*   **Community Discussions:** [vLLM Forum](https://discuss.vllm.ai)
*   **Coordinating Contributions:** [Slack](https://slack.vllm.ai)
*   **Security Disclosures:** GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories)
*   **Collaborations and Partnerships:** [vllm-questions@lists.berkeley.edu](mailto:vllm-questions@lists.berkeley.edu)

### Media Kit

For vLLM logo usage, see the [media kit repo](https://github.com/vllm-project/media-kit).