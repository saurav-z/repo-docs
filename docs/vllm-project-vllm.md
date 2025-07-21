<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

## vLLM: Serve LLMs Easily, Quickly, and Affordably

**vLLM** offers a cutting-edge, open-source solution for serving large language models (LLMs) with unparalleled speed and efficiency; explore the original repository [here](https://github.com/vllm-project/vllm).

### Key Features

*   **Blazing Fast Inference:**
    *   State-of-the-art serving throughput.
    *   Optimized CUDA kernels, including integration with FlashAttention and FlashInfer.
    *   Efficient memory management via **PagedAttention**.
    *   Continuous batching for high request throughput.
    *   Support for speculative decoding and chunked prefill.
*   **Flexible and Easy to Use:**
    *   Seamless integration with Hugging Face models.
    *   Supports parallel sampling, beam search, and more for various decoding algorithms.
    *   Includes tensor, pipeline, data, and expert parallelism.
    *   Supports streaming outputs for real-time results.
    *   Offers an OpenAI-compatible API server.
    *   Supports NVIDIA, AMD, Intel, PowerPC, TPU, and AWS Neuron hardware.
    *   Prefix caching support.
    *   Multi-LoRA support
*   **Broad Model Support:**
    *   Works with popular Transformer-like LLMs (e.g., Llama).
    *   Compatible with Mixture-of-Expert LLMs (e.g., Mixtral).
    *   Supports Embedding Models (e.g., E5-Mistral).
    *   Offers Multi-modal LLM support (e.g., LLaVA).
    *   Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).
*   **Quantization Support**:
    *   GPTQ, AWQ, AutoRound, INT4, INT8, and FP8

### Getting Started

Install vLLM using `pip`:

```bash
pip install vllm
```

For detailed instructions and further information, consult the [documentation](https://docs.vllm.ai).

*   [Installation Guide](https://docs.vllm.ai/en/latest/getting_started/installation.html)
*   [Quickstart Guide](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
*   [Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

### Contributing

We welcome contributions! For details, see [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html).

### Sponsors

vLLM is a community-driven project. We are grateful for the support of the following organizations:

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

*   **Issues and Feature Requests:** GitHub [Issues](https://github.com/vllm-project/vllm/issues) or [Discussions](https://github.com/vllm-project/vllm/discussions)
*   **Community Forum:** [vLLM Forum](https://discuss.vllm.ai)
*   **Development and Coordination:** [Slack](https://slack.vllm.ai)
*   **Security Disclosures:** GitHub [Security Advisories](https://github.com/vllm-project/vllm/security/advisories)
*   **Collaborations and Partnerships:** [vllm-questions@lists.berkeley.edu](mailto:vllm-questions@lists.berkeley.edu)

### Media Kit

Get the vLLM logo and more from our [media kit repo](https://github.com/vllm-project/media-kit).