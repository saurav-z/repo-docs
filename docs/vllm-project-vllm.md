<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h2 align="center">vLLM: Revolutionizing LLM Serving for Speed, Efficiency, and Affordability</h2>

<p align="center">
  <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

---

**vLLM** is an open-source library that makes LLM inference and serving easier, faster, and more cost-effective for everyone, offering cutting-edge performance and a user-friendly experience. ([See the original repository](https://github.com/vllm-project/vllm))

### Key Features:

*   **Blazing Fast Performance:**
    *   State-of-the-art throughput for LLM serving.
    *   **PagedAttention** for efficient memory management.
    *   Continuous batching of requests.
    *   CUDA/HIP graph for fast model execution.
    *   Quantization support (GPTQ, AWQ, AutoRound, INT4, INT8, and FP8).
    *   Optimized CUDA kernels with FlashAttention and FlashInfer integration.
    *   Speculative decoding and chunked prefill.
*   **User-Friendly & Flexible:**
    *   Seamless integration with Hugging Face models.
    *   High-throughput serving with diverse decoding algorithms (parallel sampling, beam search, etc.).
    *   Tensor and pipeline parallelism for distributed inference.
    *   Streaming output capabilities.
    *   OpenAI-compatible API server.
    *   Broad hardware support (NVIDIA, AMD, Intel, PowerPC, TPU, AWS Neuron).
    *   Prefix caching and Multi-LoRA support.
*   **Wide Model Compatibility:**
    *   Supports popular open-source models from Hugging Face:
        *   Transformer-based LLMs (e.g., Llama).
        *   Mixture-of-Experts models (e.g., Mixtral).
        *   Embedding Models (e.g., E5-Mistral).
        *   Multi-modal LLMs (e.g., LLaVA).
    *   Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

---

### Getting Started

1.  **Installation:** Install vLLM with `pip`:
    ```bash
    pip install vllm
    ```
    Or [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source).
2.  **Explore the Documentation:**
    *   [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
    *   [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
    *   [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

---

### Contributing

We welcome contributions! See [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) to get involved.

### Sponsors

vLLM is a community-driven project.  We gratefully acknowledge the support of our sponsors:

*   **Cash Donations:** a16z, Dropbox, Sequoia Capital, Skywork AI, ZhenFund
*   **Compute Resources:** AMD, Anyscale, AWS, Crusoe Cloud, Databricks, DeepInfra, Google Cloud, Intel, Lambda Lab, Nebius, Novita AI, NVIDIA, Replicate, Roblox, RunPod, Trainy, UC Berkeley, UC San Diego
*   **Slack Sponsor:** Anyscale

We also have an official fundraising venue through [OpenCollective](https://opencollective.com/vllm).

---

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

---

### Contact Us

*   **Technical Questions/Feature Requests:** GitHub [Issues](https://github.com/vllm-project/vllm/issues) or [Discussions](https://github.com/vllm-project/vllm/discussions)
*   **User Discussions:** [vLLM Forum](https://discuss.vllm.ai)
*   **Contributions/Development:** [Slack](https://slack.vllm.ai)
*   **Security Disclosures:** GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories)
*   **Collaborations/Partnerships:** [vllm-questions@lists.berkeley.edu](mailto:vllm-questions@lists.berkeley.edu)

---

### Media Kit

Find vLLM's logo and media assets in [our media kit repo](https://github.com/vllm-project/media-kit).