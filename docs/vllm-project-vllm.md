<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h2 align="center">vLLM: Serve LLMs Easily, Quickly, and Affordably</h2>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

---

vLLM empowers you to serve large language models (LLMs) with unparalleled speed and efficiency, making LLM inference accessible to everyone.  [Learn more at the original repo](https://github.com/vllm-project/vllm).

## Key Features of vLLM

*   **Blazing-Fast Performance:**
    *   State-of-the-art serving throughput.
    *   PagedAttention for efficient memory management.
    *   Continuous batching for handling requests.
    *   Optimized CUDA/HIP graph execution.
    *   Integration with FlashAttention and FlashInfer.
    *   Speculative decoding support.
    *   Chunked prefill.
*   **Flexible and Easy to Use:**
    *   Seamless integration with Hugging Face models.
    *   Supports popular decoding algorithms (parallel sampling, beam search, etc.).
    *   Tensor and pipeline parallelism for distributed inference.
    *   Streaming outputs.
    *   OpenAI-compatible API server.
    *   NVIDIA, AMD, Intel, PowerPC, TPU, and AWS Neuron support.
    *   Prefix caching support.
    *   Multi-LoRA support.
*   **Broad Model Compatibility:**
    *   Supports various model types, including Transformer-like LLMs (Llama), Mixture-of-Experts (Mixtral), Embedding Models (E5-Mistral), and Multi-modal LLMs (LLaVA).
    *   See the [Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html) list for details.
*   **Extensive Quantization Support:**
    *   GPTQ, AWQ, AutoRound, INT4, INT8, and FP8 quantization.

## Getting Started

Install vLLM with `pip`:

```bash
pip install vllm
```

or [from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source).

Refer to the official [documentation](https://docs.vllm.ai/en/latest/) for more information, including:
*   [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
*   [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
*   [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Performance Benchmarks

For a detailed performance comparison of vLLM against other LLM serving engines, please consult our [blog post](https://blog.vllm.ai/2024/09/05/perf-update.html).  Reproduce the benchmark yourself using our one-click script; details available [here](https://github.com/vllm-project/vllm/issues/8176).

## Contributing

Contributions and collaborations are welcome!  See the [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) documentation for guidelines.

## Sponsors

vLLM is a community project supported by:
<!-- Note: Please sort them in alphabetical order. -->
<!-- Note: Please keep these consistent with docs/community/sponsors.md -->
**Cash Donations:**
- a16z
- Dropbox
- Sequoia Capital
- Skywork AI
- ZhenFund

**Compute Resources:**
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

*   For technical questions and feature requests, use GitHub [Issues](https://github.com/vllm-project/vllm/issues) or [Discussions](https://github.com/vllm-project/vllm/discussions)
*   Discuss with fellow users: [vLLM Forum](https://discuss.vllm.ai)
*   Coordinate contributions and development: [Slack](https://slack.vllm.ai)
*   Security disclosures: GitHub [Security Advisories](https://github.com/vllm-project/vllm/security/advisories)
*   Collaborations and partnerships: [vllm-questions@lists.berkeley.edu](mailto:vllm-questions@lists.berkeley.edu)

## Media Kit

Use vLLM's logo with the [media kit repo](https://github.com/vllm-project/media-kit).