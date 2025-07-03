<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h1 align="center">vLLM: Revolutionizing LLM Serving for Speed, Efficiency, and Cost Savings</h1>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

---

vLLM is an open-source library that makes serving large language models (LLMs) easier, faster, and more affordable for everyone; [check out the original repository](https://github.com/vllm-project/vllm).

## Key Features of vLLM:

*   🚀 **Blazing-Fast Inference**: Experience state-of-the-art throughput thanks to innovative techniques like PagedAttention, continuous batching, and CUDA/HIP graph execution.
*   💰 **Cost-Effective LLM Serving**: Minimize expenses with optimized kernels, quantization support (GPTQ, AWQ, AutoRound, INT4/8, FP8), and efficient memory management.
*   🛠️ **Seamless Integration**: Easily integrate with popular Hugging Face models and utilize a variety of decoding algorithms, including parallel sampling and beam search.
*   🌐 **Distributed Inference Capabilities**: Leverage tensor and pipeline parallelism for distributed inference across multiple GPUs.
*   🔌 **Versatile and User-Friendly**: Utilize an OpenAI-compatible API server, streaming outputs, and broad hardware support (NVIDIA, AMD, Intel, PowerPC, TPU, and AWS Neuron).
*   ✅ **Wide Model Compatibility**: Works with most popular open-source models on HuggingFace, including Transformer-like LLMs (e.g., Llama), Mixture-of-Expert LLMs (e.g., Mixtral), and Multi-modal LLMs (e.g., LLaVA).

## About vLLM

vLLM, developed at UC Berkeley's Sky Computing Lab and now community-driven, is designed to accelerate and simplify the deployment of LLMs.  It delivers exceptional performance through PagedAttention and other cutting-edge optimizations.  From research to production, vLLM provides the tools you need to serve LLMs efficiently and cost-effectively.

## Getting Started

Install vLLM using pip:

```bash
pip install vllm
```

Visit the [documentation](https://docs.vllm.ai/en/latest/) for installation, quickstart guides, and a list of supported models.

## Contributing

We welcome contributions! Review our [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) guide.

## Sponsors

Thank you to our sponsors for their support!  (Sorted alphabetically - see original for full list).

## Citation

If you use vLLM for research, please cite our paper:

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

*   **GitHub Issues & Discussions:** For technical questions and feature requests, use [Issues](https://github.com/vllm-project/vllm/issues) or [Discussions](https://github.com/vllm-project/vllm/discussions).
*   **Forum:** Discuss with other users at the [vLLM Forum](https://discuss.vllm.ai).
*   **Slack:** Coordinate contributions and development on [Slack](https://slack.vllm.ai).
*   **Security Advisories:** Report security disclosures through GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories).
*   **Collaborations:** Contact us at [vllm-questions@lists.berkeley.edu](mailto:vllm-questions@lists.berkeley.edu).

## Media Kit

Find vLLM's logo and other assets in the [media kit repo](https://github.com/vllm-project/media-kit).