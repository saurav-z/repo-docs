<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h1 align="center">vLLM: High-Throughput LLM Serving for Everyone</h1>

<p align="center">
    <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a>
    <br>
    <a href="https://github.com/vllm-project/vllm"><b>View on GitHub</b></a>
</p>

---

vLLM is a fast and user-friendly library for serving large language models (LLMs), enabling efficient and cost-effective deployment for everyone.

## Key Features:

*   **High Throughput:** Experience state-of-the-art serving throughput with optimized kernels and efficient memory management.
*   **PagedAttention:**  Leverage PagedAttention for efficient memory management of attention key and value with continuous batching of incoming requests.
*   **Quantization Support:** Utilize quantization techniques like GPTQ, AWQ, AutoRound, INT4, INT8, and FP8 to optimize model size and performance.
*   **Flexible and Easy to Use:** Seamlessly integrate with popular Hugging Face models, with streaming outputs and an OpenAI-compatible API server.
*   **Distributed Inference:** Supports tensor, pipeline, data, and expert parallelism for distributed inference.
*   **Model Compatibility:** Works with a wide range of models, including Transformer-like LLMs (Llama), Mixture-of-Expert LLMs (Mixtral), embedding models (E5-Mistral), and Multi-modal LLMs (LLaVA).
*   **Hardware Support:** Optimized for NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, TPU, and AWS Neuron.

## Getting Started

Install vLLM with `pip`:

```bash
pip install vllm
```

Explore the documentation for more details:

*   [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
*   [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
*   [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome community contributions!  See [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for details.

## Sponsors

vLLM is a community-driven project supported by generous organizations.  Thank you to:

### Cash Donations:
- a16z
- Dropbox
- Sequoia Capital
- Skywork AI
- ZhenFund

### Compute Resources:
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

### Slack Sponsor:
- Anyscale

We also have an official fundraising venue through [OpenCollective](https://opencollective.com/vllm).

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
*   For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
*   For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
*   For security disclosures, use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
*   For collaborations and partnerships, contact us at [vllm-questions@lists.berkeley.edu](mailto:vllm-questions@lists.berkeley.edu)

## Media Kit

Find vLLM's logo and more in the [media kit repo](https://github.com/vllm-project/media-kit).