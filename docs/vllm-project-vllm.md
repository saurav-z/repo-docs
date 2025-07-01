<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h1 align="center">vLLM: Revolutionizing LLM Serving for Speed and Efficiency</h1>

<p align="center">
  <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> | <a href="https://github.com/vllm-project/vllm"><b>GitHub Repo</b></a>
</p>

---

vLLM is a powerful and user-friendly library designed to make Large Language Model (LLM) inference faster, more efficient, and more accessible for everyone; check out the [vLLM GitHub repository](https://github.com/vllm-project/vllm) for more details!

## Key Features

*   **Blazing-Fast Inference:** Achieve state-of-the-art throughput thanks to innovative techniques.
    *   **PagedAttention:** Efficiently manages attention key and value memory.
    *   **Continuous Batching:** Optimizes performance by batching incoming requests.
    *   **CUDA/HIP Graph:** Leverage fast model execution.
    *   **Optimized Kernels:** Utilizes CUDA/HIP, FlashAttention, and FlashInfer for optimal performance.
    *   **Speculative Decoding & Chunked Prefill:** Enhance inference speed.
*   **Quantization Support:** Optimize for performance and reduce resource consumption using GPTQ, AWQ, AutoRound, and INT4/INT8/FP8 quantization.
*   **Flexible and Easy to Use:** Seamlessly integrate with popular Hugging Face models and various decoding algorithms.
    *   **Parallel Sampling & Beam Search:** Offer high-throughput serving options.
    *   **Distributed Inference:** Supports tensor and pipeline parallelism.
    *   **Streaming Outputs:** Real-time results.
    *   **OpenAI-Compatible API:** Simplifies integration with existing tools.
    *   **Broad Hardware Support:** Compatible with NVIDIA, AMD, Intel, PowerPC, TPU, and AWS Neuron.
    *   **Prefix Caching & Multi-LoRA:** Enhanced functionality.
*   **Extensive Model Compatibility:** Supports a wide range of open-source models:
    *   Transformer-based LLMs (e.g., Llama)
    *   Mixture-of-Experts LLMs (e.g., Mixtral)
    *   Embedding Models (e.g., E5-Mistral)
    *   Multi-modal LLMs (e.g., LLaVA)

## Getting Started

Get up and running with vLLM quickly:

1.  **Installation:**

```bash
pip install vllm
```

2.  **Explore the Documentation:**

    *   [Installation Guide](https://docs.vllm.ai/en/latest/getting_started/installation.html)
    *   [Quickstart Guide](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
    *   [Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

Join the vLLM community and contribute to its development:

*   [Contributing Guidelines](https://docs.vllm.ai/en/latest/contributing/index.html)

## Sponsors

vLLM is a community-driven project supported by these organizations:

### Cash Donations
* a16z
* Dropbox
* Sequoia Capital
* Skywork AI
* ZhenFund

### Compute Resources
* AMD
* Anyscale
* AWS
* Crusoe Cloud
* Databricks
* DeepInfra
* Google Cloud
* Intel
* Lambda Lab
* Nebius
* Novita AI
* NVIDIA
* Replicate
* Roblox
* RunPod
* Trainy
* UC Berkeley
* UC San Diego

### Slack Sponsor
* Anyscale

Also, support vLLM's development by donating through [OpenCollective](https://opencollective.com/vllm).

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

*   **Technical Questions & Feature Requests:** [GitHub Issues](https://github.com/vllm-project/vllm/issues) or [Discussions](https://github.com/vllm-project/vllm/discussions)
*   **User Discussions:** [vLLM Forum](https://discuss.vllm.ai)
*   **Contributions & Development:** [Slack](https://slack.vllm.ai)
*   **Security Disclosures:** [GitHub Security Advisories](https://github.com/vllm-project/vllm/security/advisories)
*   **Collaborations & Partnerships:** [vllm-questions@lists.berkeley.edu](mailto:vllm-questions@lists.berkeley.edu)

## Media Kit

*   Use vLLM's logo: [Media Kit Repo](https://github.com/vllm-project/media-kit)