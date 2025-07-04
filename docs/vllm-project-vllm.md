<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h2 align="center">vLLM: The Fastest and Easiest Way to Serve Large Language Models</h2>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

---

vLLM provides a streamlined solution for serving Large Language Models (LLMs), offering exceptional speed, ease of use, and cost-effectiveness.  **[Check out the original repository](https://github.com/vllm-project/vllm) to learn more!**

### Key Features:

*   **Blazing Fast Inference:** Experience state-of-the-art throughput with PagedAttention, continuous batching, CUDA/HIP graph execution, and optimized kernels.
*   **Efficient Memory Management:** PagedAttention technology dramatically improves memory efficiency, enabling faster inference and support for larger models.
*   **Broad Hardware Support:** Run vLLM on NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, TPUs, and AWS Neuron for maximum flexibility.
*   **Model Compatibility:** Seamlessly integrate with popular Hugging Face models, including Transformers, Mixture-of-Experts, and Multi-modal LLMs.
*   **Quantization Support:** Utilize quantization techniques such as GPTQ, AWQ, AutoRound, INT4, INT8, and FP8 for reduced memory footprint and faster inference.
*   **Flexible Decoding Algorithms:** Leverage various decoding algorithms, including parallel sampling, beam search, and speculative decoding, to optimize for different use cases.
*   **Easy-to-Use API:** Utilize an OpenAI-compatible API server for simplified integration.
*   **Distributed Inference:** Supports tensor and pipeline parallelism for scaling inference across multiple GPUs.
*   **Streaming Output:** Receive results in real-time with streaming capabilities.
*   **Additional features:** Prefix caching support & Multi-LoRA support

### About

vLLM, developed at UC Berkeley's Sky Computing Lab and now a community-driven project, is your go-to solution for LLM serving. It offers a comprehensive toolkit for efficient model deployment, supporting a wide range of models and hardware configurations.

### Getting Started

Install vLLM with pip:

```bash
pip install vllm
```

Or build from source:  (See the [documentation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source) for details)

Explore the [documentation](https://docs.vllm.ai/en/latest/) for installation, quickstart guides, and a list of supported models.

### Contributing

We welcome contributions! Learn how to get involved at [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html).

### Sponsors

vLLM is a community project, supported by generous contributions from:

*   a16z
*   AMD
*   Anyscale
*   AWS
*   Crusoe Cloud
*   Databricks
*   DeepInfra
*   Dropbox
*   Google Cloud
*   Intel
*   Lambda Lab
*   Nebius
*   Novita AI
*   NVIDIA
*   Replicate
*   Roblox
*   RunPod
*   Sequoia Capital
*   Skywork AI
*   Trainy
*   UC Berkeley
*   UC San Diego
*   ZhenFund

(Slack Sponsor: Anyscale)

Support vLLM through [OpenCollective](https://opencollective.com/vllm) to help with development, maintenance, and adoption.

### Citation

If you use vLLM, please cite our paper:

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

### Contact Us

*   **Technical Questions & Feature Requests:** [GitHub Issues](https://github.com/vllm-project/vllm/issues) or [Discussions](https://github.com/vllm-project/vllm/discussions)
*   **User Discussions:** [vLLM Forum](https://discuss.vllm.ai)
*   **Contribution Coordination:** [Slack](https://slack.vllm.ai)
*   **Security Disclosures:** [GitHub Security Advisories](https://github.com/vllm-project/vllm/security/advisories)
*   **Collaborations & Partnerships:** [vllm-questions@lists.berkeley.edu](mailto:vllm-questions@lists.berkeley.edu)

### Media Kit

Access the vLLM logo and other media assets in the [media kit repo](https://github.com/vllm-project/media-kit).