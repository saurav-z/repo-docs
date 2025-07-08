<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

## vLLM: Serve Large Language Models (LLMs) Easily, Quickly, and Affordably

[vLLM](https://github.com/vllm-project/vllm) empowers everyone to serve LLMs efficiently, offering unparalleled speed and cost-effectiveness.

**Key Features:**

*   🚀 **Blazing-Fast Inference:** Experience state-of-the-art throughput with innovations like PagedAttention, continuous batching, and CUDA/HIP graph execution.
*   💰 **Cost-Effective Serving:** Optimize resource utilization and reduce inference costs with quantization support (GPTQ, AWQ, AutoRound, and more), optimized kernels, and speculative decoding.
*   🛠️ **Easy Integration:** Seamlessly use popular Hugging Face models and enjoy features like streaming outputs, OpenAI-compatible API server, and multi-LoRA support.
*   🌐 **Versatile Compatibility:** Supports NVIDIA, AMD, Intel, PowerPC CPUs/GPUs, TPUs, and AWS Neuron, along with tensor and pipeline parallelism for distributed inference.
*   📦 **Broad Model Support:** Ready to use with most popular open-source models on Hugging Face, including Transformer-like LLMs (e.g., Llama), Mixture-of-Experts LLMs (e.g., Mixtral), Embedding Models (e.g., E5-Mistral), and Multi-modal LLMs (e.g., LLaVA).
*   ✨ **Advanced Decoding Algorithms:**  Supports parallel sampling, beam search, and chunked prefill for high-throughput serving.

**Latest News:**

*   [2025/05] vLLM is now a hosted project under PyTorch Foundation!

    *   [More announcements](https://pytorch.org/blog/pytorch-foundation-welcomes-vllm/).
*   [2025/05] NYC vLLM Meetup. Find the slides [here](https://docs.google.com/presentation/d/1_q_aW_ioMJWUImf1s1YM-ZhjXz8cUeL0IJvaquOYBeA/edit?usp=sharing).
*   [2025/04] vLLM Asia Developer Day. Find the meetup slides [here](https://docs.google.com/presentation/d/19cp6Qu8u48ihB91A064XfaXruNYiBOUKrBxAmDOllOo/edit?usp=sharing).
*   [2025/01] vLLM V1 Alpha Release: architectural upgrade with 1.7x speedup and more features! Check out the blog [here](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html).

**(See [README](https://github.com/vllm-project/vllm) for all previous news)**

---

**About vLLM**

vLLM is a fast and easy-to-use library for LLM inference and serving, originally developed at UC Berkeley's Sky Computing Lab, and is now a community-driven open-source project.

**Getting Started**

Install vLLM with pip:

```bash
pip install vllm
```

Explore the [documentation](https://docs.vllm.ai) for:

*   [Installation Guide](https://docs.vllm.ai/en/latest/getting_started/installation.html)
*   [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
*   [Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

**Contributing**

Contribute to vLLM!  Learn how at [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html).

**Sponsors**

vLLM is supported by the following organizations:

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

**Citation**

If you use vLLM in your research, please cite our paper:

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

**Contact Us**

*   **Technical Questions & Feature Requests:** [GitHub Issues](https://github.com/vllm-project/vllm/issues) or [Discussions](https://github.com/vllm-project/vllm/discussions)
*   **Community Discussions:** [vLLM Forum](https://discuss.vllm.ai)
*   **Development Coordination:** [Slack](https://slack.vllm.ai)
*   **Security Disclosures:** [GitHub Security Advisories](https://github.com/vllm-project/vllm/security/advisories)
*   **Collaborations & Partnerships:** [vllm-questions@lists.berkeley.edu](mailto:vllm-questions@lists.berkeley.edu)

**Media Kit**

Access vLLM's logo and brand assets in [the media kit repo](https://github.com/vllm-project/media-kit).