<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

## vLLM: Fast, Easy, and Affordable LLM Serving

vLLM is revolutionizing LLM inference and serving, making it faster, easier, and more cost-effective for everyone; explore the project on [GitHub](https://github.com/vllm-project/vllm).

**Key Features:**

*   **Blazing Fast Inference:**
    *   State-of-the-art throughput for optimal performance.
    *   Efficient **PagedAttention** for optimized memory management.
    *   Continuous batching of requests for higher efficiency.
    *   Fast model execution leveraging CUDA/HIP graphs.
    *   Supports various quantization techniques (GPTQ, AWQ, AutoRound, INT4, INT8, FP8).
    *   Optimized CUDA kernels with FlashAttention and FlashInfer integration.
    *   Includes speculative decoding and chunked prefill capabilities.
*   **User-Friendly and Flexible:**
    *   Seamlessly integrates with popular Hugging Face models.
    *   Supports high-throughput serving with parallel sampling, beam search, and more.
    *   Offers tensor and pipeline parallelism for distributed inference.
    *   Enables streaming outputs for real-time results.
    *   Provides an OpenAI-compatible API server.
    *   Supports NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, TPU, and AWS Neuron.
    *   Offers prefix caching and multi-LoRA support.
*   **Extensive Model Support:**
    *   Compatible with most popular open-source models on Hugging Face, including:
        *   Transformer-based LLMs (e.g., Llama).
        *   Mixture-of-Experts LLMs (e.g., Mixtral, Deepseek-V2 and V3).
        *   Embedding Models (e.g., E5-Mistral).
        *   Multi-modal LLMs (e.g., LLaVA).
    *   Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

---

## Latest News

*   [2025/05] NYC vLLM Meetup: Slides available [here](https://docs.google.com/presentation/d/1_q_aW_ioMJWUImf1s1YM-ZhjXz8cUeL0IJvaquOYBeA/edit?usp=sharing).
*   [2025/05] vLLM is now a hosted project under PyTorch Foundation! Announcement [here](https://pytorch.org/blog/pytorch-foundation-welcomes-vllm/).
*   [2025/04] vLLM Asia Developer Day: Slides available [here](https://docs.google.com/presentation/d/19cp6Qu8u48ihB91A064XfaXruNYiBOUKrBxAmDOllOo/edit?usp=sharing).
*   [2025/01] vLLM V1 Alpha Release: Major architectural upgrade with a 1.7x speedup! Details [here](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html).

<details>
<summary>Previous News</summary>

- [2025/03] vLLM x Ollama Inference Night: Slides available [here](https://docs.google.com/presentation/d/16T2PDD1YwRnZ4Tu8Q5r6n53c5Lr5c73UV9Vd2_eBo4U/edit?usp=sharing).
- [2025/03] vLLM China Meetup: Slides available [here](https://docs.google.com/presentation/d/1REHvfQMKGnvz6p3Fd23HhSO4c8j5WPGZV0bKYLwnHyQ/edit?usp=sharing).
- [2025/03] East Coast vLLM Meetup: Slides available [here](https://docs.google.com/presentation/d/1NHiv8EUFF1NLd3fEYODm56nDmL26lEeXCaDgyDlTsRs/edit#slide=id.g31441846c39_0_0).
- [2025/02] Ninth vLLM meetup with Meta: Slides available [here](https://docs.google.com/presentation/d/1jzC_PZVXrVNSFVCW-V4cFXb6pn7zZ2CyP_Flwo05aqg/edit?usp=sharing) and AMD [here](https://drive.google.com/file/d/1Zk5qEJIkTmlQ2eQcXQZlljAx3m9s7nwn/view?usp=sharing).
- [2025/01] Eighth vLLM meetup with Google Cloud: Slides available [here](https://docs.google.com/presentation/d/1epVkt4Zu8Jz_S5OhEHPc798emsYh2BwYfRuDDVEF7u4/edit?usp=sharing) and Google Cloud team [here](https://drive.google.com/file/d/1h24pHewANyRL11xy5dXUbvRC9F9Kkjix/view?usp=sharing).
- [2024/12] vLLM joins [pytorch ecosystem](https://pytorch.org/blog/vllm-joins-pytorch)! Easy, Fast, and Cheap LLM Serving for Everyone!
- [2024/11] Seventh vLLM meetup with Snowflake: Slides available [here](https://docs.google.com/presentation/d/1e3CxQBV3JsfGp30SwyvS3eM_tW-ghOhJ9PAJGK6KR54/edit?usp=sharing) and Snowflake team [here](https://docs.google.com/presentation/d/1qF3RkDAbOULwz9WK5TOltt2fE9t6uIc_hVNLFAaQX6A/edit?usp=sharing).
- [2024/10] Developer Slack: [slack.vllm.ai](https://slack.vllm.ai)
- [2024/10] Ray Summit 2024 vLLM track: Talks available [here](https://www.youtube.com/playlist?list=PLzTswPQNepXl6AQwifuwUImLPFRVpksjR).
- [2024/09] Sixth vLLM meetup with NVIDIA: Slides available [here](https://docs.google.com/presentation/d/1wrLGwytQfaOTd5wCGSPNhoaW3nq0E-9wqyP7ny93xRs/edit?usp=sharing).
- [2024/07] Fifth vLLM meetup with AWS: Slides available [here](https://docs.google.com/presentation/d/1RgUD8aCfcHocghoP3zmXzck9vX3RCI9yfUAB2Bbcl4Y/edit?usp=sharing).
- [2024/07] Llama 3.1 support: Blog post [here](https://blog.vllm.ai/2024/07/23/llama31.html).
- [2024/06] Fourth vLLM meetup with Cloudflare and BentoML: Slides available [here](https://docs.google.com/presentation/d/1iJ8o7V2bQEi0BFEljLTwc5G1S10_Rhv3beed5oB0NJ4/edit?usp=sharing).
- [2024/04] Third vLLM meetup with Roblox: Slides available [here](https://docs.google.com/presentation/d/1A--47JAK4BJ39t954HyTkvtfwn0fkqtsL8NGFuslReM/edit?usp=sharing).
- [2024/01] Second vLLM meetup with IBM: Slides available [here](https://docs.google.com/presentation/d/12mI2sKABnUw5RBWXDYY-HtHth4iMSNcEoQ10jDQbxgA/edit?usp=sharing).
- [2023/10] First vLLM meetup with a16z: Slides available [here](https://docs.google.com/presentation/d/1QL-XPFXiFpDBh86DbEegFXBXFXjix4v032GhShbKf3s/edit?usp=sharing).
- [2023/08] Grant from Andreessen Horowitz (a16z).
- [2023/06] vLLM official release.

</details>

---

## About vLLM

vLLM, originally developed at UC Berkeley's Sky Computing Lab, is a leading library for fast and easy-to-use LLM inference and serving.  It has evolved into a community-driven project with significant contributions from both academia and industry.

**Performance Benchmarks:**  See how vLLM stacks up against other LLM serving engines in our [blog post](https://blog.vllm.ai/2024/09/05/perf-update.html).  The benchmark implementation is available in the [.buildkite/nightly-benchmarks/](.buildkite/nightly-benchmarks/) directory, and you can [reproduce the results](https://github.com/vllm-project/vllm/issues/8176).

---

## Getting Started

Install vLLM using `pip`:

```bash
pip install vllm
```

Or, [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source).

Comprehensive resources are available in the documentation:

*   [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
*   [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
*   [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

---

## Contributing

We encourage contributions from the community. Find out how to get involved at [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html).

---

## Sponsors

vLLM's development and testing are supported by the following organizations:

<!-- Note: Please sort them in alphabetical order. -->
<!-- Note: Please keep these consistent with docs/community/sponsors.md -->
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

We also have an official fundraising venue through [OpenCollective](https://opencollective.com/vllm). We plan to use the fund to support the development, maintenance, and adoption of vLLM.

---

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

---

## Contact Us

<!-- --8<-- [start:contact-us] -->

*   For technical questions and feature requests, use GitHub [Issues](https://github.com/vllm-project/vllm/issues) or [Discussions](https://github.com/vllm-project/vllm/discussions).
*   For discussing with fellow users, use the [vLLM Forum](https://discuss.vllm.ai).
*   For coordinating contributions and development, use [Slack](https://slack.vllm.ai).
*   For security disclosures, use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature.
*   For collaborations and partnerships, contact us at [vllm-questions@lists.berkeley.edu](mailto:vllm-questions@lists.berkeley.edu).
<!-- --8<-- [end:contact-us] -->

---

## Media Kit

Access vLLM's logo and other media assets via the [media kit repo](https://github.com/vllm-project/media-kit).