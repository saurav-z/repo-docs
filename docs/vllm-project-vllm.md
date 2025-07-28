<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h1 align="center">vLLM: Fast and Easy LLM Serving for Everyone</h1>

<p align="center">
  <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> | <a href="https://github.com/vllm-project/vllm"><b>GitHub</b></a>
</p>

---

## About vLLM

**vLLM revolutionizes large language model (LLM) serving by providing a fast, efficient, and user-friendly solution for deploying and interacting with powerful AI models.**

Developed at UC Berkeley's Sky Computing Lab, vLLM is a community-driven open-source project that enables efficient LLM inference and serving. It allows you to easily deploy and use state-of-the-art LLMs, making them accessible to a wider audience.

## Key Features

*   **Blazing Fast Performance:**
    *   **PagedAttention:** Optimized memory management for efficient attention key and value storage.
    *   **Continuous Batching:**  Handles incoming requests efficiently.
    *   **CUDA/HIP Graph Execution:** Accelerates model execution.
    *   **Quantization Support:**  Includes GPTQ, AWQ, AutoRound, and INT4/INT8/FP8 quantization for model optimization.
    *   **Optimized Kernels:** Integrates with FlashAttention and FlashInfer for faster processing.
    *   **Speculative Decoding:**  Improves the speed of the decoding process.
    *   **Chunked Prefill:**  Efficiently handles the initial stages of LLM processing.
*   **Ease of Use & Flexibility:**
    *   **Hugging Face Integration:** Seamlessly works with popular Hugging Face models.
    *   **Decoding Algorithms:** Supports parallel sampling, beam search, and other decoding methods.
    *   **Distributed Inference:** Offers tensor, pipeline, data, and expert parallelism.
    *   **Streaming Outputs:** Provides real-time results.
    *   **OpenAI-Compatible API:** Offers an API server compatible with OpenAI's API for easy integration.
    *   **Hardware Support:** Supports NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, TPUs, and AWS Neuron.
    *   **Prefix Caching:** Supports prefix caching to speed up generation of text based on a prompt.
    *   **Multi-LoRA Support:** Supports Multi-LoRA for customizing responses.
*   **Model Compatibility:**
    *   Supports most popular open-source models, including Transformer-based LLMs (like Llama), Mixture-of-Experts LLMs (like Mixtral), Embedding Models (like E5-Mistral) and Multi-modal LLMs (like LLaVA).

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM easily using pip:

```bash
pip install vllm
```

Or, build from source following the instructions in the [documentation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source).

Explore the comprehensive [documentation](https://docs.vllm.ai/en/latest/) for more information:

*   [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
*   [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
*   [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

vLLM thrives on community contributions! Learn how you can get involved by checking out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html).

## Sponsors

vLLM is proudly supported by these organizations:

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

Support vLLM's development and future by donating through our [OpenCollective](https://opencollective.com/vllm).

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
*   Connect with other users on the [vLLM Forum](https://discuss.vllm.ai)
*   Coordinate contributions and development via [Slack](https://slack.vllm.ai)
*   For security disclosures, use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories)
*   For collaborations and partnerships, email us at [vllm-questions@lists.berkeley.edu](mailto:vllm-questions@lists.berkeley.edu)

## Media Kit

Find vLLM's official logo and other media assets in the [media kit repo](https://github.com/vllm-project/media-kit).