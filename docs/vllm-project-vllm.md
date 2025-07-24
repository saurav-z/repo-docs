<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

# vLLM: Fast and Easy LLM Serving for Everyone

vLLM is a powerful library for serving Large Language Models (LLMs) that provides exceptional speed, ease of use, and cost efficiency.  Check out the original repository [here](https://github.com/vllm-project/vllm).

**Key Features:**

*   **Blazing-Fast Inference:** Achieves state-of-the-art serving throughput with optimized CUDA kernels and efficient memory management.
*   **PagedAttention:** Revolutionizes memory management for LLMs, significantly improving performance.
*   **Continuous Batching:** Dynamically batches incoming requests to maximize GPU utilization.
*   **Quantization Support:** Supports GPTQ, AWQ, AutoRound, and INT4/INT8/FP8 quantization for reduced memory footprint and faster inference.
*   **Flexible and Easy to Use:** Integrates seamlessly with popular Hugging Face models and offers a user-friendly OpenAI-compatible API server.
*   **Distributed Inference:** Supports tensor, pipeline, data, and expert parallelism for scaling to larger models.
*   **Wide Hardware Support:** Runs on NVIDIA GPUs, AMD CPUs/GPUs, Intel CPUs/GPUs, PowerPC CPUs, TPU, and AWS Neuron.
*   **Model Compatibility:** Supports a wide range of open-source models, including Transformer-like LLMs (e.g., Llama), Mixture-of-Expert LLMs (e.g., Mixtral), Embedding Models (e.g., E5-Mistral), and Multi-modal LLMs (e.g., LLaVA).

**Highlighted Advantages:**

*   **Speed:** Optimized for high-throughput LLM serving.
*   **Cost-Effective:** Reduced memory footprint and efficient resource utilization.
*   **User-Friendly:** Easy to integrate and deploy with an OpenAI-compatible API.

**Getting Started:**

Install vLLM using pip:

```bash
pip install vllm
```

Explore the [documentation](https://docs.vllm.ai) for detailed information, including:

*   [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
*   [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
*   [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

**Contribute:**

The vLLM project thrives on community contributions. Learn how to get involved by visiting [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html).

**Sponsors:**

vLLM is supported by a vibrant community of sponsors.  (List of sponsors here)

**Citation:**

If you use vLLM in your research, please cite our paper:

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

**Contact Us:**

*   **Issues & Feature Requests:** GitHub [Issues](https://github.com/vllm-project/vllm/issues) or [Discussions](https://github.com/vllm-project/vllm/discussions)
*   **Community Forum:** [vLLM Forum](https://discuss.vllm.ai)
*   **Development Coordination:** [Slack](https://slack.vllm.ai)
*   **Security Disclosures:** GitHub [Security Advisories](https://github.com/vllm-project/vllm/security/advisories)
*   **Collaborations & Partnerships:** [vllm-questions@lists.berkeley.edu](mailto:vllm-questions@lists.berkeley.edu)

**Media Kit:**

For vLLM logo usage, see the [media kit repo](https://github.com/vllm-project/media-kit).