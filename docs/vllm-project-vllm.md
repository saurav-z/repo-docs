<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

## vLLM: The Fastest LLM Serving Engine for Everyone

vLLM is a cutting-edge library designed for efficient and cost-effective serving of Large Language Models (LLMs), empowering developers with unparalleled performance and ease of use; explore the original repository at [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm).

**Key Features of vLLM:**

*   🚀 **Blazing-Fast Inference:** Achieves state-of-the-art serving throughput through innovations like PagedAttention, continuous batching, and CUDA/HIP graph execution.
*   🧠 **Optimized Memory Management:**  Employs PagedAttention for efficient management of attention key and value memory.
*   ⚙️ **Flexible Quantization Support:** Supports various quantization techniques, including GPTQ, AWQ, AutoRound, and INT4/INT8/FP8, to optimize model size and speed.
*   🛠️ **Seamless Integration:**  Works seamlessly with popular Hugging Face models, providing a familiar and easy-to-use interface.
*   🌐 **Distributed Inference:** Supports tensor parallelism and pipeline parallelism for scaling LLM inference across multiple GPUs.
*   💻 **OpenAI-Compatible API Server:**  Offers an OpenAI-compatible API server for easy integration with existing applications.
*   💡 **Advanced Decoding Algorithms:** Implements high-throughput serving with diverse decoding algorithms such as parallel sampling, beam search, and more.
*   🔌 **Broad Hardware Compatibility:** Supports NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, TPUs, and AWS Neuron.
*   ➕ **Advanced Features:** Includes support for prefix caching and multi-LoRA.

### Key Benefits:

*   **Increased Throughput:** Optimized for maximum LLM serving throughput.
*   **Reduced Costs:** Efficient memory management and quantization techniques reduce infrastructure expenses.
*   **Easy Integration:** Simple integration with popular models and frameworks.
*   **Scalability:** Supports distributed inference for handling large workloads.

## Getting Started

Install vLLM effortlessly using pip:

```bash
pip install vllm
```

Explore the comprehensive documentation for installation, quickstart guides, and a complete list of supported models.

*   [Documentation](https://docs.vllm.ai/en/latest/)
*   [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
*   [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
*   [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

Join the vLLM community! Your contributions are highly valued. Find out more about getting involved:

*   [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html)

## Sponsors

vLLM is a community-driven project supported by generous organizations. Thank you to our sponsors:

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

*Slack Sponsor: Anyscale*

We also have an official fundraising venue through [OpenCollective](https://opencollective.com/vllm) to support development and adoption.

## Citation

If you utilize vLLM for your research, please cite our paper:

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

*   For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues) or [Discussions](https://github.com/vllm-project/vllm/discussions)
*   For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
*   For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
*   For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
*   For collaborations and partnerships, please contact us at [vllm-questions@lists.berkeley.edu](mailto:vllm-questions@lists.berkeley.edu)

## Media Kit

Access vLLM's logo and other media assets:

*   [Media Kit](https://github.com/vllm-project/media-kit)