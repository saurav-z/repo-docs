# vLLM: Fast and Easy LLM Serving

**Accelerate your Large Language Model (LLM) inference with vLLM, the cutting-edge library designed for speed, efficiency, and ease of use.** ([View the source code on GitHub](https://github.com/vllm-project/vllm))

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

---

## Key Features

*   **Blazing Fast Inference:** Achieve state-of-the-art serving throughput with optimized CUDA kernels.
*   **PagedAttention for Memory Efficiency:** Efficiently manage attention key and value memory with [PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html).
*   **Continuous Batching:** Maximize GPU utilization by continuously batching incoming requests.
*   **Quantization Support:** Supports various quantization techniques like GPTQ, AWQ, AutoRound, INT4, INT8, and FP8 for reduced memory footprint.
*   **Extensive Model Support:** Seamlessly integrates with popular Hugging Face models, including Transformer-like LLMs (e.g., Llama), Mixture-of-Expert LLMs (e.g., Mixtral), Embedding Models (e.g., E5-Mistral), and Multi-modal LLMs (e.g., LLaVA).  See the [supported models](https://docs.vllm.ai/en/latest/models/supported_models.html).
*   **Flexible Decoding Algorithms:** High-throughput serving with parallel sampling, beam search, and more.
*   **Distributed Inference:**  Supports Tensor, pipeline, data, and expert parallelism for scaling inference.
*   **OpenAI-Compatible API Server:** Easy integration with existing applications.
*   **Broad Hardware Compatibility:**  Supports NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, TPU, and AWS Neuron.
*   **Additional features**: including prefix caching, Multi-LoRA, and speculative decoding.

## About vLLM

vLLM is a powerful and user-friendly library for serving Large Language Models (LLMs), designed for high performance and ease of use. Developed at UC Berkeley's [Sky Computing Lab](https://sky.cs.berkeley.edu), vLLM is now a thriving community-driven project.  It offers significant speed improvements, memory efficiency, and flexible deployment options, making LLM serving accessible to everyone.

## Getting Started

Get started with vLLM quickly using pip:

```bash
pip install vllm
```

Detailed instructions and further information can be found in our [documentation](https://docs.vllm.ai).

*   [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
*   [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
*   [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We warmly welcome contributions and collaborations!  Learn how to get involved by visiting [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html).

## Sponsors

vLLM is a community project supported by generous organizations providing compute resources and financial contributions.  We appreciate their support!

*   **Cash Donations:** a16z, Dropbox, Sequoia Capital, Skywork AI, ZhenFund
*   **Compute Resources:** AMD, Anyscale, AWS, Crusoe Cloud, Databricks, DeepInfra, Google Cloud, Intel, Lambda Lab, Nebius, Novita AI, NVIDIA, Replicate, Roblox, RunPod, Trainy, UC Berkeley, UC San Diego

*Slack Sponsor: Anyscale*

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

*   For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues) or [Discussions](https://github.com/vllm-project/vllm/discussions)
*   For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
*   For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
*   For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
*   For collaborations and partnerships, please contact us at [vllm-questions@lists.berkeley.edu](mailto:vllm-questions@lists.berkeley.edu)

## Media Kit

For vLLM logo usage, please refer to our [media kit repo](https://github.com/vllm-project/media-kit).
```
Key improvements and SEO considerations:

*   **Clear, Concise Title:** The title now includes "Fast and Easy LLM Serving", making it more descriptive for search engines.
*   **One-Sentence Hook:** Added a compelling introductory sentence to capture the reader's attention.
*   **SEO Keywords:** Included relevant keywords like "LLM," "Large Language Model," "Inference," and "Serving" throughout the content.
*   **Organized Structure:** The README is well-structured with clear headings, making it easy to read and navigate.
*   **Bulleted Key Features:**  Uses bullet points to highlight the core benefits of vLLM, improving readability and scanability.
*   **Internal Linking:** Links within the document (e.g., to the documentation, paper, supported models) promote user engagement.
*   **Concise Language:**  Streamlined the language for better clarity and impact.
*   **Contact Information Formatting:** Improved contact information formatting for accessibility.
*   **Contextual Links:** Provided relevant links to external resources (e.g., blog, paper, Twitter) and internal pages to provide users with more context.
*   **Sponsor Information:** Keeps sponsor information updated and easy to find.
*   **Emphasis on Benefits:** The README focuses on *what* vLLM offers (speed, ease, efficiency) rather than just *what it is*.
*   **Source Code Link:** Added link back to GitHub repo.