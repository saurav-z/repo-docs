<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h1 align="center">vLLM: Fast and Efficient LLM Serving</h1>

<p align="center">
  <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a>
</p>

---

## About vLLM

**vLLM revolutionizes Large Language Model (LLM) serving by offering a fast, easy-to-use, and cost-effective solution for everyone.** This powerful library, born from the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley and now a thriving community project, significantly accelerates LLM inference, making it ideal for both research and production environments.

**Key Features of vLLM:**

*   ⚡ **Blazing Fast Inference:** Achieves state-of-the-art throughput with innovative techniques.
*   🧠 **PagedAttention:** Efficiently manages attention key and value memory.
*   🔄 **Continuous Batching:** Optimizes performance by batching incoming requests dynamically.
*   ⚙️ **CUDA/HIP Graph Execution:** Leverages fast model execution with CUDA/HIP graphs.
*   🧮 **Quantization Support:** Supports GPTQ, AWQ, AutoRound, INT4, INT8, and FP8 for reduced memory usage and faster inference.
*   🚀 **Optimized Kernels:** Includes optimized CUDA kernels, with integrations like FlashAttention and FlashInfer.
*   🔍 **Speculative Decoding & Chunked Prefill:** Enhances speed with speculative decoding and chunked prefill.
*   🫂 **Hugging Face Integration:** Seamlessly integrates with popular Hugging Face models.
*   🌐 **Decoding Algorithms:** Supports various decoding algorithms, including *parallel sampling* and *beam search*.
*   💻 **Distributed Inference:** Offers tensor and pipeline parallelism for distributed inference.
*   📤 **Streaming Outputs:** Provides streaming outputs for a responsive user experience.
*   🌍 **OpenAI-Compatible API:** Offers an OpenAI-compatible API server.
*   🖥️ **Hardware Support:** Supports NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, TPU, and AWS Neuron.
*   💾 **Prefix Caching & Multi-LoRA Support:** Improves efficiency with prefix caching and multi-LoRA support.
*   ✅ **Wide Model Support:** Supports many models on Hugging Face, including Transformer-like LLMs, Mixture-of-Expert LLMs, Embedding Models, and Multi-modal LLMs.

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM using pip:

```bash
pip install vllm
```

Or build from source: [Installation from Source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source)

Explore the resources below to dive deeper:

*   [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
*   [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
*   [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Performance

vLLM offers top-tier LLM serving performance.  See the latest benchmarks, comparing vLLM to other LLM serving engines, in the [vLLM blog](https://blog.vllm.ai/2024/09/05/perf-update.html).

## Contributing

We invite contributions from everyone!  See [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for details on how to get involved.

## Sponsors

vLLM is a community-driven project.  We are thankful for the support from the following organizations:

**Cash Donations:**
- a16z
- Dropbox
- Sequoia Capital
- Skywork AI
- ZhenFund

**Compute Resources:**
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

**Slack Sponsor:** Anyscale

We also have an official fundraising venue through [OpenCollective](https://opencollective.com/vllm). We plan to use the fund to support the development, maintenance, and adoption of vLLM.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

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

Download vLLM's logo and other media assets from [our media kit repo](https://github.com/vllm-project/media-kit).

[Go back to the top](https://github.com/vllm-project/vllm)
```
Key improvements and explanations:

*   **SEO Optimization:**  Includes relevant keywords throughout (LLM, large language model, inference, serving, fast, efficient, etc.).  Uses headings (H1, H2, H3) to structure the content, which is crucial for SEO.  The title and introduction are designed to attract attention and clearly state what vLLM does.
*   **One-Sentence Hook:** The introductory sentence is now "vLLM revolutionizes Large Language Model (LLM) serving by offering a fast, easy-to-use, and cost-effective solution for everyone," which clearly describes the core value proposition.
*   **Clear Structure:**  Uses headings and bullet points to make the information easily scannable and readable, which is good for both users and search engines.
*   **Concise Summary:**  The "About" section provides a focused overview, highlighting the key benefits and features.  The key features are presented in bullet points for clarity.
*   **Call to Action:** Encourages users to "Explore the resources below to dive deeper".
*   **Emphasis on Performance:** Explicitly mentions performance and links to the blog post with benchmarks.
*   **Complete Information:**  Includes all important links, contact information, and the citation.
*   **Clean Formatting:**  Uses markdown correctly for better rendering on GitHub.
*   **Community Focused:** Emphasizes the community aspect of vLLM.
*   **Removed outdated News:** Previous news was removed, as requested.
*   **Added Back to Top:** Added a link back to the top, as requested.
*   **Clear Hardware Support:** Includes a list of hardware supported for LLM serving.