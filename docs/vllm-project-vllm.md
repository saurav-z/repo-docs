<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h2 align="center">vLLM: Accelerate Your LLM Serving with Speed, Efficiency, and Ease!</h2>

<p align="center">
  <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

---

## About vLLM: The Ultimate LLM Serving Solution

vLLM is a powerful, open-source library designed to make Large Language Model (LLM) inference and serving fast, easy, and cost-effective for everyone.  Developed at UC Berkeley's Sky Computing Lab and now a thriving community project, vLLM empowers you to deploy and scale LLMs with unmatched efficiency.  **Learn more and contribute on the original repository:  [vLLM on GitHub](https://github.com/vllm-project/vllm)**.

### Key Features and Benefits:

*   **Blazing Fast Inference:**
    *   **PagedAttention:**  Revolutionary memory management for optimized attention key/value storage.
    *   **Continuous Batching:** Efficiently handles incoming requests in batches.
    *   **CUDA/HIP Graph Execution:** Leverages GPU capabilities for rapid model execution.
    *   **Quantization Support:** Supports GPTQ, AWQ, AutoRound, INT4, INT8, and FP8 for reduced memory footprint and faster inference.
    *   **Optimized Kernels:**  Includes integrations like FlashAttention and FlashInfer for superior performance.
    *   **Speculative Decoding and Chunked Prefill:** Accelerates text generation.

*   **Flexible and Easy to Use:**
    *   **Hugging Face Integration:** Seamlessly works with popular Hugging Face models.
    *   **Decoding Algorithms:** Supports parallel sampling, beam search, and more for diverse generation styles.
    *   **Distributed Inference:** Includes tensor and pipeline parallelism for scaling across multiple GPUs.
    *   **Streaming Outputs:**  Real-time text generation.
    *   **OpenAI-Compatible API Server:** Simple integration with existing tools.
    *   **Hardware Support:** Supports NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, TPU, and AWS Neuron.
    *   **Prefix Caching and Multi-LoRA Support:** Enhanced efficiency and flexibility.

*   **Broad Model Compatibility:**
    *   Supports a vast array of popular open-source models, including:
        *   Transformer-like LLMs (e.g., Llama)
        *   Mixture-of-Expert LLMs (e.g., Mixtral, Deepseek-V2 and V3)
        *   Embedding Models (e.g., E5-Mistral)
        *   Multi-modal LLMs (e.g., LLaVA)

    *   Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

### Latest News

*   [2025/05] We hosted [NYC vLLM Meetup](https://lu.ma/c1rqyf1f)! Please find the meetup slides [here](https://docs.google.com/presentation/d/1_q_aW_ioMJWUImf1s1YM-ZhjXz8cUeL0IJvaquOYBeA/edit?usp=sharing).
*   [2025/05] vLLM is now a hosted project under PyTorch Foundation! Please find the announcement [here](https://pytorch.org/blog/pytorch-foundation-welcomes-vllm/).
*   [2025/04] We hosted [Asia Developer Day](https://www.sginnovate.com/event/limited-availability-morning-evening-slots-remaining-inaugural-vllm-asia-developer-day)! Please find the meetup slides from the vLLM team [here](https://docs.google.com/presentation/d/19cp6Qu8u48ihB91A064XfaXruNYiBOUKrBxAmDOllOo/edit?usp=sharing).
*   [2025/01] We are excited to announce the alpha release of vLLM V1: A major architectural upgrade with 1.7x speedup! Clean code, optimized execution loop, zero-overhead prefix caching, enhanced multimodal support, and more. Please check out our blog post [here](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html).

<details>
<summary>Previous News</summary>

- [2025/03] We hosted [vLLM x Ollama Inference Night](https://lu.ma/vllm-ollama)! Please find the meetup slides from the vLLM team [here](https://docs.google.com/presentation/d/16T2PDD1YwRnZ4Tu8Q5r6n53c5Lr5c73UV9Vd2_eBo4U/edit?usp=sharing).
- [2025/03] We hosted [the first vLLM China Meetup](https://mp.weixin.qq.com/s/n77GibL2corAtQHtVEAzfg)! Please find the meetup slides from vLLM team [here](https://docs.google.com/presentation/d/1REHvfQMKGnvz6p3Fd23HhSO4c8j5WPGZV0bKYLwnHyQ/edit?usp=sharing).
- [2025/03] We hosted [the East Coast vLLM Meetup](https://lu.ma/7mu4k4xx)! Please find the meetup slides [here](https://docs.google.com/presentation/d/1NHiv8EUFF1NLd3fEYODm56nDmL26lEeXCaDgyDlTsRs/edit#slide=id.g31441846c39_0_0).
- [2025/02] We hosted [the ninth vLLM meetup](https://lu.ma/h7g3kuj9) with Meta! Please find the meetup slides from vLLM team [here](https://docs.google.com/presentation/d/1jzC_PZVXrVNSFVCW-V4cFXb6pn7zZ2CyP_Flwo05aqg/edit?usp=sharing) and AMD [here](https://drive.google.com/file/d/1Zk5qEJIkTmlQ2eQcXQZlljAx3m9s7nwn/view?usp=sharing). The slides from Meta will not be posted.
- [2025/01] We hosted [the eighth vLLM meetup](https://lu.ma/zep56hui) with Google Cloud! Please find the meetup slides from vLLM team [here](https://docs.google.com/presentation/d/1epVkt4Zu8Jz_S5OhEHPc798emsYh2BwYfRuDDVEF7u4/edit?usp=sharing), and Google Cloud team [here](https://drive.google.com/file/d/1h24pHewANyRL11xy5dXUbvRC9F9Kkjix/view?usp=sharing).
- [2024/12] vLLM joins [pytorch ecosystem](https://pytorch.org/blog/vllm-joins-pytorch)! Easy, Fast, and Cheap LLM Serving for Everyone!
- [2024/11] We hosted [the seventh vLLM meetup](https://lu.ma/h0qvrajz) with Snowflake! Please find the meetup slides from vLLM team [here](https://docs.google.com/presentation/d/1e3CxQBV3JsfGp30SwyvS3eM_tW-ghOhJ9PAJGK6KR54/edit?usp=sharing), and Snowflake team [here](https://docs.google.com/presentation/d/1qF3RkDAbOULwz9WK5TOltt2fE9t6uIc_hVNLFAaQX6A/edit?usp=sharing).
- [2024/10] We have just created a developer slack ([slack.vllm.ai](https://slack.vllm.ai)) focusing on coordinating contributions and discussing features. Please feel free to join us there!
- [2024/10] Ray Summit 2024 held a special track for vLLM! Please find the opening talk slides from the vLLM team [here](https://docs.google.com/presentation/d/1B_KQxpHBTRa_mDF-tR6i8rWdOU5QoTZNcEg2MKZxEHM/edit?usp=sharing). Learn more from the [talks](https://www.youtube.com/playlist?list=PLzTswPQNepXl6AQwifuwUImLPFRVpksjR) from other vLLM contributors and users!
- [2024/09] We hosted [the sixth vLLM meetup](https://lu.ma/87q3nvnh) with NVIDIA! Please find the meetup slides [here](https://docs.google.com/presentation/d/1wrLGwytQfaOTd5wCGSPNhoaW3nq0E-9wqyP7ny93xRs/edit?usp=sharing).
- [2024/07] We hosted [the fifth vLLM meetup](https://lu.ma/lp0gyjqr) with AWS! Please find the meetup slides [here](https://docs.google.com/presentation/d/1RgUD8aCfcHocghoP3zmXzck9vX3RCI9yfUAB2Bbcl4Y/edit?usp=sharing).
- [2024/07] In partnership with Meta, vLLM officially supports Llama 3.1 with FP8 quantization and pipeline parallelism! Please check out our blog post [here](https://blog.vllm.ai/2024/07/23/llama31.html).
- [2024/06] We hosted [the fourth vLLM meetup](https://lu.ma/agivllm) with Cloudflare and BentoML! Please find the meetup slides [here](https://docs.google.com/presentation/d/1iJ8o7V2bQEi0BFEljLTwc5G1S10_Rhv3beed5oB0NJ4/edit?usp=sharing).
- [2024/04] We hosted [the third vLLM meetup](https://robloxandvllmmeetup2024.splashthat.com/) with Roblox! Please find the meetup slides [here](https://docs.google.com/presentation/d/1A--47JAK4BJ39t954HyTkvtfwn0fkqtsL8NGFuslReM/edit?usp=sharing).
- [2024/01] We hosted [the second vLLM meetup](https://lu.ma/ygxbpzhl) with IBM! Please find the meetup slides [here](https://docs.google.com/presentation/d/12mI2sKABnUw5RBWXDYY-HtHth4iMSNcEoQ10jDQbxgA/edit?usp=sharing).
- [2023/10] We hosted [the first vLLM meetup](https://lu.ma/first-vllm-meetup) with a16z! Please find the meetup slides [here](https://docs.google.com/presentation/d/1QL-XPFXiFpDBh86DbEegFXBXFXjix4v032GhShbKf3s/edit?usp=sharing).
- [2023/08] We would like to express our sincere gratitude to [Andreessen Horowitz](https://a16z.com/2023/08/30/supporting-the-open-source-ai-community/) (a16z) for providing a generous grant to support the open-source development and research of vLLM.
- [2023/06] We officially released vLLM! FastChat-vLLM integration has powered [LMSYS Vicuna and Chatbot Arena](https://chat.lmsys.org) since mid-April. Check out our [blog post](https://vllm.ai).

</details>

## Getting Started with vLLM

Install vLLM using pip:

```bash
pip install vllm
```

Explore our comprehensive [documentation](https://docs.vllm.ai/en/latest/) to learn more:

*   [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
*   [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
*   [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing to vLLM

We welcome contributions from the community! Please see [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for guidelines.

## Sponsors

vLLM is proudly supported by the following organizations. Your contributions help make vLLM a success!

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

<!-- --8<-- [start:contact-us] -->
-   For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues) or [Discussions](https://github.com/vllm-project/vllm/discussions)
-   For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
-   For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
-   For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
-   For collaborations and partnerships, please contact us at [vllm-questions@lists.berkeley.edu](mailto:vllm-questions@lists.berkeley.edu)
<!-- --8<-- [end:contact-us] -->

## Media Kit

-   If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
```
Key improvements and SEO enhancements:

*   **Targeted Keywords:**  Used keywords like "LLM serving," "Large Language Model," "inference," "efficiency," "speed," "open source," and model names like "Llama" strategically.
*   **Clear Headings:**  Organized content with clear, descriptive headings and subheadings for readability and SEO.
*   **Concise Summary and Hook:**  The first sentence and H2 immediately grab attention and highlight the key benefit.
*   **Benefit-Driven Language:** Focused on the advantages of using vLLM (fast, easy, cost-effective).
*   **Call to Action:**  Encourages users to explore the documentation and join the community.
*   **Formatted for Readability:**  Uses bullet points for key features, making the information easy to scan.
*   **Emphasis on Open Source:** Highlights the open-source nature to attract users.
*   **Link to the Repo:**  The critical link to the original repository is now highly visible.
*   **Complete Summary:** Comprehensive, covering the original content effectively.
*   **SEO-Friendly Title and Description (not included in the output, but implied):** A good title would be something like "vLLM: Fast, Easy, and Cost-Effective LLM Serving | Open Source".  A description should highlight the benefits and keywords, like "Deploy and scale Large Language Models (LLMs) with vLLM, the open-source library for high-performance inference.  Get speed, efficiency, and cost savings."