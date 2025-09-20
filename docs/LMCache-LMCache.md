<div align="center">
  <p align="center">
    <img src="https://raw.githubusercontent.com/LMCache/LMCache/dev/asset/logo.png" width="720" alt="LMCache Logo">
  </p>
</div>

[![Docs](https://img.shields.io/badge/docs-live-brightgreen)](https://docs.lmcache.ai/)
[![PyPI](https://img.shields.io/pypi/v/lmcache)](https://pypi.org/project/lmcache/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lmcache)](https://pypi.org/project/lmcache/)
[![Unit Tests](https://badge.buildkite.com/ce25f1819a274b7966273bfa54f0e02f092c3de0d7563c5c9d.svg)](https://buildkite.com/lmcache/lmcache-unittests)
[![Code Quality](https://github.com/lmcache/lmcache/actions/workflows/code_quality_checks.yml/badge.svg?branch=dev&label=tests)](https://github.com/LMCache/LMCache/actions/workflows/code_quality_checks.yml)
[![Integration Tests](https://badge.buildkite.com/108ddd4ab482a2480999dec8c62a640a3315ed4e6c4e86798e.svg)](https://buildkite.com/lmcache/lmcache-vllm-integration-tests)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/10841/badge)](https://www.bestpractices.dev/projects/10841)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/LMCache/LMCache/badge)](https://scorecard.dev/viewer/?uri=github.com/LMCache/LMCache)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/LMCache/LMCache/)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/w/LMCache/LMCache)](https://github.com/LMCache/LMCache/graphs/commit-activity)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/lmcache)](https://pypi.org/project/lmcache/)
[![YouTube Channel Views](https://img.shields.io/youtube/channel/views/UC58zMz55n70rtf1Ak2PULJA)](https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA)

---

[**Blog**](https://blog.lmcache.ai/) | [**Documentation**](https://docs.lmcache.ai/) | [**Join Slack**](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-3bgx768yd-H8WkOTmPtbxVYJ5nuZ4dmA) | [**Interest Form**](https://forms.gle/MHwLiYDU6kcW3dLj7) | [**Roadmap**](https://github.com/LMCache/LMCache/issues/1253)

🔥 **LMCache: Accelerate your LLM applications with intelligent caching to reduce latency and costs.**

## Key Features

*   **Optimized LLM Performance:** Significantly reduces Time-to-First-Token (TTFT) and boosts throughput, especially for long-context scenarios.
*   **KV Cache Reuse:** Efficiently caches and reuses KV caches of any reused text across different serving engine instances.
*   **Integration with vLLM:** Seamlessly integrates with vLLM v1, including CPU KVCache offloading, disaggregated prefill, and P2P KVCache sharing.
*   **Versatile Storage Options:** Supports caching on CPU, disk, and NIXL for flexible deployment.
*   **Wide Compatibility:**  Supported in the [vLLM production stack](https://github.com/vllm-project/production-stack/), [llm-d](https://github.com/llm-d/llm-d/), and [KServe](https://github.com/kserve).
*   **Non-Prefix KV Cache Support:**  Stable support for non-prefix KV caches, enhancing flexibility.
*   **Easy Installation:**  Simple installation via pip, making it easy to integrate into your existing projects.

## What is LMCache?

LMCache is an innovative LLM serving engine extension designed to drastically improve performance. It tackles the challenges of long-context scenarios by intelligently caching KV caches of reusable text across various locations (GPU, CPU DRAM, Local Disk).  This approach allows LMCache to reuse KV caches for *any* reused text, not just prefixes, leading to significant GPU cycle savings and reduced user response times. By combining LMCache with vLLM, developers can achieve substantial delay savings (3-10x) and GPU cycle reductions in LLM use cases like multi-round QA and RAG.

## Installation

Get started with LMCache quickly:

```bash
pip install lmcache
```

Detailed [installation instructions](https://docs.lmcache.ai/getting_started/installation) are available in the documentation, particularly for users not using the latest vLLM version.

## Getting Started

Explore the [Quickstart Examples](https://docs.lmcache.ai/getting_started/quickstart/) in our documentation to begin using LMCache.

## Documentation

Access comprehensive documentation on the official [LMCache documentation](https://docs.lmcache.ai/) site. Stay updated with the latest developments and insights through the [LMCache blog](https://blog.lmcache.ai/).

## Examples

Explore practical [examples](https://github.com/LMCache/LMCache/tree/dev/examples) to understand various use cases.

## Contributing

We welcome contributions!  Refer to the [Contributing Guide](CONTRIBUTING.md) for details.  See [[Onboarding] Welcoming contributors with good first issues!](https://github.com/LMCache/LMCache/issues/627)

## Citation

If you use LMCache in your research, please cite our papers:

```
@inproceedings{liu2024cachegen,
  title={Cachegen: Kv cache compression and streaming for fast large language model serving},
  author={Liu, Yuhan and Li, Hanchen and Cheng, Yihua and Ray, Siddhant and Huang, Yuyang and Zhang, Qizheng and Du, Kuntai and Yao, Jiayi and Lu, Shan and Ananthanarayanan, Ganesh and others},
  booktitle={Proceedings of the ACM SIGCOMM 2024 Conference},
  pages={38--56},
  year={2024}
}

@article{cheng2024large,
  title={Do Large Language Models Need a Content Delivery Network?},
  author={Cheng, Yihua and Du, Kuntai and Yao, Jiayi and Jiang, Junchen},
  journal={arXiv preprint arXiv:2409.13761},
  year={2024}
}

@inproceedings{10.1145/3689031.3696098,
  author = {Yao, Jiayi and Li, Hanchen and Liu, Yuhan and Ray, Siddhant and Cheng, Yihua and Zhang, Qizheng and Du, Kuntai and Lu, Shan and Jiang, Junchen},
  title = {CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion},
  year = {2025},
  url = {https://doi.org/10.1145/3689031.3696098},
  doi = {10.1145/3689031.3696098},
  booktitle = {Proceedings of the Twentieth European Conference on Computer Systems},
  pages = {94–109},
}
```

## Connect with Us

*   [LinkedIn](https://www.linkedin.com/company/lmcache-lab/?viewAsMember=true)
*   [Twitter](https://x.com/lmcache)
*   [YouTube](https://www.youtube.com/@LMCacheTeam)

## License

LMCache is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

---
[Back to the Top](#) - View the original repository on [GitHub](https://github.com/LMCache/LMCache)
```
Key improvements and SEO considerations:

*   **Clear, Concise Hook:**  Immediately grabs attention.
*   **Keyword Optimization:**  Uses relevant keywords like "LLM," "caching," "latency," "throughput," "vLLM," and related terms throughout the text.
*   **Well-Defined Headings:**  Organizes the information for readability and SEO.
*   **Bulleted Key Features:**  Easy to scan and highlights the value proposition.
*   **Detailed Description:** Expands on the value proposition, explaining what LMCache does and why it's beneficial.
*   **Strong Call to Action:** Encourages users to explore the documentation, examples, and community.
*   **Internal Linking:**  Uses links within the README to direct the user to all the relevant resources.
*   **Back to Top Link:** Improves navigation
*   **GitHub Link at the End:**  Clear link to the original repository.