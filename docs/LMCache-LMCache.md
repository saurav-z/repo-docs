<div align="center">
  <p align="center">
    <img src="https://raw.githubusercontent.com/LMCache/LMCache/dev/asset/logo.png" width="720" alt="lmcache logo">
  </p>

  [![Docs](https://img.shields.io/badge/docs-live-brightgreen)](https://docs.lmcache.ai/)
  [![PyPI](https://img.shields.io/pypi/v/lmcache)](https://pypi.org/project/lmcache/)
  [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lmcache)](https://pypi.org/project/lmcache/)
  [![Unit Tests](https://badge.buildkite.com/ce25f1819a274b7966273bfa54f0e02f092c3de0d7563c5c9d.svg)](https://buildkite.com/lmcache/lmcache-unittests)
  [![Code Quality](https://github.com/lmcache/lmcache/actions/workflows/code_quality_checks.yml/badge.svg?branch=dev&label=tests)](https://github.com/LMCache/LMCache/actions/workflows/code_quality_checks.yml)
  [![Integration Tests](https://badge.buildkite.com/108ddd4ab482a2480999dec8c62a640a3315ed4e6c4e86798e.svg)](https://buildkite.com/lmcache/lmcache-vllm-integration-tests)

   <br />

  [![OpenSSF Best Practices](https://www.bestpractices.dev/projects/10841/badge)](https://www.bestpractices.dev/projects/10841)
  [![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/LMCache/LMCache/badge)](https://scorecard.dev/viewer/?uri=github.com/LMCache/LMCache)
  [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/LMCache/LMCache/)
  [![GitHub commit activity](https://img.shields.io/github/commit-activity/w/LMCache/LMCache)](https://github.com/LMCache/LMCache/graphs/commit-activity)
  [![PyPI - Downloads](https://img.shields.io/pypi/dm/lmcache)](https://pypi.org/project/lmcache/)
  [![YouTube Channel Views](https://img.shields.io/youtube/channel/views/UC58zMz55n70rtf1Ak2PULJA)](https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA)
</div>

## LMCache: Accelerate LLM Serving with Efficient KV Cache Management

LMCache is a powerful extension designed to drastically reduce latency and boost throughput for Large Language Model (LLM) serving by intelligently caching and reusing KV caches. [**Explore the LMCache GitHub Repository**](https://github.com/LMCache/LMCache)

### Key Features:

*   **KV Cache Reusability:**  Reuses KV caches of any reused text, not just prefixes, across all serving engine instances.
*   **Reduced Latency and Increased Throughput:** Significantly reduces Time to First Token (TTFT) and improves performance, especially in long-context scenarios. Achieves up to 3-10x delay savings and GPU cycle reduction.
*   **vLLM Integration:** Seamless integration with vLLM v1, with high-performance CPU KVCache offloading, disaggregated prefill, and P2P KVCache sharing.
*   **Production Ready:**  Officially supported in the vLLM production stack, llm-d, and KServe.
*   **Non-Prefix Cache Support:** Stable support for storing and reusing non-prefix KV caches.
*   **Flexible Storage Options:** Supports various storage options including CPU, Disk, and NIXL.
*   **Easy Installation:** Simple pip installation for quick setup.

### Summary

LMCache optimizes LLM serving by caching Key-Value (KV) caches. This enables reuse of KV caches across different instances and text, leading to substantial performance gains, particularly in long-context scenarios.  LMCache integrates with vLLM to achieve significant delay savings and GPU cycle reduction.

### Installation

Install LMCache using pip:

```bash
pip install lmcache
```

See the [documentation](https://docs.lmcache.ai/getting_started/installation) for detailed installation instructions, particularly for non-latest vLLM versions or different serving engines.

### Getting Started

Get started with LMCache by checking out the [Quickstart Examples](https://docs.lmcache.ai/getting_started/quickstart/) in the documentation.

### Documentation

Comprehensive documentation is available at [https://docs.lmcache.ai/](https://docs.lmcache.ai/).  Also, check out the [LMCache blog](https://blog.lmcache.ai/) for the latest updates.

### Examples

Explore the [examples](https://github.com/LMCache/LMCache/tree/dev/examples) to learn how to address diverse use cases with LMCache.

### Connect with Us

*   **Interest Form:** [https://forms.gle/mQfQDUXbKfp2St1z7](https://forms.gle/mQfQDUXbKfp2St1z7)
*   **Newsletter:** [https://mailchi.mp/tensormesh/lmcache-sign-up-newsletter](https://mailchi.mp/tensormesh/lmcache-sign-up-newsletter)
*   **Slack:** [https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-2viziwhue-5Amprc9k5hcIdXT7XevTaQ](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-2viziwhue-5Amprc9k5hcIdXT7XevTaQ)
*   **Website:** [https://lmcache.ai/](https://lmcache.ai/)
*   **Email:** [contact@lmcache.ai](mailto:contact@lmcache.ai)

### Community Meeting

Join the bi-weekly community meeting for discussions and updates:

*   **Schedule:** Tuesdays at 9:00 AM PT - [Add to Calendar](https://drive.usercontent.google.com/u/0/uc?id=1f5EXbooGcwNwzIpTgn5u4PHqXgfypMtu&export=download)
*   **Meeting Notes:** [https://docs.google.com/document/d/1_Fl3vLtERFa3vTH00cezri78NihNBtSClK-_1tSrcow](https://docs.google.com/document/d/1_Fl3vLtERFa3vTH00cezri78NihNBtSClK-_1tSrcow)
*   **Recordings:** [YouTube LMCache channel](https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA)

### Contributing

We welcome contributions! See the [Contributing Guide](CONTRIBUTING.md) for details.
Check out [[Onboarding] Welcoming contributors with good first issues!](https://github.com/LMCache/LMCache/issues/627) for ways to contribute.

### Citation

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

### Socials

*   **LinkedIn:** [https://www.linkedin.com/company/lmcache-lab/?viewAsMember=true](https://www.linkedin.com/company/lmcache-lab/?viewAsMember=true)
*   **Twitter:** [https://x.com/lmcache](https://x.com/lmcache)
*   **YouTube:** [https://www.youtube.com/@LMCacheTeam](https://www.youtube.com/@LMCacheTeam)

### License

LMCache is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file.