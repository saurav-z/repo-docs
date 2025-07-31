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

## LMCache: Accelerate LLM Inference with Efficient KV Cache Management

LMCache dramatically speeds up Large Language Model (LLM) inference by efficiently caching and reusing KV (key-value) caches. Explore the [LMCache repository](https://github.com/LMCache/LMCache) for more details!

### Key Features

*   **Reduced TTFT & Increased Throughput:** Significantly lowers Time-To-First-Token (TTFT) and boosts throughput, especially in long-context scenarios.
*   **KV Cache Reuse:**  Reuses KV caches of any reused text across serving engine instances, saving GPU cycles.
*   **Integration with vLLM:** Provides high-performance CPU KVCache offloading, disaggregated prefill, and P2P KVCache sharing when used with vLLM.
*   **Broad Support:** Officially supported in the [vLLM production stack](https://github.com/vllm-project/production-stack/), [llm-d](https://github.com/llm-d/llm-d/), and [KServe](https://github.com/kserve/kserve).
*   **Non-Prefix KV Cache Support:** Stable support for non-prefix KV caches.
*   **Flexible Storage Options:** Supports CPU, Disk, and [NIXL](https://github.com/ai-dynamo/nixl) for KV cache storage.

### Installation

Install LMCache with pip:

```bash
pip install lmcache
```

Requires a Linux NVIDIA GPU platform. For detailed instructions, refer to the [Installation Guide](https://docs.lmcache.ai/getting_started/installation) in the documentation.

### Getting Started

Explore our [Quickstart Examples](https://docs.lmcache.ai/getting_started/quickstart/) in the documentation to begin using LMCache.

### Documentation

Comprehensive documentation is available at [https://docs.lmcache.ai/](https://docs.lmcache.ai/).

### Examples

Check out the [examples](https://github.com/LMCache/LMCache/tree/dev/examples) to learn how to apply LMCache to various use cases.

### Stay Connected

*   **Website:** [https://lmcache.ai/](https://lmcache.ai/)
*   **Blog:** [https://blog.lmcache.ai/](https://blog.lmcache.ai/)
*   **Join our Slack:** [https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-36x1m765z-8FgDA_73vcXtlZ_4XvpE6Q](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-36x1m765z-8FgDA_73vcXtlZ_4XvpE6Q)
*   **Newsletter:** [https://mailchi.mp/tensormesh/lmcache-sign-up-newsletter](https://mailchi.mp/tensormesh/lmcache-sign-up-newsletter)
*   **Interest Form:** [https://forms.gle/MHwLiYDU6kcW3dLj7](https://forms.gle/MHwLiYDU6kcW3dLj7)
*   **Contact:** contact@lmcache.ai

### Community

*   **Community Meeting:** Join the bi-weekly community meeting (Tuesdays, 9:00 AM PT) - [Add to Calendar](https://drive.usercontent.google.com/u/0/uc?id=1f5EXbooGcwNwzIpTgn5u4PHqXgfypMtu&export=download)
*   **Meeting Notes:** [https://docs.google.com/document/d/1_Fl3vLtERFa3vTH00cezri78NihNBtSClK-_1tSrcow](https://docs.google.com/document/d/1_Fl3vLtERFa3vTH00cezri78NihNBtSClK-_1tSrcow)
*   **YouTube:** [https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA](https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA)

### Contributing

Contributions are welcome! See our [Contributing Guide](CONTRIBUTING.md) for details.

### Citation

If you use LMCache in your research, please cite the following papers:

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

LMCache is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.