<div align="center">
  <p align="center">
    <img src="https://raw.githubusercontent.com/LMCache/LMCache/dev/asset/logo.png" width="720" alt="lmcache logo">
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

<br />
**[LMCache](https://github.com/LMCache/LMCache) significantly speeds up Large Language Model (LLM) serving by intelligently caching and reusing KV caches.**

## Key Features

*   **Speed & Efficiency:** Reduces Time-To-First-Token (TTFT) and boosts throughput, especially in long-context scenarios.
*   **Cache Reuse:**  Stores KV caches across various locations (GPU, CPU, Disk) to reuse cached data for any reused text, regardless of the engine instance.
*   **vLLM Integration:** Seamlessly integrates with vLLM v1, offering high-performance CPU KVCache offloading, disaggregated prefill, and P2P KVCache sharing.
*   **Production Ready:**  Supported in the vLLM production stack, llm-d, and KServe.
*   **Non-Prefix Support:**  Offers stable support for non-prefix KV caches.
*   **Storage Flexibility:** Supports CPU, Disk, and [NIXL](https://github.com/ai-dynamo/nixl) storage options.
*   **Easy Installation:** Simple installation via pip.

##  What is LMCache?

LMCache is a powerful LLM serving engine extension designed to optimize performance and reduce latency. It drastically improves efficiency, making it ideal for use cases like multi-round QA and RAG. By caching reusable text's KV caches in various locations, LMCache significantly reduces GPU cycles and accelerates response times.  Experience 3-10x delay savings and GPU cycle reduction when combining LMCache with vLLM.

## Installation

Install LMCache using pip:

```bash
pip install lmcache
```

For detailed instructions, see the [installation guide](https://docs.lmcache.ai/getting_started/installation) in the documentation.

## Getting Started

Explore the [Quickstart Examples](https://docs.lmcache.ai/getting_started/quickstart/) in our documentation to get up and running quickly.

## Documentation

Comprehensive documentation is available at [https://docs.lmcache.ai/](https://docs.lmcache.ai/).

Stay updated with our [blog](https://blog.lmcache.ai/).

## Examples

Find practical examples at [https://github.com/LMCache/LMCache/tree/dev/examples](https://github.com/LMCache/LMCache/tree/dev/examples) to learn how to apply LMCache in various scenarios.

##  Connect with Us

*   [Interest Form](https://forms.gle/MHwLiYDU6kcW3dLj7)
*   [Newsletter Sign-Up](https://mailchi.mp/tensormesh/lmcache-sign-up-newsletter)
*   [Slack](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-2viziwhue-5Amprc9k5hcIdXT7XevTaQ)
*   [Website](https://lmcache.ai/)
*   [Email](contact@lmcache.ai)

## Community Meetings

Join our bi-weekly community meetings:

*   Held bi-weekly on Tuesdays at 9:00 AM PT - [Add to Calendar](https://drive.usercontent.google.com/u/0/uc?id=1f5EXbooGcwNwzIpTgn5u4PHqXgfypMtu&export=download)
*   Meeting Notes: [Google Docs](https://docs.google.com/document/d/1_Fl3vLtERFa3vTH00cezri78NihNBtSClK-_1tSrcow)
*   Recordings: [YouTube Channel](https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA)

## Contributing

We welcome contributions!  See our [Contributing Guide](CONTRIBUTING.md) for details.

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

## Social Media

*   [LinkedIn](https://www.linkedin.com/company/lmcache-lab/?viewAsMember=true)
*   [Twitter](https://x.com/lmcache)
*   [YouTube](https://www.youtube.com/@LMCacheTeam)

## License

LMCache is licensed under the Apache License 2.0.  See the [LICENSE](LICENSE) file for details.