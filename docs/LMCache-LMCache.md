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

## LMCache: Supercharge Your LLM Performance with Intelligent Caching

LMCache is an LLM serving engine extension that dramatically reduces the Time-To-First-Token (TTFT) and boosts throughput, especially in long-context scenarios.

**[Explore the LMCache repository](https://github.com/LMCache/LMCache)**

| [**Blog**](https://blog.lmcache.ai/) | [**Documentation**](https://docs.lmcache.ai/) | [**Join Slack**](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-36x1m765z-8FgDA_73vcXtlZ_4XvpE6Q) | [**Interest Form**](https://forms.gle/MHwLiYDU6kcW3dLj7) | [**Roadmap**](https://github.com/LMCache/LMCache/issues/1253) |
---

🔥 **NEW: For enterprise-scale deployment of LMCache and vLLM, please check out vLLM [Production Stack](https://github.com/vllm-project/production-stack). LMCache is also officially supported in [llm-d](https://github.com/llm-d/llm-d/) and [KServe](https://github.com/kserve/kserve)!**

## Key Features of LMCache

*   **Accelerated LLM Serving:** Reduce TTFT and increase throughput by leveraging KV cache reuse.
*   **KV Cache Reuse:** Efficiently reuse KV caches of any reused text, not just prefixes, across various serving instances.
*   **vLLM Integration:** Seamlessly integrates with vLLM for enhanced performance.
    *   High performance CPU KVCache offloading
    *   Disaggregated prefill
    *   P2P KVCache sharing
*   **Production Ready:** Officially supported in the [vLLM production stack](https://github.com/vllm-project/production-stack/), [llm-d](https://github.com/llm-d/llm-d/), and [KServe](https://github.com/kserve/kserve)
*   **Non-Prefix KV Cache Support:** Stable support for non-prefix KV caches, enhancing flexibility.
*   **Flexible Storage Options:** Supports caching on CPU, Disk, and [NIXL](https://github.com/ai-dynamo/nixl).

## Installation

Get started with LMCache in a few steps:

```bash
pip install lmcache
```

For more detailed installation instructions, especially for specific environments or configurations, please refer to the [installation guide](https://docs.lmcache.ai/getting_started/installation) in our documentation.  Troubleshooting for "undefined symbol" or torch mismatch versions can also be found in the documentation.

## Getting Started

Dive into LMCache with our [Quickstart Examples](https://docs.lmcache.ai/getting_started/quickstart/) to learn how to integrate and optimize your LLM applications.

## Documentation

Comprehensive documentation is available online at [https://docs.lmcache.ai/](https://docs.lmcache.ai/).

## Examples

Explore practical use cases and implementation details in our [examples](https://github.com/LMCache/LMCache/tree/dev/examples) to see LMCache in action.

## Connect with Us

We're excited to hear from you! Connect with us through these channels:

*   [Interest Form](https://forms.gle/mQfQDUXbKfp2St1z7)
*   [Sign up for our newsletter](https://mailchi.mp/tensormesh/lmcache-sign-up-newsletter)
*   [Join LMCache Slack](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-2viziwhue-5Amprc9k5hcIdXT7XevTaQ)
*   [LMCache Website](https://lmcache.ai/)
*   [Email](mailto:contact@lmcache.ai)

## Community Meetings

Join our bi-weekly community meetings to discuss LMCache and related topics. All are welcome!

*   **Schedule:** Tuesdays at 9:00 AM PT
*   **Meeting Link:** [https://uchicago.zoom.us/j/6603596916?pwd=Z1E5MDRWUSt2am5XbEt4dTFkNGx6QT09](https://uchicago.zoom.us/j/6603596916?pwd=Z1E5MDRWUSt2am5XbEt4dTFkNGx6QT09)
*   [Add to Calendar](https://drive.usercontent.google.com/u/0/uc?id=1f5EXbooGcwNwzIpTgn5u4PHqXgfypMtu&export=download)
*   **Meeting Notes:** [Meeting Notes](https://docs.google.com/document/d/1_Fl3vLtERFa3vTH00cezri78NihNBtSClK-_1tSrcow)
*   **Recordings:** [YouTube LMCache channel](https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA)

## Contributing

We encourage contributions! Check out the [Contributing Guide](CONTRIBUTING.md) to learn how to get involved.

See the [Onboarding] Welcoming contributors with good first issues!](https://github.com/LMCache/LMCache/issues/627) for good first issues.

## Citation

If you use LMCache in your research, please cite our papers:

```bibtex
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

Stay updated on LMCache:

*   [LinkedIn](https://www.linkedin.com/company/lmcache-lab/?viewAsMember=true)
*   [Twitter](https://x.com/lmcache)
*   [YouTube](https://www.youtube.com/@LMCacheTeam)

## License

LMCache is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.