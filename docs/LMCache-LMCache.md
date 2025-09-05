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

## LMCache: Accelerate LLM Serving with Efficient KV Caching

LMCache optimizes Large Language Model (LLM) serving by caching reusable text, drastically reducing response times and GPU costs.  [Learn more on our GitHub](https://github.com/LMCache/LMCache).

**Key Features:**

*   **Seamless vLLM Integration:** Enhanced performance with vLLM v1, including high-performance CPU KVCache offloading, disaggregated prefill, and P2P KVCache sharing.
*   **Multi-Platform Support:** Compatible with the vLLM production stack, llm-d, and KServe for flexible deployment options.
*   **Non-Prefix Cache Support:** Enables stable caching of non-prefix KV caches for improved efficiency.
*   **Versatile Storage Options:** Supports KV cache storage on CPU, Disk, and NIXL for optimized resource utilization.
*   **Easy Installation:** Simple pip installation for quick setup and integration.

## Summary

LMCache is an innovative LLM serving engine extension, engineered to slash "Time-to-First-Token" (TTFT) and dramatically boost throughput, particularly in long-context scenarios.  It achieves this by intelligently caching and reusing KV caches of any repeated text (not just prefixes) across multiple serving engine instances, including (GPU, CPU DRAM, Local Disk). By combining LMCache with vLLM, you can achieve 3-10x delay savings and reduce GPU cycle usage in numerous LLM applications, including multi-round QA and RAG.

## Installation

Install LMCache using pip:

```bash
pip install lmcache
```

Ensure you're on a Linux NVIDIA GPU platform for optimal performance. For detailed installation instructions, including troubleshooting "undefined symbol" errors and torch mismatches, refer to the [documentation](https://docs.lmcache.ai/getting_started/installation).

## Getting Started

Explore the [Quickstart Examples](https://docs.lmcache.ai/getting_started/quickstart/) in our documentation to quickly get up and running with LMCache.

## Documentation

Comprehensive documentation is available at [https://docs.lmcache.ai/](https://docs.lmcache.ai/).  Stay updated with the latest developments on the [LMCache blog](https://blog.lmcache.ai/).

## Examples

See practical implementations in the [examples](https://github.com/LMCache/LMCache/tree/dev/examples) directory, showcasing various use cases.

## Connect with Us

*   **Interest Form:** [https://forms.gle/MHwLiYDU6kcW3dLj7](https://forms.gle/MHwLiYDU6kcW3dLj7)
*   **Newsletter:** [https://mailchi.mp/tensormesh/lmcache-sign-up-newsletter](https://mailchi.mp/tensormesh/lmcache-sign-up-newsletter)
*   **Slack:** [https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-2viziwhue-5Amprc9k5hcIdXT7XevTaQ](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-2viziwhue-5Amprc9k5hcIdXT7XevTaQ)
*   **Website:** [https://lmcache.ai/](https://lmcache.ai/)
*   **Email:** [contact@lmcache.ai](mailto:contact@lmcache.ai)

## Community Meeting

Join our bi-weekly community meetings to stay connected and share your experiences:

*   **Meeting Schedule:** Tuesdays at 9:00 AM PT
    *   [Add to Calendar](https://drive.usercontent.google.com/u/0/uc?id=1f5EXbooGcwNwzIpTgn5u4PHqXgfypMtu&export=download)
*   **Meeting Notes:** [https://docs.google.com/document/d/1_Fl3vLtERFa3vTH00cezri78NihNBtSClK-_1tSrcow](https://docs.google.com/document/d/1_Fl3vLtERFa3vTH00cezri78NihNBtSClK-_1tSrcow)
*   **Recordings:** [YouTube LMCache Channel](https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA)

## Contributing

We welcome contributions!  See the [Contributing Guide](CONTRIBUTING.md) for details.  Check out the [[Onboarding] Welcoming contributors with good first issues!](https://github.com/LMCache/LMCache/issues/627) to find ways to contribute.

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

## Socials

*   **LinkedIn:** [https://www.linkedin.com/company/lmcache-lab/?viewAsMember=true](https://www.linkedin.com/company/lmcache-lab/?viewAsMember=true)
*   **Twitter:** [https://x.com/lmcache](https://x.com/lmcache)
*   **YouTube:** [https://www.youtube.com/@LMCacheTeam](https://www.youtube.com/@LMCacheTeam)

## License

LMCache is licensed under the Apache License 2.0.  See the [LICENSE](LICENSE) file for details.