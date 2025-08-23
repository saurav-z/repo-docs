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

# LMCache: Supercharge Your LLM Serving with Intelligent Caching

LMCache revolutionizes Large Language Model (LLM) serving by dramatically reducing Time-To-First-Token (TTFT) and increasing throughput, particularly for long-context applications.  [Explore LMCache on GitHub](https://github.com/LMCache/LMCache)

## Key Features

*   **Accelerated LLM Performance:** Significantly reduces TTFT and boosts throughput by caching and reusing KV caches across various storage locations (GPU, CPU DRAM, Local Disk).
*   **Non-Prefix Cache Support:**  Efficiently caches and reuses KV caches for *any* reused text, not just prefixes.
*   **vLLM Integration:** Seamlessly integrates with vLLM v1, providing CPU KVCache offloading, disaggregated prefill, and P2P KVCache sharing for optimized performance.
*   **Versatile Storage Options:** Supports caching on CPU, Disk, and [NIXL](https://github.com/ai-dynamo/nixl), enabling flexible resource utilization.
*   **Production-Ready:**  Officially supported in the [vLLM production stack](https://github.com/vllm-project/production-stack/), [llm-d](https://github.com/llm-d/llm-d/), and [KServe](https://github.com/kserve/kserve).
*   **Easy Installation:** Simple pip installation for quick setup and deployment.

## Installation

Install LMCache easily with pip:

```bash
pip install lmcache
```

For more detailed instructions, including those for non-stable vLLM versions and alternative serving engines, refer to the [detailed installation guide](https://docs.lmcache.ai/getting_started/installation) in the documentation.

## Getting Started

Dive into LMCache quickly with our [Quickstart Examples](https://docs.lmcache.ai/getting_started/quickstart/) to see the benefits in action.

## Documentation

Comprehensive [documentation](https://docs.lmcache.ai/) is available to guide you through all aspects of LMCache.

## Examples

Explore practical use cases with our [examples](https://github.com/LMCache/LMCache/tree/dev/examples) to see how LMCache can be applied in various scenarios.

## Resources

*   **Blog:** [LMCache Blogs](https://blog.lmcache.ai/)
*   **Join Slack:** [Join Slack](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-36x1m765z-8FgDA_73vcXtlZ_4XvpE6Q)
*   **Interest Form:** [Interest Form](https://forms.gle/MHwLiYDU6kcW3dLj7)
*   **Roadmap:** [Roadmap](https://github.com/LMCache/LMCache/issues/1253)
*   **Website:** [LMCache Website](https://lmcache.ai/)

## Community & Support

*   **Bi-Weekly Community Meeting:** Join us bi-weekly on Tuesdays at 9:00 AM PT - [Add to Calendar](https://drive.usercontent.google.com/u/0/uc?id=1f5EXbooGcwNwzIpTgn5u4PHqXgfypMtu&export=download)
    *   Meeting notes are available [here](https://docs.google.com/document/d/1_Fl3vLtERFa3vTH00cezri78NihNBtSClK-_1tSrcow).
    *   Meeting recordings are available on the [YouTube LMCache channel](https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA).
*   **Contact:**  For any questions, reach out via email at [contact@lmcache.ai](mailto:contact@lmcache.ai).

## Contributing

We welcome contributions!  Please see the [Contributing Guide](CONTRIBUTING.md) for details on how to get involved, and check out [[Onboarding] Welcoming contributors with good first issues!](https://github.com/LMCache/LMCache/issues/627) for good first issues.

## Citation

If you use LMCache in your research, please cite these papers:

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

*   [LinkedIn](https://www.linkedin.com/company/lmcache-lab/?viewAsMember=true)
*   [Twitter](https://x.com/lmcache)
*   [YouTube](https://www.youtube.com/@LMCacheTeam)

## License

LMCache is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.