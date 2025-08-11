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

# LMCache: Accelerate LLM Serving with Intelligent KV Caching

**LMCache significantly reduces the time-to-first-token (TTFT) and increases throughput for Large Language Models (LLMs) by intelligently caching KV caches.**  Learn more and contribute on the [original repository](https://github.com/LMCache/LMCache).

**Key Features:**

*   **KV Cache Reuse:**  LMCache caches and reuses KV caches of repeated text across instances and various locations (GPU, CPU DRAM, Local Disk) for optimized performance.
*   **vLLM Integration:** Seamlessly integrates with vLLM for enhanced performance, including CPU KVCache offloading, disaggregated prefill, and P2P KVCache sharing.
*   **Wide Support:** Officially supported in the [vLLM production stack](https://github.com/vllm-project/production-stack/), [llm-d](https://github.com/llm-d/llm-d/), and [KServe](https://github.com/kserve).
*   **Flexible Storage:** Supports storing KV caches on CPU, Disk, and [NIXL](https://github.com/ai-dynamo/nixl).
*   **Non-Prefix Cache Support:** Provides stable support for non-prefix KV caches, enhancing caching capabilities.
*   **Easy Installation:** Simple installation via `pip install lmcache`.

## Quickstart

1.  **Installation:**
    ```bash
    pip install lmcache
    ```
    *Works on Linux NVIDIA GPU platform.*
    Detailed installation instructions are available in the [documentation](https://docs.lmcache.ai/getting_started/installation).

2.  **Getting Started:**  Explore the [Quickstart Examples](https://docs.lmcache.ai/getting_started/quickstart/) in the docs to begin.

## Documentation

Comprehensive documentation is available to guide you through LMCache: [https://docs.lmcache.ai/](https://docs.lmcache.ai/)

## Examples

Explore hands-on [examples](https://github.com/LMCache/LMCache/tree/dev/examples) to see how LMCache can be applied to various use cases.

## Interested in Connecting?

*   [Interest Form](https://forms.gle/mQfQDUXbKfp2St1z7)
*   [Sign up for our newsletter](https://mailchi.mp/tensormesh/lmcache-sign-up-newsletter)
*   [Join LMCache Slack](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-2viziwhue-5Amprc9k5hcIdXT7XevTaQ)
*   [Check out LMCache website](https://lmcache.ai/)
*   [Email](contact@lmcache.ai)

## Community Meetings

Join the bi-weekly LMCache community meetings to discuss and collaborate.

*   **Schedule:** Tuesdays at 9:00 AM PT - [Add to Calendar](https://drive.usercontent.google.com/u/0/uc?id=1f5EXbooGcwNwzIpTgn5u4PHqXgfypMtu&export=download)
*   **Meeting Notes:** [Google Doc](https://docs.google.com/document/d/1_Fl3vLtERFa3vTH00cezri78NihNBtSClK-_1tSrcow)
*   **Recordings:** [YouTube Channel](https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA)

## Contributing

We welcome contributions!  Please review the [Contributing Guide](CONTRIBUTING.md) for details.

## Citation

If you use LMCache for your research, please cite our papers:

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

Licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.