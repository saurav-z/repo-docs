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

**LMCache accelerates LLM serving by intelligently caching and reusing KV caches for faster response times and reduced GPU costs.**  [Explore the LMCache repository](https://github.com/LMCache/LMCache).

---

## Key Features

*   **Accelerated LLM Performance:** Significantly reduces Time-to-First-Token (TTFT) and boosts throughput, especially in long-context scenarios.
*   **KV Cache Reuse:** Efficiently reuses KV caches of any reused text across various serving instances, not just prefixes, saving valuable GPU cycles.
*   **vLLM Integration:** Seamlessly integrates with vLLM v1, offering features like high-performance CPU KVCache offloading, disaggregated prefill, and P2P KVCache sharing.
*   **Production Ready:** Supported in the [vLLM production stack](https://github.com/vllm-project/production-stack/), [llm-d](https://github.com/llm-d/llm-d/), and [KServe](https://github.com/kserve/kserve).
*   **Non-Prefix Cache Support:** Provides stable support for non-prefix KV caches, enhancing flexibility.
*   **Flexible Storage:** Supports various storage options including CPU, Disk, and [NIXL](https://github.com/ai-dynamo/nixl).
*   **Easy Installation:** Install via pip and works on Linux NVIDIA GPU platforms.

## What is LMCache?

LMCache is an LLM serving engine extension designed to reduce TTFT and increase throughput, particularly in long-context scenarios. It achieves this by storing KV caches of reusable text across different locations (GPU, CPU DRAM, Local Disk). This enables LMCache to reuse the KV caches of **_any_** reused text in **_any_** serving engine instance. As a result, LMCache saves GPU cycles and reduces user response times. By combining LMCache with vLLM, developers can achieve 3-10x delay savings and GPU cycle reduction in many LLM use cases, including multi-round QA and RAG.

## Installation

To get started with LMCache, install it using pip:

```bash
pip install lmcache
```

Detailed [installation instructions](https://docs.lmcache.ai/getting_started/installation) are available in our documentation.

## Getting Started

Explore our [Quickstart Examples](https://docs.lmcache.ai/getting_started/quickstart/) in the documentation to quickly understand and implement LMCache.

## Documentation

Comprehensive documentation is available to guide you through the features and usage of LMCache. Access the [documentation](https://docs.lmcache.ai/) online for detailed information.

## Examples

Find practical implementations and learn how to use LMCache effectively with our [examples](https://github.com/LMCache/LMCache/tree/dev/examples), designed for various use cases.

## Stay Connected

*   **Blog:** [LMCache Blogs](https://blog.lmcache.ai/)
*   **Join the Community:** [Slack](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-36x1m765z-8FgDA_73vcXtlZ_4XvpE6Q)
*   **Interest Form:** [Interest Form](https://forms.gle/MHwLiYDU6kcW3dLj7)
*   **Website:** [LMCache Website](https://lmcache.ai/)
*   **Email:** [contact@lmcache.ai](mailto:contact@lmcache.ai)

## Community Meeting

Join our weekly community meetings to discuss LMCache and stay updated.

Meetings alternate weekly between:

*   Tuesdays at 9:00 AM PT – [Add to Calendar](https://drive.google.com/file/d/15Xz8-LtpBQ5QgR7KrorOOyfuohCFQmwn/view?usp=drive_link)
*   Tuesdays at 6:30 PM PT – [Add to Calendar](https://drive.google.com/file/d/1WMZNFXV24kWzprDjvO-jQ7mOY7whqEdG/view?usp=drive_link)

Meeting notes are available [here](https://docs.google.com/document/d/1_Fl3vLtERFa3vTH00cezri78NihNBtSClK-_1tSrcow), and recordings are on the [YouTube LMCache channel](https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA).

## Contributing

We welcome contributions!  Please refer to the [Contributing Guide](CONTRIBUTING.md) for details on how to contribute.

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

## Social Media

*   [LinkedIn](https://www.linkedin.com/company/lmcache-lab/?viewAsMember=true)
*   [Twitter](https://x.com/lmcache)
*   [YouTube](https://www.youtube.com/@LMCacheTeam)

## License

LMCache is licensed under the Apache License 2.0.  See the [LICENSE](LICENSE) file for details.