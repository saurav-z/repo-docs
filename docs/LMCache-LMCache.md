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

[**LMCache: Accelerate LLM Serving with Efficient KV Cache Management**](https://github.com/LMCache/LMCache)

--------------------------------------------------------------------------------

| [**Blog**](https://blog.lmcache.ai/)
| [**Documentation**](https://docs.lmcache.ai/)
| [**Join Slack**](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-36x1m765z-8FgDA_73vcXtlZ_4XvpE6Q)
| [**Interest Form**](https://forms.gle/MHwLiYDU6kcW3dLj7)
| [**Roadmap**](https://github.com/LMCache/LMCache/issues/1253)

🔥 **NEW: For enterprise-scale deployment of LMCache and vLLM, please check out vLLM [Production Stack](https://github.com/vllm-project/production-stack). LMCache is also officially supported in [llm-d](https://github.com/llm-d/llm-d/) and [KServe](https://github.com/kserve/kserve)!**

## About LMCache

LMCache is a powerful extension designed to optimize Large Language Model (LLM) serving, reducing Time-To-First-Token (TTFT) and increasing throughput, especially in long-context scenarios. LMCache achieves this by intelligently caching and reusing Key-Value (KV) caches across multiple instances and locations, including GPU, CPU DRAM, and local disk. This approach significantly cuts down on GPU usage and accelerates response times.

## Key Features

*   **Enhanced Performance with vLLM Integration:** Integrated with vLLM v1, providing high-performance CPU KV cache offloading, disaggregated prefill, and P2P KV cache sharing.
*   **Wide Ecosystem Support:** Officially supported in the [vLLM production stack](https://github.com/vllm-project/production-stack/), [llm-d](https://github.com/llm-d/llm-d/), and [KServe](https://github.com/kserve/kserve).
*   **Flexible KV Cache Management:** Stable support for non-prefix KV caches, allowing for efficient reuse of cached data regardless of sequence start.
*   **Diverse Storage Options:** Supports KV cache storage across CPU, Disk, and [NIXL](https://github.com/ai-dynamo/nixl) to optimize resource utilization.
*   **Seamless Installation:** Easy to install via pip, ensuring compatibility with the latest vLLM versions.

## Installation

Get started with LMCache quickly by installing it via pip:

```bash
pip install lmcache
```

For detailed installation instructions, especially if you are not using the latest stable vLLM version or if you need to resolve dependency issues, please refer to the [installation guide](https://docs.lmcache.ai/getting_started/installation) in the documentation.

## Getting Started

Explore the [Quickstart Examples](https://docs.lmcache.ai/getting_started/quickstart/) in the documentation to quickly understand and implement LMCache in your projects.

## Documentation

Comprehensive documentation is available at [https://docs.lmcache.ai/](https://docs.lmcache.ai/), providing in-depth information and guidance.

## Examples

Experiment with LMCache using practical [examples](https://github.com/LMCache/LMCache/tree/dev/examples) that demonstrate how to address various LLM use cases.

## Connect with Us

Stay connected with LMCache and our community:

*   Fill out the [interest form](https://forms.gle/mQfQDUXbKfp2St1z7)
*   [Sign up for our newsletter](https://mailchi.mp/tensormesh/lmcache-sign-up-newsletter)
*   [Join the LMCache Slack](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-2viziwhue-5Amprc9k5hcIdXT7XevTaQ)
*   Visit the [LMCache website](https://lmcache.ai/)
*   Email us at [contact@lmcache.ai](mailto:contact@lmcache.ai)

## Community Meeting

Join our bi-weekly community meetings to stay updated and collaborate:

*   **Meetings:** Bi-weekly on Tuesdays at 9:00 AM PT. [Add to Calendar](https://drive.usercontent.google.com/u/0/uc?id=1f5EXbooGcwNwzIpTgn5u4PHqXgfypMtu&export=download)
*   **Meeting Notes:** Summaries of standups, discussions, and action items are available in this [document](https://docs.google.com/document/d/1_Fl3vLtERFa3vTH00cezri78NihNBtSClK-_1tSrcow).
*   **Recordings:** Past meetings are recorded and available on the [YouTube LMCache channel](https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA).

## Contributing

We welcome your contributions! Please review the [Contributing Guide](CONTRIBUTING.md) for details on how to contribute.

Also, check out [[Onboarding] Welcoming contributors with good first issues!](https://github.com/LMCache/LMCache/issues/627)

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
*   [Youtube](https://www.youtube.com/@LMCacheTeam)

## License

LMCache is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.