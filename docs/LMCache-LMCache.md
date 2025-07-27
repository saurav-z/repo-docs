# LMCache: Accelerate LLM Inference with Efficient KV Cache Management

**LMCache dramatically speeds up Large Language Model (LLM) inference, reducing latency and maximizing throughput through smart KV cache utilization.**  Learn more and contribute on the original repository: [LMCache on GitHub](https://github.com/LMCache/LMCache).

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


--------------------------------------------------------------------------------

| [**Blog**](https://blog.lmcache.ai/)
| [**Documentation**](https://docs.lmcache.ai/)
| [**Join Slack**](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-36x1m765z-8FgDA_73vcXtlZ_4XvpE6Q)
| [**Interest Form**](https://forms.gle/MHwLiYDU6kcW3dLj7)
| [**Roadmap**](https://github.com/LMCache/LMCache/issues/574)

🔥 **NEW: For enterprise-scale deployment of LMCache and vLLM, please check out vLLM [Production Stack](https://github.com/vllm-project/production-stack). LMCache is also officially supported in [llm-d](https://github.com/llm-d/llm-d/) and [KServe](https://github.com/kserve/kserve)!**

## Key Features of LMCache:

*   **Reduced Latency & Increased Throughput:**  LMCache leverages intelligent KV cache management to minimize time-to-first-token (TTFT) and maximize the efficiency of LLM inference.
*   **KV Cache Reuse:** Stores KV caches across different locations (GPU, CPU DRAM, Disk) to reuse any reused text, not necessarily prefix, in any serving engine instance, saving GPU cycles.
*   **Integration with vLLM:** Optimized integration with vLLM v1, offering features like high-performance CPU KVCache offloading, disaggregated prefill, and P2P KVCache sharing.
*   **Broad Support:** Compatible with the vLLM production stack, llm-d, and KServe.
*   **Non-Prefix KV Cache Support:** Provides stable support for non-prefix KV caches.
*   **Flexible Storage Options:** Supports various storage options, including CPU, Disk, and NIXL.
*   **Easy Installation:**  Simple installation via pip and compatible with Linux NVIDIA GPU platforms.

## How LMCache Works

LMCache improves LLM serving by storing and reusing Key-Value (KV) caches of reusable text across various storage locations, including GPU, CPU DRAM, and local disk. This allows LMCache to reuse KV caches for any reused text, regardless of position or the serving engine instance. The result is significant savings in GPU cycles and reduced response times for users. With LMCache and vLLM, developers can expect 3-10x delay savings and GPU cycle reductions in LLM use cases like multi-round QA and RAG.

![performance](https://github.com/user-attachments/assets/86137f17-f216-41a0-96a7-e537764f7a4c)

## Installation

Install LMCache easily using pip:

```bash
pip install lmcache
```

For detailed installation instructions, please refer to the [documentation](https://docs.lmcache.ai/getting_started/installation).

## Getting Started

Explore the [Quickstart Examples](https://docs.lmcache.ai/getting_started/quickstart/) in the documentation to quickly get up and running with LMCache.

## Documentation

Comprehensive documentation is available at [docs.lmcache.ai](https://docs.lmcache.ai/), and you can also find updates and insights on the [LMCache blog](https://blog.lmcache.ai/).

## Examples

Explore practical use cases with the [examples](https://github.com/LMCache/LMCache/tree/dev/examples).

## Stay Connected

*   [Interest Form](https://forms.gle/mQfQDUXbKfp2St1z7)
*   [Newsletter](https://mailchi.mp/tensormesh/lmcache-sign-up-newsletter)
*   [Slack](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-2viziwhue-5Amprc9k5hcIdXT7XevTaQ)
*   [Website](https://lmcache.ai/)
*   [Email](contact@lmcache.ai)

## Community

*   **Community meetings** are held bi-weekly on Tuesdays at 9:00 AM PT ([Add to Calendar](https://drive.usercontent.google.com/u/0/uc?id=1f5EXbooGcwNwzIpTgn5u4PHqXgfypMtu&export=download)).
*   Meeting notes and summaries are available in this [document](https://docs.google.com/document/d/1_Fl3vLtERFa3vTH00cezri78NihNBtSClK-_1tSrcow).
*   Recordings are available on the [YouTube LMCache channel](https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA).

## Contribute

We value all contributions. Please see the [Contributing Guide](CONTRIBUTING.md) for details on how to contribute.

## Citation

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

## Social

*   [LinkedIn](https://www.linkedin.com/company/lmcache-lab/?viewAsMember=true)
*   [Twitter](https://x.com/lmcache)
*   [YouTube](https://www.youtube.com/@LMCacheTeam)

## License

LMCache is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.