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

# LMCache: Accelerate LLM Serving with Efficient KV Cache Management

LMCache drastically improves Large Language Model (LLM) serving performance by optimizing KV cache reuse, slashing latency and boosting throughput.  [Explore the LMCache project on GitHub](https://github.com/LMCache/LMCache).

## Key Features

*   **Enhanced Performance:** Reduces Time-To-First-Token (TTFT) and increases throughput for LLM inference.
*   **KV Cache Reuse:** Efficiently reuses KV caches for any reused text, not just prefixes, across multiple instances.
*   **vLLM Integration:** Seamlessly integrates with vLLM v1 for high-performance CPU KVCache offloading, disaggregated prefill, and P2P KVCache sharing.
*   **Flexible Storage:** Supports various storage options including CPU, Disk, and [NIXL](https://github.com/ai-dynamo/nixl).
*   **Production Ready:** Supported in the [vLLM production stack](https://github.com/vllm-project/production-stack/), [llm-d](https://github.com/llm-d/llm-d/), and [KServe](https://github.com/kserve/kserve).
*   **Stable Non-Prefix Cache Support:**  Provides reliable support for non-prefix KV caches.
*   **Easy Installation:** Simple pip installation for quick setup.

## What is LMCache?

LMCache is an extension designed to supercharge your LLM serving engine by cleverly storing and reusing KV caches across different locations (GPU, CPU DRAM, Local Disk). This means that when your LLM encounters the same text, it can retrieve pre-computed KV caches, saving valuable GPU resources and drastically reducing response times for users. By combining LMCache with vLLM, experience up to a 3-10x improvement in delay savings and GPU cycle reduction across diverse LLM use cases, including multi-round QA and RAG.

## Installation

Install LMCache using pip:

```bash
pip install lmcache
```

Requires a Linux NVIDIA GPU platform. For detailed instructions, see the [Installation Guide](https://docs.lmcache.ai/getting_started/installation).

## Getting Started

Get up and running quickly with the [Quickstart Examples](https://docs.lmcache.ai/getting_started/quickstart/) in our documentation.

## Documentation

Comprehensive documentation is available at [LMCache Documentation](https://docs.lmcache.ai/).

## Examples

Explore practical applications and see how LMCache works with our [examples](https://github.com/LMCache/LMCache/tree/dev/examples).

## Connect with Us

*   **Website:** [lmcache.ai](https://lmcache.ai/)
*   **Blog:** [blog.lmcache.ai](https://blog.lmcache.ai/)
*   **Slack:** [Join LMCache Slack](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-36x1m765z-8FgDA_73vcXtlZ_4XvpE6Q)
*   **Interest Form:** [Interest Form](https://forms.gle/MHwLiYDU6kcW3dLj7)
*   **Newsletter:** [Sign Up for Newsletter](https://mailchi.mp/tensormesh/lmcache-sign-up-newsletter)
*   **Email:** [contact@lmcache.ai](mailto:contact@lmcache.ai)

## Community

*   **Community Meeting:** Join our weekly community meetings to discuss LMCache development and usage. Meetings are held weekly, alternating between:
    *   Tuesdays at 9:00 AM PT – [Add to Calendar](https://drive.google.com/file/d/15Xz8-LtpBQ5QgR7KrorOOyfuohCFQmwn/view?usp=drive_link)
    *   Tuesdays at 6:30 PM PT – [Add to Calendar](https://drive.google.com/file/d/1WMZNFXV24kWzprDjvO-jQ7mOY7whqEdG/view?usp=drive_link)
*   **Meeting Notes:** Access meeting summaries, discussions, and action items in this [document](https://docs.google.com/document/d/1_Fl3vLtERFa3vTH00cezri78NihNBtSClK-_1tSrcow).
*   **Meeting Recordings:** Watch recordings of community meetings on our [YouTube channel](https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA).

## Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) to learn how you can help.

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

*   [LinkedIn](https://www.linkedin.com/company/lmcache-lab/?viewAsMember=true)
*   [Twitter](https://x.com/lmcache)
*   [YouTube](https://www.youtube.com/@LMCacheTeam)

## License

LMCache is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.