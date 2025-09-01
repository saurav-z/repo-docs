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

## LMCache: Accelerate LLM Performance with Efficient KV Cache Management

LMCache is a powerful extension for LLM serving engines, significantly boosting performance by optimizing Key-Value (KV) cache management.  [Explore LMCache on GitHub](https://github.com/LMCache/LMCache).

**Key Features:**

*   🚀 **Reduced TTFT & Increased Throughput:**  LMCache drastically lowers Time To First Token (TTFT) and enhances throughput, especially in long-context scenarios.
*   💾 **KV Cache Reusability:** Efficiently reuses KV caches of any repeated text, regardless of its position in the context, across various serving engine instances.
*   ⚙️ **Integration with vLLM:** Seamlessly integrates with vLLM v1, providing high-performance CPU KVCache offloading, disaggregated prefill, and P2P KVCache sharing.
*   🌐 **Broad Compatibility:** Supported in the vLLM production stack, llm-d, and KServe.
*   💾 **Flexible Storage Options:** Supports CPU, Disk, and [NIXL](https://github.com/ai-dynamo/nixl) for KV cache storage.
*   ✅ **Stable Support:** Provides stable support for non-prefix KV caches, enhancing flexibility.

##  Why LMCache?

LMCache revolutionizes LLM performance by caching reusable text fragments, such as those in multi-round QA and RAG applications. This reduces GPU workload and dramatically speeds up responses, leading to 3-10x delay savings and GPU cycle reduction.

## Installation

Easily install LMCache using pip:

```bash
pip install lmcache
```

For detailed installation guidance, particularly for specific vLLM versions or alternative serving engines, consult the [documentation](https://docs.lmcache.ai/getting_started/installation).

## Getting Started

Dive into LMCache with our [Quickstart Examples](https://docs.lmcache.ai/getting_started/quickstart/) for practical demonstrations.

## Documentation

Comprehensive documentation is available at [docs.lmcache.ai](https://docs.lmcache.ai/), and keep up-to-date with the latest developments on the [LMCache blog](https://blog.lmcache.ai/).

## Examples

Explore diverse use cases with our hands-on [examples](https://github.com/LMCache/LMCache/tree/dev/examples).

## Community & Resources

*   **Join the Community:**  Get involved through our [Slack channel](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-36x1m765z-8FgDA_73vcXtlZ_4XvpE6Q) and the [community meeting]( https://uchicago.zoom.us/j/6603596916?pwd=Z1E5MDRWUSt2am5XbEt4dTFkNGx6QT09) (bi-weekly on Tuesdays at 9:00 AM PT - [Add to Calendar](https://drive.usercontent.google.com/u/0/uc?id=1f5EXbooGcwNwzIpTgn5u4PHqXgfypMtu&export=download)).
*   **Stay Informed:** Sign up for our [newsletter](https://mailchi.mp/tensormesh/lmcache-sign-up-newsletter) and follow us on [LinkedIn](https://www.linkedin.com/company/lmcache-lab/?viewAsMember=true) and [Twitter](https://x.com/lmcache).
*   **Watch Recordings:** Catch up on past community meetings via our [YouTube Channel](https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA).
*   **Connect with Us:** Reach out to us via email: [contact@lmcache.ai](mailto:contact@lmcache.ai).
*   **Contribute:** Learn how to contribute by reading the [Contributing Guide](CONTRIBUTING.md) or finding [good first issues!](https://github.com/LMCache/LMCache/issues/627)

## Roadmap

Find out what is coming next on the [Roadmap](https://github.com/LMCache/LMCache/issues/1253).

## Citation

If you are using LMCache in your research, please cite the following papers:

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

## License

LMCache is licensed under the [Apache License 2.0](LICENSE).