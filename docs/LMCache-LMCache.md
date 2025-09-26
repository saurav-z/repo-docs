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
<br/>

--------------------------------------------------------------------------------

| [**Blog**](https://blog.lmcache.ai/)
| [**Documentation**](https://docs.lmcache.ai/)
| [**Join Slack**](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-3bgx768yd-H8WkOTmPtbxVYJ5nuZ4dmA)
| [**Interest Form**](https://forms.gle/MHwLiYDU6kcW3dLj7)
| [**Roadmap**](https://github.com/LMCache/LMCache/issues/1253)

🔥 **LMCache dramatically accelerates LLM serving by intelligently caching KV caches, reducing latency and increasing throughput. Learn more at [LMCache's GitHub](https://github.com/LMCache/LMCache)!**

## What is LMCache?

LMCache is a powerful extension for LLM serving engines, designed to significantly reduce Time-To-First-Token (TTFT) and boost throughput, especially for long-context applications.  By caching KV caches across multiple locations (GPU, CPU, and local disk), LMCache reuses KV caches for any reused text, regardless of the serving engine instance. This results in significant GPU cycle savings and reduced user response times.

## Key Features

*   **Enhanced Performance:** Achieve 3-10x delay savings and GPU cycle reduction when combined with vLLM, optimizing performance for multi-round QA, RAG, and similar LLM use cases.
*   **vLLM Integration:** Seamlessly integrates with vLLM v1, offering high-performance CPU KVCache offloading, disaggregated prefill, and P2P KVCache sharing.
*   **Production-Ready:** Supported in the vLLM production stack, llm-d, and KServe for enterprise-scale deployment.
*   **Non-Prefix KV Cache Support:** Stable support for non-prefix KV caches, increasing flexibility.
*   **Flexible Storage:** Supports storage on CPU, Disk, and [NIXL](https://github.com/ai-dynamo/nixl) for versatile deployment options.
*   **Easy Installation:** Installs easily via pip and integrates with the latest vLLM versions.

## Installation

Install LMCache using pip:

```bash
pip install lmcache
```

For detailed instructions and troubleshooting, especially if you're using a different serving engine or are not on the latest vLLM stable version, consult the [LMCache documentation](https://docs.lmcache.ai/getting_started/installation).

## Getting Started

Explore the [Quickstart Examples](https://docs.lmcache.ai/getting_started/quickstart/) in the documentation to quickly get up and running with LMCache.

## Documentation

Comprehensive documentation is available at [https://docs.lmcache.ai/](https://docs.lmcache.ai/), and the [LMCache blog](https://blog.lmcache.ai/) offers regular updates and insights.

## Examples

Hands-on examples showcasing various use cases can be found in the [examples directory](https://github.com/LMCache/LMCache/tree/dev/examples).

## Connect with Us

*   **Interested in learning more?** Fill out the [interest form](https://forms.gle/mQfQDUXbKfp2St1z7).
*   **Stay updated:** [Sign up for our newsletter](https://mailchi.mp/tensormesh/lmcache-sign-up-newsletter).
*   **Join the community:** [Join LMCache Slack](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-2viziwhue-5Amprc9k5hcIdXT7XevTaQ).
*   **Visit our website:** [lmcache.ai](https://lmcache.ai/)
*   **Contact us directly:** [contact@lmcache.ai](mailto:contact@lmcache.ai)

## Community Meeting

Join our bi-weekly community meetings every Tuesday at 9:00 AM PT: [Zoom Link](https://uchicago.zoom.us/j/6603596916?pwd=Z1E5MDRWUSt2am5XbEt4dTFkNGx6QT09) – [Add to Google Calendar](https://calendar.google.com/calendar/u/0/r?cid=Y19mNGY2ZmMwZjUxMWYyYTZmZmE1ZTVlMGI2Yzk2NmFmZjNhM2Y4ODZiZmU5OTU5MDJlMmE3ZmUyOGZmZThlOWY5QGdyb3VwLmNhbGVuZGFyLmdvb2dsZS5jb20)

Meeting notes are available [here](https://docs.google.com/document/d/1_Fl3vLtERFa3vTH00cezri78NihNBtSClK-_1tSrcow), and recordings are on the [YouTube LMCache channel](https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA).

## Contributing

Contributions are welcome! See the [Contributing Guide](CONTRIBUTING.md) and [[Onboarding] Welcoming contributors with good first issues!](https://github.com/LMCache/LMCache/issues/627) for details.

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

LMCache is licensed under the Apache License 2.0.  See the [LICENSE](LICENSE) file for details.