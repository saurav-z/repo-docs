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

# LMCache: Accelerate LLM Inference with Efficient KV Cache Management

LMCache dramatically speeds up Large Language Model (LLM) inference by intelligently caching and reusing key-value (KV) pairs.  [Explore the LMCache repository](https://github.com/LMCache/LMCache).

**Key Features:**

*   **Reduced TTFT and Increased Throughput:** Significantly lowers Time-To-First-Token and boosts overall throughput, especially in long-context scenarios.
*   **KV Cache Reusability:**  Reuses KV caches for *any* reused text, not just prefixes, across different serving engine instances, saving GPU cycles.
*   **vLLM Integration:** Seamlessly integrates with vLLM for high-performance CPU KVCache offloading, disaggregated prefill, and P2P KVCache sharing.
*   **Production Ready:** Supported in the [vLLM production stack](https://github.com/vllm-project/production-stack/), [llm-d](https://github.com/llm-d/llm-d/), and [KServe](https://github.com/kserve).
*   **Flexible Storage Options:** Supports CPU, Disk, and [NIXL](https://github.com/ai-dynamo/nixl) storage for KV caches.
*   **Non-Prefix KV Cache Support:** Stable support for KV caches beyond simple prefixes.

## Installation

Install LMCache using pip:

```bash
pip install lmcache
```

Works on Linux NVIDIA GPU platform.  Detailed installation instructions are available in the [documentation](https://docs.lmcache.ai/getting_started/installation).

## Getting Started

Get up and running quickly with the [Quickstart Examples](https://docs.lmcache.ai/getting_started/quickstart/).

## Documentation

Comprehensive documentation is available at [https://docs.lmcache.ai/](https://docs.lmcache.ai/).  Stay updated with the latest news and insights on the [LMCache blog](https://blog.lmcache.ai/).

## Examples

Explore practical use cases and implementations with our [examples](https://github.com/LMCache/LMCache/tree/dev/examples).

## Community and Support

*   **Join the Community:** [Join Slack](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-3bgx768yd-H8WkOTmPtbxVYJ5nuZ4dmA) and [fill out the interest form](https://forms.gle/MHwLiYDU6kcW3dLj7).
*   **Stay Informed:** [Sign up for our newsletter](https://mailchi.mp/tensormesh/lmcache-sign-up-newsletter).
*   **Community Meetings:** Bi-weekly community meetings are held on Tuesdays at 9:00 AM PT.  [Add to Calendar](https://drive.usercontent.google.com/u/0/uc?id=1f5EXbooGcwNwzIpTgn5u4PHqXgfypMtu&export=download).  Meeting notes and recordings are available [here](https://docs.google.com/document/d/1_Fl3vLtERFa3vTH00cezri78NihNBtSClK-_1tSrcow) and on the [YouTube LMCache channel](https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA), respectively.
*   **Contact Us:** [Drop an email](mailto:contact@lmcache.ai).

## Contributing

We welcome contributions!  Please see our [Contributing Guide](CONTRIBUTING.md) and the [[Onboarding] Welcoming contributors with good first issues!](https://github.com/LMCache/LMCache/issues/627).

## Citation

If you use LMCache in your research, please cite our papers: (citations provided in original README)

## Social Media

*   [LinkedIn](https://www.linkedin.com/company/lmcache-lab/?viewAsMember=true)
*   [Twitter](https://x.com/lmcache)
*   [Youtube](https://www.youtube.com/@LMCacheTeam)

## License

LMCache is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.