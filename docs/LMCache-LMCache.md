# LMCache: Accelerate LLM Serving with Efficient KV Cache Management

**LMCache dramatically reduces response times and lowers GPU costs for Large Language Models by intelligently caching and reusing key-value (KV) caches.** ([See the original repo](https://github.com/LMCache/LMCache))

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

---

## Key Features

*   **Speed up LLMs:** Reduce Time-To-First-Token (TTFT) and increase throughput, especially in long-context scenarios.
*   **KV Cache Reuse:** Stores and reuses KV caches across GPU, CPU DRAM, and local disk for any reused text, saving valuable GPU cycles.
*   **vLLM Integration:** Seamlessly integrates with vLLM, offering high-performance CPU KVCache offloading, disaggregated prefill, and P2P KVCache sharing.
*   **Non-Prefix Cache Support:** Stable support for caching non-prefix KV caches, broadening applicability.
*   **Flexible Storage Options:** Supports CPU, Disk, and [NIXL](https://github.com/ai-dynamo/nixl) for efficient cache storage.
*   **Production Ready:** Officially supported in the [vLLM production stack](https://github.com/vllm-project/production-stack/), [llm-d](https://github.com/llm-d/llm-d/), and [KServe](https://github.com/kserve).

## Installation

Install LMCache easily using pip:

```bash
pip install lmcache
```

Detailed [installation instructions](https://docs.lmcache.ai/getting_started/installation) are available, especially for users of different serving engines or older vLLM versions.

## Getting Started

Explore our [Quickstart Examples](https://docs.lmcache.ai/getting_started/quickstart/) in the documentation.

## Documentation

Comprehensive documentation is available at [docs.lmcache.ai](https://docs.lmcache.ai/).

## Examples

Explore practical use cases with our [examples](https://github.com/LMCache/LMCache/tree/dev/examples).

## Community & Support

*   **Join Slack:** [Join Slack](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-3bgx768yd-H8WkOTmPtbxVYJ5nuZ4dmA)
*   **Blog:** [LMCache blogs](https://blog.lmcache.ai/)
*   **Community Meetings:**  Bi-weekly on Tuesdays at 9:00 AM PT ([Add to Calendar](https://drive.usercontent.google.com/u/0/uc?id=1f5EXbooGcwNwzIpTgn5u4PHqXgfypMtu&export=download)). Meeting notes are available [here](https://docs.google.com/document/d/1_Fl3vLtERFa3vTH00cezri78NihNBtSClK-_1tSrcow).  Recordings are on the [YouTube LMCache channel](https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA).
*   **Interest Form:** [Interest Form](https://forms.gle/MHwLiYDU6kcW3dLj7)
*   **Roadmap**: [Roadmap](https://github.com/LMCache/LMCache/issues/1253)
*   **Contact**: [contact@lmcache.ai](mailto:contact@lmcache.ai)

## Contribute

We welcome contributions! Check out the [Contributing Guide](CONTRIBUTING.md). See also [[Onboarding] Welcoming contributors with good first issues!](https://github.com/LMCache/LMCache/issues/627)

## Citation

If you use LMCache in your research, please cite the following papers: (citations included)

## Social Media

*   [LinkedIn](https://www.linkedin.com/company/lmcache-lab/?viewAsMember=true)
*   [Twitter](https://x.com/lmcache)
*   [YouTube](https://www.youtube.com/@LMCacheTeam)

## License

LMCache is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.