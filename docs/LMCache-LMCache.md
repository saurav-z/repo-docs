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

---

[**LMCache**](https://github.com/LMCache/LMCache) is a groundbreaking solution that dramatically accelerates LLM serving by efficiently caching and reusing KV caches.

**Key Features of LMCache:**

*   **Reduced TTFT & Increased Throughput:** Significantly lowers the time-to-first-token and boosts LLM serving throughput, especially in long-context scenarios.
*   **KV Cache Reuse:**  Leverages KV caches of reused text across various locations (GPU, CPU DRAM, Local Disk) for efficient resource utilization.
*   **vLLM Integration:** Seamlessly integrates with vLLM v1, offering high-performance CPU KVCache offloading, disaggregated prefill, and P2P KVCache sharing.
*   **Wide Support:** Compatible with the [vLLM Production Stack](https://github.com/vllm-project/production-stack/), [llm-d](https://github.com/llm-d/llm-d/), and [KServe](https://github.com/kserve/kserve).
*   **Non-Prefix Cache Support:** Provides stable support for non-prefix KV caches.
*   **Flexible Storage:** Supports storage on CPU, Disk, and [NIXL](https://github.com/ai-dynamo/nixl).
*   **Easy Installation:** Simple installation via pip.

## Installation

Install LMCache using pip:

```bash
pip install lmcache
```

(Works on Linux NVIDIA GPU platform)

For more detailed installation instructions, see the [documentation](https://docs.lmcache.ai/getting_started/installation).

## Getting Started

Explore our [Quickstart Examples](https://docs.lmcache.ai/getting_started/quickstart/) in the documentation to begin using LMCache.

## Documentation

Access comprehensive documentation at [https://docs.lmcache.ai/](https://docs.lmcache.ai/).

## Examples

Explore practical use cases with our [examples](https://github.com/LMCache/LMCache/tree/dev/examples).

## Connect with Us

*   [Interest Form](https://forms.gle/MHwLiYDU6kcW3dLj7)
*   [Newsletter](https://mailchi.mp/tensormesh/lmcache-sign-up-newsletter)
*   [Slack](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-2viziwhue-5Amprc9k5hcIdXT7XevTaQ)
*   [Website](https://lmcache.ai/)
*   [Email](contact@lmcache.ai)

## Community Meetings

Join our weekly community meetings to discuss LMCache:

*   **Weekly Schedule (Alternating):**
    *   Tuesdays at 9:00 AM PT - [Add to Calendar](https://drive.google.com/file/d/15Xz8-LtpBQ5QgR7KrorOOyfuohCFQmwn/view?usp=drive_link)
    *   Tuesdays at 6:30 PM PT - [Add to Calendar](https://drive.google.com/file/d/1WMZNFXV24kWzprDjvO-jQ7mOY7whqEdG/view?usp=drive_link)
*   Meeting notes are available [here](https://docs.google.com/document/d/1_Fl3vLtERFa3vTH00cezri78NihNBtSClK-_1tSrcow).
*   Meeting recordings are on the [YouTube LMCache channel](https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA).

## Contributing

We welcome contributions!  See the [Contributing Guide](CONTRIBUTING.md).

## Citation

If you use LMCache in your research, please cite our papers:
*(citations included in original README)*

## Social Media

*   [LinkedIn](https://www.linkedin.com/company/lmcache-lab/?viewAsMember=true)
*   [Twitter](https://x.com/lmcache)
*   [YouTube](https://www.youtube.com/@LMCacheTeam)

## License

LMCache is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file.
```
Key improvements and SEO enhancements:

*   **Clear, concise opening sentence.** Immediately grabs attention and states the core value proposition.
*   **SEO-friendly headings:**  Uses relevant keywords like "LLM," "KV Cache," "TTFT," and "Throughput" in headings.
*   **Bulleted key features:** Makes the most important aspects of LMCache easy to scan and understand.
*   **Concise descriptions:** Keeps feature descriptions brief and informative.
*   **Links to relevant resources:** Includes links to documentation, examples, and community resources.
*   **Complete information:** Retains all the useful information from the original README.
*   **Call to action:** Encourages users to explore, connect, and contribute.
*   **Keywords**: Optimized for relevant keywords throughout the text to improve searchability.
*   **Structure:** Uses clear formatting for readability and SEO benefits.