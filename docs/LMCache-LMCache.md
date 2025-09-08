# LMCache: Accelerate LLM Inference with Intelligent KV Cache Management

**Supercharge your LLM performance and reduce costs with LMCache, the KV cache extension that significantly boosts throughput and minimizes latency.**  ([View on GitHub](https://github.com/LMCache/LMCache))

---

## Key Features

*   **Enhanced Performance:** Dramatically reduces Time to First Token (TTFT) and increases throughput, especially in long-context scenarios.
*   **KV Cache Optimization:** Efficiently stores and reuses Key-Value (KV) caches across various locations (GPU, CPU DRAM, Disk), saving precious GPU cycles.
*   **Seamless Integration:** Works with vLLM (including high-performance CPU KVCache offloading, disaggregated prefill, and P2P KVCache sharing), and is supported in the vLLM production stack, llm-d, and KServe.
*   **Flexible Storage:** Supports KV cache storage on CPU, Disk, and NIXL.
*   **Non-Prefix Support:** Provides stable support for non-prefix KV caches.
*   **Easy Installation:**  Install via pip and integrates with the latest vLLM.

---

## Summary

LMCache is a powerful LLM serving engine extension that dramatically improves performance by optimizing KV cache management. It addresses the challenges of long-context scenarios and reduces GPU usage, making LLM deployment more efficient and cost-effective.  By reusing KV caches of any reused text across multiple instances, LMCache minimizes latency and maximizes throughput. Developers experience significant speed improvements, with 3-10x delay savings and GPU cycle reduction in various LLM use cases, including multi-round QA and RAG.

---

## Installation

```bash
pip install lmcache
```

Refer to the [documentation](https://docs.lmcache.ai/getting_started/installation) for detailed instructions and troubleshooting, especially if you're not using the latest stable vLLM version.

---

## Getting Started

Explore the [Quickstart Examples](https://docs.lmcache.ai/getting_started/quickstart/) in the documentation to quickly understand and implement LMCache.

---

## Resources

*   **Documentation:** [https://docs.lmcache.ai/](https://docs.lmcache.ai/)
*   **Blog:** [https://blog.lmcache.ai/](https://blog.lmcache.ai/)
*   **Examples:** [https://github.com/LMCache/LMCache/tree/dev/examples](https://github.com/LMCache/LMCache/tree/dev/examples)
*   **Community:**
    *   Join Slack: [https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-36x1m765z-8FgDA_73vcXtlZ_4XvpE6Q](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-36x1m765z-8FgDA_73vcXtlZ_4XvpE6Q)
    *   Community Meeting: Bi-weekly, Tuesdays at 9:00 AM PT - [Add to Calendar](https://drive.usercontent.google.com/u/0/uc?id=1f5EXbooGcwNwzIpTgn5u4PHqXgfypMtu&export=download)
    *   Meeting Notes: [https://docs.google.com/document/d/1_Fl3vLtERFa3vTH00cezri78NihNBtSClK-_1tSrcow](https://docs.google.com/document/d/1_Fl3vLtERFa3vTH00cezri78NihNBtSClK-_1tSrcow)
    *   YouTube Channel: [https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA](https://www.youtube.com/channel/UC58zMz55n70rtf1Ak2PULJA)
*   **Roadmap:** [https://github.com/LMCache/LMCache/issues/1253](https://github.com/LMCache/LMCache/issues/1253)
*   **Interest Form:** [https://forms.gle/MHwLiYDU6kcW3dLj7](https://forms.gle/MHwLiYDU6kcW3dLj7)

---

## Contributing

Contributions are welcome! See the [Contributing Guide](CONTRIBUTING.md) and the [Onboarding Issues](https://github.com/LMCache/LMCache/issues/627) for guidance.

---

## Citation

If you use LMCache for your research, please cite the following papers:

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

---

## Socials

*   **LinkedIn:** [https://www.linkedin.com/company/lmcache-lab/?viewAsMember=true](https://www.linkedin.com/company/lmcache-lab/?viewAsMember=true)
*   **Twitter:** [https://x.com/lmcache](https://x.com/lmcache)
*   **YouTube:** [https://www.youtube.com/@LMCacheTeam](https://www.youtube.com/@LMCacheTeam)

---

## License

LMCache is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.