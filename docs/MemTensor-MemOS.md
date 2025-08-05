<div align="center">
  <a href="https://memos.openmem.net/">
    <img src="https://statics.memtensor.com.cn/memos/memos-banner.gif" alt="MemOS Banner">
  </a>

<h1 align="center">
  <img src="https://statics.memtensor.com.cn/logo/memos_color_m.png" alt="MemOS Logo" width="50"/> MemOS 1.0: Stellar - The Memory Operating System for LLMs  <img src="https://img.shields.io/badge/status-Preview-blue" alt="Preview Badge"/>
</h1>

  <p>
    <a href="https://www.memtensor.com.cn/">
      <img alt="Static Badge" src="https://img.shields.io/badge/Maintained_by-MemTensor-blue">
    </a>
    <a href="https://pypi.org/project/MemoryOS">
      <img src="https://img.shields.io/pypi/v/MemoryOS?label=pypi%20package" alt="PyPI Version">
    </a>
    <a href="https://pypi.org/project/MemoryOS">
      <img src="https://img.shields.io/pypi/pyversions/MemoryOS.svg" alt="Supported Python versions">
    </a>
    <a href="https://pypi.org/project/MemoryOS">
      <img src="https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey" alt="Supported Platforms">
    </a>
    <a href="https://memos-docs.openmem.net/home/overview/">
      <img src="https://img.shields.io/badge/Documentation-view-blue.svg" alt="Documentation">
    </a>
    <a href="https://arxiv.org/abs/2507.03724">
        <img src="https://img.shields.io/badge/arXiv-2507.03724-b31b1b.svg" alt="ArXiv Paper">
    </a>
    <a href="https://github.com/MemTensor/MemOS/discussions">
      <img src="https://img.shields.io/badge/GitHub-Discussions-181717.svg?logo=github" alt="GitHub Discussions">
    </a>
    <a href="https://discord.gg/Txbx3gebZR">
      <img src="https://img.shields.io/badge/Discord-join%20chat-7289DA.svg?logo=discord" alt="Discord">
    </a>
    <a href="https://statics.memtensor.com.cn/memos/qr-code.png">
      <img src="https://img.shields.io/badge/WeChat-Group-07C160.svg?logo=wechat" alt="WeChat Group">
    </a>
    <a href="https://opensource.org/license/apache-2-0/">
      <img src="https://img.shields.io/badge/License-Apache_2.0-green.svg?logo=apache" alt="License">
    </a>
  </p>
</div>

---

<img src="https://statics.memtensor.com.cn/memos/sota_score.jpg" alt="SOTA SCORE">

**MemOS (Memory Operating System) revolutionizes Large Language Models by equipping them with advanced long-term memory capabilities, enabling superior context awareness and performance.**

**Key Features:**

*   **Memory-Augmented Generation (MAG):** Unified API for memory operations, seamlessly integrates with LLMs, enabling enhanced chat and reasoning capabilities through contextual memory retrieval.
*   **Modular Memory Architecture (MemCube):** Flexible and modular design for easy integration and management of diverse memory types.
*   **Multiple Memory Types:**
    *   **Textual Memory:** Stores and retrieves unstructured or structured text knowledge.
    *   **Activation Memory:** Employs key-value pairs (`KVCacheMemory`) to accelerate LLM inference and context reuse.
    *   **Parametric Memory:** Preserves model adaptation parameters (e.g., LoRA weights).
*   **Extensible Design:** Easily expand and adapt memory modules, data sources, and LLM integrations.

**Performance Benchmarks:**

MemOS demonstrates significant performance gains over baseline memory solutions:

| Model          | Avg. Score | Multi-Hop | Open Domain | Single-Hop | Temporal Reasoning |
| :------------- | :--------- | :-------- | :---------- | :--------- | :----------------- |
| **OpenAI**     | 0.5275     | 0.6028    | 0.3299      | 0.6183     | 0.2825             |
| **MemOS**      | **0.7331** | **0.6430** | **0.5521**  | **0.7844** | **0.7321**         |
| **Improvement** | **+38.98%** | **+6.67%** | **+67.35%** | **+26.86%** | **+159.15%**       |

> 💡 **Temporal reasoning accuracy improved by 159% compared to the OpenAI baseline.**

### End-to-End Evaluation on LOCOMO

> [!NOTE]
> Comparison of LLM Judge Scores across five major tasks in the LOCOMO benchmark. Each bar shows the mean evaluation score judged by LLMs for a given method-task pair, with standard deviation as error bars. MemOS-0630 consistently outperforms baseline methods (LangMem, Zep, OpenAI, Mem0) across all task types, especially in multi-hop and temporal reasoning scenarios.

<img src="https://statics.memtensor.com.cn/memos/score_all_end2end.jpg" alt="END2END SCORE">

**Getting Started:**

Here's a quick demonstration of creating a `MemCube`, loading it, accessing memories, and saving it.

```python
from memos.mem_cube.general import GeneralMemCube

# Initialize a MemCube from a local directory
mem_cube = GeneralMemCube.init_from_dir("examples/data/mem_cube_2")

# Access and print all memories
print("--- Textual Memories ---")
for item in mem_cube.text_mem.get_all():
    print(item)

print("\n--- Activation Memories ---")
for item in mem_cube.act_mem.get_all():
    print(item)

# Save the MemCube to a new directory
mem_cube.dump("tmp/mem_cube")
```

**Using the Memory Operating System (MOS):**

The `MOS` is a high-level orchestration layer that manages multiple MemCubes.

```python
from memos.configs.mem_os import MOSConfig
from memos.mem_os.main import MOS

# init MOS
mos_config = MOSConfig.from_json_file("examples/data/config/simple_memos_config.json")
memory = MOS(mos_config)

# create user
user_id = "b41a34d5-5cae-4b46-8c49-d03794d206f5"
memory.create_user(user_id=user_id)

# register cube for user
memory.register_mem_cube("examples/data/mem_cube_2", user_id=user_id)

# add memory for user
memory.add(
    messages=[
        {"role": "user", "content": "I like playing football."},
        {"role": "assistant", "content": "I like playing football too."},
    ],
    user_id=user_id,
)

# Later, when you want to retrieve memory for user
retrieved_memories = memory.search(query="What do you like?", user_id=user_id)
# output text_memories: I like playing football, act_memories, para_memories
print(f"text_memories: {retrieved_memories['text_mem']}")
```

Explore the [`examples`](./examples) directory for detailed guidance.

**Installation:**

### Install MemOS via pip

```bash
pip install MemoryOS
```

### Optional Dependencies

Install these for advanced features

| Feature               | Package Name              |
| --------------------- | ------------------------- |
| Tree Memory           | `MemoryOS[tree-mem]`      |
| Memory Reader         | `MemoryOS[mem-reader]`    |
| Memory Scheduler      | `MemoryOS[mem-scheduler]` |

Example:
```bash
pip install MemoryOS[tree-mem]
pip install MemoryOS[tree-mem,mem-reader]
pip install MemoryOS[mem-scheduler]
pip install MemoryOS[tree-mem,mem-reader,mem-scheduler]
```

### External Dependencies

#### Ollama Support
Install [Ollama](https://ollama.com/) CLI:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### Transformers Support

Ensure [PyTorch](https://pytorch.org/get-started/locally/) is installed (CUDA is recommended).

#### Download Examples

Run:

```bash
memos download_examples
```

**Community and Support:**

*   **GitHub Issues:** Report issues or suggest features on [GitHub Issues](https://github.com/MemTensor/MemOS/issues).
*   **GitHub Pull Requests:** Contribute code through [Pull Requests](https://github.com/MemTensor/MemOS/pulls).
*   **GitHub Discussions:** Engage in discussions on [GitHub Discussions](https://github.com/MemTensor/MemOS/discussions).
*   **Discord:** Join our [Discord Server](https://discord.gg/Txbx3gebZR).
*   **WeChat:** Scan the QR code to join the WeChat group.

<img src="https://statics.memtensor.com.cn/memos/qr-code.png" alt="QR Code" width="600">

**Citation:**

Cite our papers if you use MemOS in your research:

```bibtex
@article{li2025memos_long,
  title={MemOS: A Memory OS for AI System},
  author={Li, Zhiyu and Song, Shichao and Xi, Chenyang and Wang, Hanyu and Tang, Chen and Niu, Simin and Chen, Ding and Yang, Jiawei and Li, Chunyu and Yu, Qingchen and Zhao, Jihao and Wang, Yezhaohui and Liu, Peng and Lin, Zehao and Wang, Pengyuan and Huo, Jiahao and Chen, Tianyi and Chen, Kai and Li, Kehang and Tao, Zhen and Ren, Junpeng and Lai, Huayi and Wu, Hao and Tang, Bo and Wang, Zhenren and Fan, Zhaoxin and Zhang, Ningyu and Zhang, Linfeng and Yan, Junchi and Yang, Mingchuan and Xu, Tong and Xu, Wei and Chen, Huajun and Wang, Haofeng and Yang, Hongkang and Zhang, Wentao and Xu, Zhi-Qin John and Chen, Siheng and Xiong, Feiyu},
  journal={arXiv preprint arXiv:2507.03724},
  year={2025},
  url={https://arxiv.org/abs/2507.03724}
}

@article{li2025memos_short,
  title={MemOS: An Operating System for Memory-Augmented Generation (MAG) in Large Language Models},
  author={Li, Zhiyu and Song, Shichao and Wang, Hanyu and Niu, Simin and Chen, Ding and Yang, Jiawei and Xi, Chenyang and Lai, Huayi and Zhao, Jihao and Wang, Yezhaohui and others},
  journal={arXiv preprint arXiv:2505.22101},
  year={2025},
  url={https://arxiv.org/abs/2505.22101}
}

@article{yang2024memory3,
author = {Yang, Hongkang and Zehao, Lin and Wenjin, Wang and Wu, Hao and Zhiyu, Li and Tang, Bo and Wenqiang, Wei and Wang, Jinbo and Zeyun, Tang and Song, Shichao and Xi, Chenyang and Yu, Yu and Kai, Chen and Xiong, Feiyu and Tang, Linpeng and Weinan, E},
title = {Memory$^3$: Language Modeling with Explicit Memory},
journal = {Journal of Machine Learning},
year = {2024},
volume = {3},
number = {3},
pages = {300--346},
issn = {2790-2048},
doi = {https://doi.org/10.4208/jml.240708},
url = {https://global-sci.com/article/91443/memory3-language-modeling-with-explicit-memory}
}
```

**Contributing:**

We welcome contributions!  See our [contribution guidelines](https://memos-docs.openmem.net/contribution/overview).

**License:**

MemOS is licensed under the [Apache 2.0 License](./LICENSE).

**News:**

*   **2025-07-07** – 🎉 *MemOS 1.0 (Stellar) Preview Release*:  A SOTA Memory OS for LLMs is now open-sourced.
*   **2025-07-04** – 🎉 *MemOS Paper Released*: [MemOS: A Memory OS for AI System](https://arxiv.org/abs/2507.03724) was published on arXiv.
*   **2025-05-28** – 🎉 *Short Paper Uploaded*: [MemOS: An Operating System for Memory-Augmented Generation (MAG) in Large Language Models](https://arxiv.org/abs/2505.22101) was published on arXiv.
*   **2024-07-04** – 🎉 *Memory3 Model Released at WAIC 2024*: The new memory-layered architecture model was unveiled at the 2024 World Artificial Intelligence Conference.
*   **2024-07-01** – 🎉 *Memory3 Paper Released*: [Memory3: Language Modeling with Explicit Memory](https://arxiv.org/abs/2407.01178) introduces the new approach to structured memory in LLMs.

[Visit the original repository on GitHub](https://github.com/MemTensor/MemOS) for more information.