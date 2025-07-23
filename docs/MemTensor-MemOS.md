<div align="center">
  <a href="https://memos.openmem.net/">
    <img src="https://statics.memtensor.com.cn/memos/memos-banner.gif" alt="MemOS Banner">
  </a>
</div>

# MemOS: The Memory Operating System for Large Language Models

MemOS is a cutting-edge operating system designed to equip Large Language Models (LLMs) with long-term memory, transforming how they process and interact with information.  **(Click the image for the original repo.)**

<div align="center">
  <a href="https://github.com/MemTensor/MemOS">
    <img src="https://statics.memtensor.com.cn/logo/memos_color_m.png" alt="MemOS Logo" width="50"/>
  </a>
</div>

[![Status](https://img.shields.io/badge/status-Preview-blue)](https://github.com/MemTensor/MemOS)
[![Maintained by](https://img.shields.io/badge/Maintained_by-MemTensor-blue)](https://www.memtensor.com.cn/)
[![PyPI Version](https://img.shields.io/pypi/v/MemoryOS?label=pypi%20package)](https://pypi.org/project/MemoryOS)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/MemoryOS.svg)](https://pypi.org/project/MemoryOS)
[![Supported Platforms](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)](https://pypi.org/project/MemoryOS)
[![Documentation](https://img.shields.io/badge/Documentation-view-blue.svg)](https://memos-docs.openmem.net/home/overview/)
[![ArXiv Paper](https://img.shields.io/badge/arXiv-2507.03724-b31b1b.svg)](https://arxiv.org/abs/2507.03724)
[![GitHub Discussions](https://img.shields.io/badge/GitHub-Discussions-181717.svg?logo=github)](https://github.com/MemTensor/MemOS/discussions)
[![Discord](https://img.shields.io/badge/Discord-join%20chat-7289DA.svg?logo=discord)](https://discord.gg/Txbx3gebZR)
[![WeChat Group](https://img.shields.io/badge/WeChat-Group-07C160.svg?logo=wechat)](https://statics.memtensor.com.cn/memos/qr-code.png)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg?logo=apache)](https://opensource.org/license/apache-2-0/)

---

<img src="https://statics.memtensor.com.cn/memos/sota_score.jpg" alt="SOTA SCORE">

## Key Features

*   **Memory-Augmented Generation (MAG):**  A unified API for seamless memory integration, enhancing LLM chat and reasoning.
*   **Modular Memory Architecture (MemCube):**  Flexible design for easy integration and management of various memory types.
*   **Multiple Memory Types:**
    *   **Textual Memory:** Store and retrieve textual knowledge.
    *   **Activation Memory:** Accelerate LLM inference using `KVCacheMemory`.
    *   **Parametric Memory:** Store model adaptation parameters (e.g., LoRA weights).
*   **Extensible:** Customize memory modules, data sources, and LLM integrations.

## Performance Benchmark

MemOS demonstrates significant improvements over baseline memory solutions in multiple reasoning tasks.

| Model       | Avg. Score | Multi-Hop | Open Domain | Single-Hop | Temporal Reasoning |
|-------------|------------|-----------|-------------|------------|---------------------|
| **OpenAI**  | 0.5275     | 0.6028    | 0.3299      | 0.6183     | 0.2825              |
| **MemOS**   | **0.7331** | **0.6430** | **0.5521**   | **0.7844** | **0.7321**          |
| **Improvement** | **+38.98%** | **+6.67%** | **+67.35%** | **+26.86%** | **+159.15%**       |

> 💡 **Temporal reasoning accuracy improved by 159% compared to the OpenAI baseline.**

### Details of End-to-End Evaluation on LOCOMO

> [!NOTE]
> Comparison of LLM Judge Scores across five major tasks in the LOCOMO benchmark. Each bar shows the mean evaluation score judged by LLMs for a given method-task pair, with standard deviation as error bars. MemOS-0630 consistently outperforms baseline methods (LangMem, Zep, OpenAI, Mem0) across all task types, especially in multi-hop and temporal reasoning scenarios.

<img src="https://statics.memtensor.com.cn/memos/score_all_end2end.jpg" alt="END2END SCORE">

## Getting Started

### Example Code
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

### Using MOS (Memory Operating System):
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

Find more detailed examples in the [`examples`](./examples) directory.

## Installation

### Install MemOS using pip

```bash
pip install MemoryOS
```

### Optional Dependencies

Install optional dependencies for specific features:

| Feature               | Package Name              |
| --------------------- | ------------------------- |
| Tree Memory           | `MemoryOS[tree-mem]`      |
| Memory Reader         | `MemoryOS[mem-reader]`    |
| Memory Scheduler      | `MemoryOS[mem-scheduler]` |

Example:

```bash
pip install MemoryOS[tree-mem,mem-reader]
```

### External Dependencies

#### Ollama Support

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### Transformers Support

Install [PyTorch](https://pytorch.org/get-started/locally/) (CUDA recommended).

#### Download Examples

```bash
memos download_examples
```

## Community & Support

*   **GitHub Issues:** Report bugs or request features. [GitHub Issues](https://github.com/MemTensor/MemOS/issues)
*   **GitHub Pull Requests:** Contribute code. [Pull Requests](https://github.com/MemTensor/MemOS/pulls)
*   **GitHub Discussions:** Ask questions and share ideas. [GitHub Discussions](https://github.com/MemTensor/MemOS/discussions)
*   **Discord:** Join our community. [Discord Server](https://discord.gg/Txbx3gebZR)
*   **WeChat:** Join our WeChat group.

<img src="https://statics.memtensor.com.cn/memos/qr-code.png" alt="QR Code" width="600">

## Citation

Cite our research if you use MemOS:

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

## Contributing

Please review the [contribution guidelines](https://memos-docs.openmem.net/contribution/overview).

## License

This project is licensed under the [Apache 2.0 License](./LICENSE).

## News

*   **2025-07-07** – 🎉 *MemOS 1.0 (Stellar) Preview Release*: SOTA Memory OS for LLMs is open-sourced.
*   **2025-07-04** – 🎉 *MemOS Paper Released*: [MemOS: A Memory OS for AI System](https://arxiv.org/abs/2507.03724) on arXiv.
*   **2025-05-28** – 🎉 *Short Paper Uploaded*: [MemOS: An Operating System for Memory-Augmented Generation (MAG) in Large Language Models](https://arxiv.org/abs/2505.22101) on arXiv.
*   **2024-07-04** – 🎉 *Memory3 Model Released at WAIC 2024*.
*   **2024-07-01** – 🎉 *Memory3 Paper Released*: [Memory3: Language Modeling with Explicit Memory](https://arxiv.org/abs/2407.01178)