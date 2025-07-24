<div align="center">
  <a href="https://memos.openmem.net/">
    <img src="https://statics.memtensor.com.cn/memos/memos-banner.gif" alt="MemOS Banner">
  </a>

<h1 align="center">
  <img src="https://statics.memtensor.com.cn/logo/memos_color_m.png" alt="MemOS Logo" width="50"/> MemOS: The Memory Operating System for Large Language Models  <img src="https://img.shields.io/badge/status-Preview-blue" alt="Preview Badge"/>
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

**Unlock the full potential of your Large Language Models with MemOS, the cutting-edge memory operating system designed to revolutionize LLM performance.**  MemOS empowers LLMs with robust long-term memory capabilities, enabling context-aware, consistent, and personalized interactions.

**Key Features:**

*   🧠 **Memory-Augmented Generation (MAG):** A unified API seamlessly integrates with LLMs, enhancing chat and reasoning through contextual memory retrieval.
*   📦 **Modular Memory Architecture (MemCube):** A flexible and modular design allows easy integration and management of diverse memory types.
*   💾 **Multiple Memory Types:**
    *   Textual Memory: Efficiently stores and retrieves unstructured or structured text data.
    *   Activation Memory: Caches key-value pairs (`KVCacheMemory`) to accelerate LLM inference and reuse context.
    *   Parametric Memory: Stores model adaptation parameters (e.g., LoRA weights).
*   🔌 **Extensible:** Easily customize and extend memory modules, data sources, and LLM integrations to fit your specific needs.

**Performance Benchmarks:**

MemOS delivers significant performance improvements over baseline memory solutions in various reasoning tasks.

| Model       | Avg. Score | Multi-Hop | Open Domain | Single-Hop | Temporal Reasoning |
|-------------|------------|-----------|-------------|------------|---------------------|
| **OpenAI**  | 0.5275     | 0.6028    | 0.3299      | 0.6183     | 0.2825              |
| **MemOS**   | **0.7331** | **0.6430** | **0.5521**   | **0.7844** | **0.7321**          |
| **Improvement** | **+38.98%** | **+6.67%** | **+67.35%** | **+26.86%** | **+159.15%**       |

> 💡 **MemOS demonstrates up to a 159% improvement in temporal reasoning accuracy compared to the OpenAI baseline.**

### End-to-End Evaluation on LOCOMO

> [!NOTE]
> MemOS consistently outperforms baseline methods (LangMem, Zep, OpenAI, Mem0) across all task types in the LOCOMO benchmark.

<img src="https://statics.memtensor.com.cn/memos/score_all_end2end.jpg" alt="END2END SCORE">

**Getting Started:**

Here's how to quickly start using MemOS:

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

**Installation:**

### Install via pip

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

Example installations:

```bash
pip install MemoryOS[tree-mem]
pip install MemoryOS[tree-mem,mem-reader]
pip install MemoryOS[mem-scheduler]
pip install MemoryOS[tree-mem,mem-reader,mem-scheduler]
```

### External Dependencies

#### Ollama Support

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### Transformers Support

Ensure PyTorch is installed (CUDA recommended for GPU acceleration).

#### Download Examples

```bash
memos download_examples
```

**Community & Support:**

*   **GitHub Issues:** Report bugs or suggest features on <a href="https://github.com/MemTensor/MemOS/issues" target="_blank">GitHub</a>.
*   **GitHub Pull Requests:** Contribute code improvements via <a href="https://github.com/MemTensor/MemOS/pulls" target="_blank">Pull Requests</a>.
*   **GitHub Discussions:** Join our <a href="https://github.com/MemTensor/MemOS/discussions" target="_blank">GitHub Discussions</a>.
*   **Discord:** Connect with us on our <a href="https://discord.gg/Txbx3gebZR" target="_blank">Discord Server</a>.
*   **WeChat:** Scan the QR code to join our WeChat group.

<img src="https://statics.memtensor.com.cn/memos/qr-code.png" alt="QR Code" width="600">

**Citation:**

If you use MemOS in your research, please cite our papers.

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

Contribute to MemOS! See the [contribution guidelines](https://memos-docs.openmem.net/contribution/overview) on our documentation.

**License:**

MemOS is licensed under the [Apache 2.0 License](./LICENSE).

**News:**

*   **2025-07-07** – 🎉 *MemOS 1.0 (Stellar) Preview Release*: The SOTA Memory OS for LLMs is now open-sourced.
*   **2025-07-04** – 🎉 *MemOS Paper Released*: [MemOS: A Memory OS for AI System](https://arxiv.org/abs/2507.03724) was published on arXiv.
*   **2025-05-28** – 🎉 *Short Paper Uploaded*: [MemOS: An Operating System for Memory-Augmented Generation (MAG) in Large Language Models](https://arxiv.org/abs/2505.22101) was published on arXiv.
*   **2024-07-04** – 🎉 *Memory3 Model Released at WAIC 2024*: The new memory-layered architecture model was unveiled at the 2024 World Artificial Intelligence Conference.
*   **2024-07-01** – 🎉 *Memory3 Paper Released*: [Memory3: Language Modeling with Explicit Memory](https://arxiv.org/abs/2407.01178) introduces the new approach to structured memory in LLMs.

[Back to top](#)  (Link to the original repo:  https://github.com/MemTensor/MemOS)
```
Key improvements and explanations:

*   **SEO Optimization:**  Added relevant keywords like "Memory Operating System," "Large Language Models," and "LLMs" throughout the text and in the headings.  The repeated use of "Memory" is also intentional for SEO.
*   **Clear Headings:**  Used clear and descriptive headings (e.g., "Key Features," "Performance Benchmarks," "Getting Started") to improve readability and SEO.
*   **Concise Summary Hook:**  Started with a strong, one-sentence hook that immediately grabs attention and explains the core value proposition.
*   **Bulleted Key Features:**  Uses bullet points to clearly present the core functionalities of MemOS.
*   **Performance Section Enhanced:** Added the most important results from the original README to quickly highlight key improvements.
*   **Clear Installation Instructions:**  Made installation steps easier to follow and added optional dependencies.
*   **Community and Support:** Provides ways to find support, making the project more welcoming for new users and potential contributors.
*   **Contributing and License:** Added these sections to highlight the openness of the project.
*   **News Section:**  The news section is very important for showcasing how active the project is.
*   **Hyperlinks:** Made sure all of the links were accurate.
*   **Concise Language:**  Used more concise language to quickly convey information.
*   **Removed Redundancy:** Removed any redundant information to keep the README concise.
*   **Clear Formatting:**  Cleaned up the formatting for enhanced readability.
*  **Back to Top link:** added a link back to the original repo, as per the prompt.
* **Bolded key points**: Made important data, model names, and features in bold to quickly help the reader grasp key points.
*   **Removed unnecessary information:** Removed the "Details of End-to-End Evaluation on LOCOMO" heading as it's redundant since the image is already present.

This revised README is more engaging, informative, and optimized for search engines, making it easier for users to understand and adopt MemOS.