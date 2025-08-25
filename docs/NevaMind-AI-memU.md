<!--
  SPDX-License-Identifier: Apache-2.0
-->

<div align="center">
  <img src="assets/banner.png" alt="MemU Banner" />
</div>

# MemU: The Next-Generation Memory Framework for AI Companions

**Build AI companions that truly remember with MemU, the open-source memory framework designed for high accuracy, fast retrieval, and cost-effectiveness.** ([See the original repo](https://github.com/NevaMind-AI/memU))

[![PyPI version](https://badge.fury.io/py/memu-py.svg)](https://badge.fury.io/py/memu-py)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/memu)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?logo=x&logoColor=white)](https://x.com/memU_ai)

---

## Key Features

*   **AI Companion Specialization:** Tailored for AI companion applications.
*   **Unmatched Accuracy:** Achieves a state-of-the-art 92% accuracy in the Locomo benchmark.
*   **Cost-Effective:** Reduces costs by up to 90% through optimized online platform.
*   **Advanced Retrieval:** Leverages semantic search, hybrid search, and contextual retrieval for optimal results.
*   **24/7 Enterprise Support:** Dedicated support for enterprise customers.

---

## Why Choose MemU?

MemU is designed to provide AI Companions with exceptional memory capabilities.  It excels in several key areas:

*   **Superior Accuracy:** MemU leads the industry with a 92% average accuracy in the Locomo dataset, significantly surpassing the competition.
*   **Rapid Retrieval:**  Efficiently retrieves relevant information by categorizing and indexing memories, eliminating the need for extensive embedding searches.
*   **Cost Efficiency:**  Optimized to process numerous conversation turns at once, reducing token usage and minimizing costs.

---

## Core Concepts

MemU's architecture is centered around an intelligent memory system:

*   **Autonomous Memory File Management:** Organized like a file system, with a memory agent automatically deciding what to record, modify, or archive.
*   **Interconnected Knowledge Graph:** Automatically creates links between related memories, facilitating effortless recall.
*   **Continuous Self-Improvement:** The memory agent generates insights, identifies patterns, and creates summaries, enhancing the knowledge base over time.
*   **Adaptive Forgetting Mechanism:** Prioritizes frequently accessed memories while deprioritizing or forgetting less relevant content, optimizing performance.

---

## Get Started

MemU offers several integration options to suit your needs:

### Cloud Version ([Online Platform](https://app.memu.so))

The quickest way to begin integrating AI memories.

*   **Instant Access:** Integrate AI memories in minutes.
*   **Managed Infrastructure:** We handle scaling, updates, and maintenance.
*   **Premium Support:** Priority assistance from our engineering team with a paid subscription.

**Quick Integration Steps:**

1.  **Create Account:** Sign up at [https://app.memu.so](https://app.memu.so).
2.  **Generate API Key:** Get your API key from [https://app.memu.so/api-key/](https://app.memu.so/api-key/).
3.  **Add Code:**

    ```python
    pip install memu-py

    # Example usage
    from memu import MemuClient

    # Initialize
    memu_client = MemuClient(
        base_url="https://api.memu.so",
        api_key=os.getenv("MEMU_API_KEY")
    )
    memu_client.memorize_conversation(
        conversation=conversation_text,  # Recommend longer conversation (~8000 tokens), see https://memu.pro/blog/memu-best-practice for details
        user_id="user001",
        user_name="User",
        agent_id="assistant001",
        agent_name="Assistant"
    )
    ```

    See [`example/client/memory.py`](example/client/memory.py) for complete integration details.
    Check [API reference](docs/API_REFERENCE.md) or [our blog](https://memu.pro/blog) for more details.

### Enterprise Edition

For organizations needing advanced features:

*   **Commercial License:** Proprietary features and commercial usage rights.
*   **Custom Development:**  SSO/RBAC integration, custom algorithm optimization.
*   **Intelligence & Analytics:**  User behavior analysis, real-time monitoring, automated agent optimization.
*   **Premium Support:** 24/7 dedicated support and professional implementation services.

**Enterprise Inquiries:** [contact@nevamind.ai](mailto:contact@nevamind.ai)

### Self-Hosting (Community Edition)

For users who prefer local control and data privacy. See [self hosting README](README.self_host.md)

*   **Data Privacy:** Keep sensitive data within your infrastructure.
*   **Customization:** Modify and extend the platform.
*   **Cost Control:** Avoid recurring cloud fees for large deployments.

---

## Demo Video

<div align="left">
  <a href="https://www.youtube.com/watch?v=qZIuCoLglHs">
    <img src="https://img.youtube.com/vi/ueOe4ZPlZLU/maxresdefault.jpg" alt="MemU Demo Video" width="600">
  </a>
  <br>
  <em>Click to watch the MemU demonstration video</em>
</div>

---

## Advantages

*   **Higher Memory Accuracy:** Achieves state-of-the-art performance. [Benchmark image here]
*   **Fast Retrieval:** Optimized for quick and efficient data retrieval.
*   **Low Cost:** Designed to minimize token usage and operational expenses.

---

## Use Cases

|                                       |                                       |                                 |                                         |
| :-----------------------------------: | :-----------------------------------: | :-----------------------------: | :-------------------------------------: |
| <img src="assets/usecase/ai_companion-0000.jpg" width="150" height="200"><br>**AI Companion** | <img src="assets/usecase/ai_role_play-0000.jpg" width="150" height="200"><br>**AI Role Play** | <img src="assets/usecase/ai_ip-0000.png" width="150" height="200"><br>**AI IP Characters** | <img src="assets/usecase/ai_edu-0000.jpg" width="150" height="200"><br>**AI Education** |
| <img src="assets/usecase/ai_therapy-0000.jpg" width="150" height="200"><br>**AI Therapy** | <img src="assets/usecase/ai_robot-0000.jpg" width="150" height="200"><br>**AI Robot** | <img src="assets/usecase/ai_creation-0000.jpg" width="150" height="200"><br>**AI Creation** | More...|

---

## Contributing

Join our community and help shape the future of MemU.  Review the [Contributing Guide](CONTRIBUTING.md) to get started.

---

## License

All contributions are licensed under the **Apache License 2.0**.

---

## Community

*   **GitHub Issues:**  Report bugs, request features, and track development. [Submit an issue](https://github.com/NevaMind-AI/memU/issues)
*   **Discord:** Get real-time support and connect with the community. [Join us](https://discord.com/invite/hQZntfGsbJ)
*   **X (Twitter):** Follow us for updates and announcements. [Follow us](https://x.com/memU_ai)

---

## Ecosystem

We're proud to collaborate with:

<div align="center">

### Development Tools
<a href="https://github.com/TEN-framework/ten-framework"><img src="https://avatars.githubusercontent.com/u/113095513?s=200&v=4" alt="Ten" height="40" style="margin: 10px;"></a>
<a href="https://github.com/camel-ai/camel"><img src="https://avatars.githubusercontent.com/u/134388954?s=200&v=4" alt="Camel" height="40" style="margin: 10px;"></a>
<a href="https://github.com/eigent-ai/eigent"><img src="https://www.eigent.ai/nav/logo_icon.svg" alt="Eigent" height="40" style="margin: 10px;"></a>
<a href="https://github.com/milvus-io/milvus"><img src="https://miro.medium.com/v2/resize:fit:2400/1*-VEGyAgcIBD62XtZWavy8w.png" alt="Ten" height="40" style="margin: 10px;"></a>
<a href="https://xroute.ai/"><img src="assets/partners/xroute.png" alt="xRoute" height="40" style="margin: 10px;"></a>
<a href="https://jaaz.app/"><img src="assets/partners/jazz.png" alt="jazz" height="40" style="margin: 10px;"></a>
<a href="https://github.com/Buddie-AI/Buddie"><img src="assets/partners/buddie.png" alt="buddie" height="40" style="margin: 10px;"></a>
<a href="https://github.com/bytebase/bytebase"><img src="assets/partners/bytebase.png" alt="bytebase" height="40" style="margin: 10px;"></a>
</div>

---

*Interested in partnering with MemU? Contact us at [contact@nevamind.ai](mailto:contact@nevamind.ai)*

---

## Join Our WeChat Community

<div align="center">
<img src="assets/qrcode.png" alt="MemU WeChat and discord QR Code" width="480" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px;">
</div>

---

## Questionnaire

Provide feedback in our 3-min survey for a chance to receive free credits: https://forms.gle/H2ZuZVHv72xbqjvd7