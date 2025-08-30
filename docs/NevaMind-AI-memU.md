<div align="center">

![MemU Banner](assets/banner.png)

</div>

# MemU: The Next-Gen Memory Framework for AI Companions 🧠

**Empower your AI companions with MemU, the open-source memory framework for high accuracy, fast retrieval, and cost-effective AI interactions.** ([Back to the original repo](https://github.com/NevaMind-AI/memU))

[![PyPI version](https://badge.fury.io/py/memu-py.svg)](https://badge.fury.io/py/memu-py)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/memu)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?logo=x&logoColor=white)](https://x.com/memU_ai)

---

## Key Features of MemU

*   ✅ **AI Companion Specialization:** Designed specifically for AI companion applications, enabling them to remember and learn from interactions.
*   ✅ **92% Accuracy:** Achieve state-of-the-art performance, excelling in the Locomo benchmark.
*   ✅ **Up to 90% Cost Reduction:** Optimize your AI memory operations with our efficient platform.
*   ✅ **Advanced Retrieval Strategies:** Leverage multiple methods, including semantic search, hybrid search, and contextual retrieval, for optimal recall.
*   ✅ **24/7 Support (Enterprise):** Dedicated support for enterprise customers to ensure seamless integration and performance.

---

## Why Choose MemU?

MemU is built to provide AI companions with human-like memory, allowing them to build deeper relationships. It acts as an intelligent "memory folder" that adapts to different AI companion scenarios. With **memU**, you can build AI companions that truly remember you. They learn who you are, what you care about, and grow alongside you through every interaction.

*   **Higher Accuracy:** MemU achieves 92.09% average accuracy in the Locomo dataset.
*   **Faster Retrieval:** Eliminate the need for extensive embedding searches.
*   **Lower Cost:** Process hundreds of conversation turns at once, reducing token usage.

---

## Getting Started with MemU

### ☁️ **Cloud Version ([Online Platform](https://app.memu.so))**

The fastest way to integrate MemU into your application!

*   **Instant Access:** Integrate AI memories in minutes.
*   **Managed Infrastructure:** We handle scaling, updates, and maintenance.
*   **Premium Support:** Get priority assistance.

**How to Start:**

1.  **Create an Account:** Visit [https://app.memu.so](https://app.memu.so)
2.  **Generate API Keys:** Go to [https://app.memu.so/api-key/](https://app.memu.so/api-key/)
3.  **Install the Library:**
    ```bash
    pip install memu-py
    ```
4.  **Implement in your Code:**
    ```python
    from memu import MemuClient
    import os

    # Initialize
    memu_client = MemuClient(
        base_url="https://api.memu.so", 
        api_key=os.getenv("MEMU_API_KEY")
    )
    memu_client.memorize_conversation(
        conversation=conversation_text, # Recommend longer conversation (~8000 tokens), see https://memu.pro/blog/memu-best-practice for details
        user_id="user001", 
        user_name="User", 
        agent_id="assistant001", 
        agent_name="Assistant"
    )
    ```
    *   **See [`example/client/memory.py`](example/client/memory.py) for complete integration details.**
    *   **Check out the [API Reference](docs/API_REFERENCE.md) or [our blog](https://memu.pro/blog) for more details.**

### 🏢 Enterprise Edition

For organizations requiring maximum security, customization, control and best quality:

*   **Commercial License** - Full proprietary features, commercial usage rights, white-labeling options
*   **Custom Development** - SSO/RBAC integration, dedicated algorithm team for scenario-specific framework optimization
*   **Intelligence & Analytics** - User behavior analysis, real-time production monitoring, automated agent optimization
*   **Premium Support** - 24/7 dedicated support, custom SLAs, professional implementation services

**Enterprise Inquiries:** [contact@nevamind.ai](mailto:contact@nevamind.ai)

### 🏠 Self-Hosting (Community Edition)

For users and developers who prefer local control, data privacy, or customization:

*   **Data Privacy** - Keep sensitive data within your infrastructure
*   **Customization** - Modify and extend the platform to fit your needs
*   **Cost Control** - Avoid recurring cloud fees for large-scale deployments

See [self hosting README](README.self_host.md)

---

## **Memory Architecture**

### **Organize** - Autonomous Memory File Management

Your memories are structured as intelligent folders managed by a memory agent. We do not do explicit modeling for memories. The memory agent automatically decides what to record, modify, or archive. Think of it as having a personal librarian who knows exactly how to organize your thoughts.

### **Link** - Interconnected Knowledge Graph

Memories don't exist in isolation. Our system automatically creates meaningful connections between related memories, building a rich network of hyperlinked documents and transforming memory discovery from search into effortless recall.

### **Evolve** - Continuous Self-Improvement

Even when offline, your memory agent keeps working. It generates new insights by analyzing existing memories, identifies patterns, and creates summary documents through self-reflection. Your knowledge base becomes smarter over time, not just larger.

### **Never Forget** - Adaptive Forgetting Mechanism

The memory agent automatically prioritizes information based on usage patterns. Recently accessed memories remain highly accessible, while less relevant content is deprioritized or forgotten. This creates a personalized information hierarchy that evolves with your needs.

---

## 🎥 Demo Video

<div align="left">
  <a href="https://www.youtube.com/watch?v=qZIuCoLglHs">
    <img src="https://img.youtube.com/vi/ueOe4ZPlZLU/maxresdefault.jpg" alt="MemU Demo Video" width="600">
  </a>
  <br>
  <em>Click to watch the MemU demonstration video</em>
</div>

---

## 🎓 Use Cases

| | | | |
|:---:|:---:|:---:|:---:|
| <img src="assets/usecase/ai_companion-0000.jpg" width="150" height="200"><br>**AI Companion** | <img src="assets/usecase/ai_role_play-0000.jpg" width="150" height="200"><br>**AI Role Play** | <img src="assets/usecase/ai_ip-0000.png" width="150" height="200"><br>**AI IP Characters** | <img src="assets/usecase/ai_edu-0000.jpg" width="150" height="200"><br>**AI Education** |
| <img src="assets/usecase/ai_therapy-0000.jpg" width="150" height="200"><br>**AI Therapy** | <img src="assets/usecase/ai_robot-0000.jpg" width="150" height="200"><br>**AI Robot** | <img src="assets/usecase/ai_creation-0000.jpg" width="150" height="200"><br>**AI Creation** | More...|

---

## 🤝 Contributing

We believe in the power of open-source collaboration.  Join us in shaping the future of AI memory!

*   **[Read our detailed Contributing Guide →](CONTRIBUTING.md)**

### **📄 License**

Your contributions are licensed under the **Apache License 2.0**.

---

## 🌍 Community

*   **GitHub Issues:**  Report bugs, request features. [Submit an issue](https://github.com/NevaMind-AI/memU/issues)
*   **Discord:** Get real-time support and connect with the community. [Join us](https://discord.com/invite/hQZntfGsbJ)
*   **X (Twitter):** Stay updated on the latest news. [Follow us](https://x.com/memU_ai)

---

## 🤝 Ecosystem

A proud partner of these organizations:

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

*   *Interested in partnering with MemU?*  Contact us at [contact@nevamind.ai](mailto:contact@nevamind.ai)

---

## 📱 Join Our WeChat Community

<div align="center">
<img src="assets/qrcode.png" alt="MemU WeChat and discord QR Code" width="480" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px;">
</div>

---

## Questionnaire

Help us improve! Share your feedback on our 3-min survey and get 30 free quota：https://forms.gle/H2ZuZVHv72xbqjvd7