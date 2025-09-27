<div align="center">

![MemU Banner](assets/banner.png)
</div>

# MemU: The Next-Generation Memory Framework for AI Companions

**Unlock the power of persistent memory for your AI companions with MemU, offering unparalleled accuracy and cost-effectiveness!**  ([View on GitHub](https://github.com/NevaMind-AI/memU))

[![PyPI version](https://badge.fury.io/py/memu-py.svg)](https://badge.fury.io/py/memu-py)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/memu)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?logo=x&logoColor=white)](https://x.com/memU_ai)

---

## Key Features of MemU:

*   🎯 **AI Companion Specialization:**  Designed specifically for building intelligent AI companions, offering superior performance in this domain.
*   ✅ **Unmatched Accuracy:** Achieve state-of-the-art performance with a 92% accuracy rate in the Locomo benchmark, ensuring reliable memory recall.
*   💰 **Cost-Effective Solutions:**  Reduce your operational costs by up to 90% through optimized online platform usage.
*   🧠 **Advanced Retrieval Strategies:** Benefit from multiple retrieval methods, including semantic search, hybrid search, and contextual retrieval, for comprehensive memory access.
*   🤝 **24/7 Enterprise Support:**  Get dedicated support for enterprise customers.

---

## Why Choose MemU?

MemU is an open-source memory framework revolutionizing how AI companions remember and learn.  Built for high accuracy, lightning-fast retrieval, and cost efficiency, MemU acts as an intelligent "memory folder" that adapts to various AI companion scenarios.

*   **Build AI companions that truly understand you:** They learn who you are, what you care about, and evolve with every interaction.
*   **Organize memories automatically:** MemU acts like a personal librarian, managing and linking memories for effortless recall.
*   **Continuous self-improvement:** Even offline, your memory agent analyzes and refines existing memories, making your AI smarter over time.
*   **Prioritized information:**  An adaptive forgetting mechanism prioritizes relevant information, ensuring easy access to what matters most.

---

## 🚀 Get Started with MemU

Choose the best way to integrate MemU into your project:

### ☁️ Cloud Version ([Online Platform](https://app.memu.so))

The easiest way to integrate your application with MemU.

*   **Instant Integration:** Get started in minutes.
*   **Managed Infrastructure:** We handle scaling, updates, and maintenance.
*   **Premium Support:** Priority assistance from our engineering team.

**Quick Start Guide:**

1.  **Create an Account:**  Sign up at [https://app.memu.so](https://app.memu.so)
2.  **Generate API Keys:**  Go to [https://app.memu.so/api-key/](https://app.memu.so/api-key/) to obtain your API keys.
3.  **Install the Package:** `pip install memu-py`
4.  **Use the Code:**

    ```python
    # Example usage
    import os
    from memu import MemuClient

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
    Check [API reference](docs/API_REFERENCE.md) or [our blog](https://memu.pro/blog) for more details. 
5.  **Complete Integration:**  See [`example/client/memory.py`](example/client/memory.py) for complete integration details.

### 🏢 Enterprise Edition

For organizations needing maximum security, control, and customization:

*   **Commercial License:** Full proprietary features.
*   **Custom Development:** SSO/RBAC integration and dedicated algorithm optimization.
*   **Intelligence & Analytics:** User behavior analysis and real-time monitoring.
*   **Premium Support:** 24/7 dedicated support and custom SLAs.

**Contact:** [contact@nevamind.ai](mailto:contact@nevamind.ai)

### 🏠 Self-Hosting (Community Edition)

For users preferring local control and data privacy:

*   **Data Privacy:** Keep sensitive data within your infrastructure.
*   **Customization:** Modify and extend the platform to fit your needs.
*   **Cost Control:** Avoid cloud fees for large deployments.

See [self hosting README](README.self_host.md)

---

## ✨ Memory as File System: Key Benefits

*   **Organize:** Autonomous memory file management with an intelligent memory agent.
*   **Link:** Interconnected knowledge graph building relationships between memories.
*   **Evolve:** Continuous self-improvement and insight generation from existing memories.
*   **Never Forget:** Adaptive forgetting mechanism prioritizing relevant information.

---

## 🎥  See MemU in Action: Demo Video

<div align="left">
  <a href="https://www.youtube.com/watch?v=qZIuCoLglHs">
    <img src="https://img.youtube.com/vi/ueOe4ZPlZLU/maxresdefault.jpg" alt="MemU Demo Video" width="600">
  </a>
  <br>
  <em>Click to watch the MemU demonstration video</em>
</div>

---

## 😺 Advantages of MemU

### Higher Memory Accuracy

MemU achieves an outstanding 92.09% average accuracy in the Locomo dataset across all reasoning tasks, greatly surpassing competitors. Technical Report will be published soon!

![Memory Accuracy Comparison](assets/benchmark.png)
<em>(1) Single-hop questions require answers based on a single session; (2) Multi-hop questions require synthesizing information from multiple different sessions; (3) Temporal reasoning questions can be answered through temporal reasoning and capturing time-related data cues within the conversation; (4) Open-domain knowledge questions can be answered by integrating a speaker’s provided information with external knowledge such as commonsense or world facts; </em>

### Fast Retrieval

Categorizing important information into documents ensures quick retrieval of relevant content without exhaustive embedding searches.

### Low Cost

MemU efficiently processes numerous conversation turns at once, minimizing API calls and token usage.

---

## 🎓 Use Cases

MemU is versatile and can be applied to various scenarios:

| | | | |
|:---:|:---:|:---:|:---:|
| <img src="assets/usecase/ai_companion-0000.jpg" width="150" height="200"><br>**AI Companion** | <img src="assets/usecase/ai_role_play-0000.jpg" width="150" height="200"><br>**AI Role Play** | <img src="assets/usecase/ai_ip-0000.png" width="150" height="200"><br>**AI IP Characters** | <img src="assets/usecase/ai_edu-0000.jpg" width="150" height="200"><br>**AI Education** |
| <img src="assets/usecase/ai_therapy-0000.jpg" width="150" height="200"><br>**AI Therapy** | <img src="assets/usecase/ai_robot-0000.jpg" width="150" height="200"><br>**AI Robot** | <img src="assets/usecase/ai_creation-0000.jpg" width="150" height="200"><br>**AI Creation** | More...|
---

## 🤝 Contribute to MemU

We welcome your contributions!  Help us build the future of AI memory.

*   📋 **[Read our detailed Contributing Guide →](CONTRIBUTING.md)**
*   **License:** Your contributions are licensed under the Apache License 2.0.

---

## 🌍 Community & Support

*   **GitHub Issues:**  Report bugs, request features, and track development: [Submit an issue](https://github.com/NevaMind-AI/memU/issues)
*   **Discord:** Get real-time support and connect with the community: [Join us](https://discord.com/invite/hQZntfGsbJ)
*   **X (Twitter):** Stay updated on the latest news: [Follow us](https://x.com/memU_ai)
*   **General Inquiries:** [info@nevamind.ai](mailto:info@nevamind.ai)

---

## 🤝 Ecosystem - Partners & Integrations

We're proud to collaborate with these exceptional organizations:

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

## 📱 Stay Connected: WeChat Community

Join our WeChat community for the latest updates and discussions:

<div align="center">
<img src="assets/qrcode.png" alt="MemU WeChat and discord QR Code" width="480" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px;">
</div>

---

## 📢  We Value Your Feedback!

Help us improve MemU:  Complete our 3-minute survey and get 30 free quota:  https://forms.gle/H2ZuZVHv72xbqjvd7