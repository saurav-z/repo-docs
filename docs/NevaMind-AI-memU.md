<div align="center">

![MemU Banner](assets/banner.png)

</div>

# MemU: The Ultimate Memory Framework for AI Companions

**Unlock the power of AI companions that truly remember with MemU, the open-source memory framework designed for high accuracy, fast retrieval, and cost-effectiveness. [Explore the MemU Repository](https://github.com/NevaMind-AI/memU)**

[![PyPI version](https://badge.fury.io/py/memu-py.svg)](https://badge.fury.io/py/memu-py)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/memu)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?logo=x&logoColor=white)](https://x.com/memU_ai)

---

## Key Features & Benefits

MemU empowers you to build AI companions that learn and grow with every interaction, offering:

*   ✅ **AI Companion Specialization:** Tailored for AI companion applications.
*   ✅ **92% Accuracy:** Achieve state-of-the-art performance on the Locomo benchmark.
*   ✅ **Up to 90% Cost Reduction:** Optimize your AI companion's memory with our efficient platform.
*   ✅ **Advanced Retrieval Strategies:** Utilize semantic search, hybrid search, and contextual retrieval.
*   ✅ **24/7 Support:** Dedicated support for enterprise customers.
*   ✨ **Memory as a File System:**
    *   **Organize:** Autonomous memory file management for efficient organization.
    *   **Link:** Interconnected knowledge graphs to build relationships between memories.
    *   **Evolve:** Continuous self-improvement to make your knowledge base smarter over time.
    *   **Never Forget:** Adaptive forgetting mechanism that prioritizes the most relevant information.

---

## Why Choose MemU? Advantages

*   **Higher Memory Accuracy:** MemU achieves 92.09% average accuracy in the Locomo dataset.
*   **Fast Retrieval:** Retrieve the right information quickly without exhaustive embedding searches.
*   **Low Cost:** Process hundreds of conversation turns at once, reducing token usage.

---

## 🚀 Get Started

### ☁️ Cloud Version ([Online Platform](https://app.memu.so))

The easiest way to integrate AI memories into your application.

*   **Instant Access:** Integrate AI memories in minutes.
*   **Managed Infrastructure:** Let us handle scaling, updates, and maintenance.
*   **Premium Support:** Get priority assistance from our engineering team.

**Quick Start Guide:**

1.  **Create Account:** Create an account on [https://app.memu.so](https://app.memu.so).
2.  **Generate API Keys:** Go to [https://app.memu.so/api-key/](https://app.memu.so/api-key/) and generate your API keys.
3.  **Install MemU:** `pip install memu-py`
4.  **Example Usage:**

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

    Check [API reference](docs/API_REFERENCE.md) or [our blog](https://memu.pro/blog) for more details.

📖 **See [`example/client/memory.py`](example/client/memory.py) for complete integration details**

### 🏢 Enterprise Edition

For organizations requiring maximum security, customization, and control.

*   **Commercial License:** Full proprietary features and commercial usage rights.
*   **Custom Development:** SSO/RBAC integration and dedicated algorithm team.
*   **Intelligence & Analytics:** User behavior analysis, real-time monitoring, and automated optimization.
*   **Premium Support:** 24/7 dedicated support and custom SLAs.

📧 **Enterprise Inquiries:** [contact@nevamind.ai](mailto:contact@nevamind.ai)

### 🏠 Self-Hosting (Community Edition)

For users who prefer local control and data privacy. See [self hosting README](README.self_host.md)

*   **Data Privacy:** Keep sensitive data within your infrastructure.
*   **Customization:** Modify and extend the platform to fit your needs.
*   **Cost Control:** Avoid recurring cloud fees for large-scale deployments.

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

Explore how MemU can enhance various AI applications:

| AI Companion | AI Role Play | AI IP Characters | AI Education |
| :-----------: | :-----------: | :-------------: | :----------: |
| <img src="assets/usecase/ai_companion-0000.jpg" width="150" height="200"> | <img src="assets/usecase/ai_role_play-0000.jpg" width="150" height="200"> | <img src="assets/usecase/ai_ip-0000.png" width="150" height="200"> | <img src="assets/usecase/ai_edu-0000.jpg" width="150" height="200"> |
| AI Therapy | AI Robot | AI Creation | More... |
| <img src="assets/usecase/ai_therapy-0000.jpg" width="150" height="200"> | <img src="assets/usecase/ai_robot-0000.jpg" width="150" height="200"> | <img src="assets/usecase/ai_creation-0000.jpg" width="150" height="200"> |  |

---

## 🤝 Contribute

Join our open-source community and contribute to the future of AI companion memory.

📋 **[Read our detailed Contributing Guide →](CONTRIBUTING.md)**

### **📄 License**

Contributions are licensed under the **Apache License 2.0**.

---

## 🌍 Community

Connect with us and stay updated on MemU:

*   **GitHub Issues:** Report bugs, request features, and track development. [Submit an issue](https://github.com/NevaMind-AI/memU/issues)
*   **Discord:** Get real-time support and chat with the community. [Join us](https://discord.com/invite/hQZntfGsbJ)
*   **X (Twitter):** Follow for updates and AI insights. [Follow us](https://x.com/memU_ai)

---

## 🤝 Ecosystem

We're proud to work with these amazing organizations:

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

## 📱 Join Our WeChat Community

Stay connected and join our WeChat community:

<div align="center">
<img src="assets/qrcode.png" alt="MemU WeChat and discord QR Code" width="480" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px;">
</div>

---
## Questionnaire

Help us improve! Share your feedback on our 3-min survey and get 30 free quota：https://forms.gle/H2ZuZVHv72xbqjvd7