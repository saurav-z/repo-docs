<div align="center">
  <img src="assets/banner.png" alt="MemU Banner" />
</div>

# MemU: The AI Companion Memory Framework for Unforgettable Interactions

**MemU empowers you to build AI companions that remember, learn, and grow with every interaction, offering superior accuracy, speed, and cost-effectiveness.**  Explore the original repository at: [https://github.com/NevaMind-AI/memU](https://github.com/NevaMind-AI/memU).

[![PyPI version](https://badge.fury.io/py/memu-py.svg)](https://badge.fury.io/py/memu-py)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/memu)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?logo=x&logoColor=white)](https://x.com/memU_ai)

---

## Key Features

*   ✅ **AI Companion Specialization:** Designed specifically for AI companion applications, ensuring optimal performance.
*   ✅ **92% Accuracy:** Achieve state-of-the-art accuracy in memory retrieval based on Locomo benchmark.
*   ✅ **Up to 90% Cost Reduction:** Optimize costs through an efficient, online platform.
*   ✅ **Advanced Retrieval Strategies:** Employ multiple methods including semantic search, hybrid search, and contextual retrieval.
*   ✅ **Advanced Memory Management:** Autonomous Memory File Management, Interconnected Knowledge Graph, Continuous Self-Improvement, and Adaptive Forgetting Mechanism
*   ✅ **24/7 Support:** Enterprise customers enjoy dedicated support.

---

## Why Choose MemU?

MemU revolutionizes how AI companions remember and learn. Here's why it stands out:

*   **Superior Accuracy:** Achieve a 92% average accuracy in Locomo dataset.
*   **Fast Retrieval:** Quickly access relevant information, eliminating the need for extensive embedding searches.
*   **Cost-Effective:** Process hundreds of conversation turns at once, significantly reducing token usage.

---

## Getting Started

MemU offers flexible options for integration:

### ☁️ **Cloud Version ([Online Platform](https://app.memu.so))**

Get up and running quickly with our cloud platform. Perfect for teams and individuals seeking immediate access.

*   **Instant Access:** Integrate AI memories within minutes.
*   **Managed Infrastructure:** We handle scaling, updates, and maintenance.
*   **Premium Support:** Priority assistance for subscribers.

**Steps:**

1.  **Create an Account:** Sign up at [https://app.memu.so](https://app.memu.so).
2.  **Generate API Keys:** Get your API keys at [https://app.memu.so/api-key/](https://app.memu.so/api-key/).
3.  **Integrate with Code:** Install the Python package and incorporate MemU into your project:

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
    conversation=conversation_text, # Recommend longer conversation (~8000 tokens), see https://memu.pro/blog/memu-best-practice for details
    user_id="user001", 
    user_name="User", 
    agent_id="assistant001", 
    agent_name="Assistant"
)
```

**Complete integration details are available in [`example/client/memory.py`](example/client/memory.py)** and the [API reference](docs/API_REFERENCE.md) or [our blog](https://memu.pro/blog).

### 🏢 **Enterprise Edition**

For organizations needing control, security, and customization:

*   **Commercial License:** Access to proprietary features and commercial usage rights.
*   **Custom Development:** SSO/RBAC, dedicated algorithm team.
*   **Intelligence & Analytics:** User behavior analysis, real-time monitoring.
*   **Premium Support:** 24/7 support, custom SLAs.

**Contact:** [contact@nevamind.ai](mailto:contact@nevamind.ai)

### 🏠 **Self-Hosting (Community Edition)**

For developers who prioritize local control and data privacy.

*   **Data Privacy:** Keep data within your infrastructure.
*   **Customization:** Modify and extend the platform.
*   **Cost Control:** Avoid cloud fees.

See [self hosting README](README.self_host.md)

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

## Memory as a File System

MemU offers advanced memory management features:

*   **Organize:** Autonomous Memory File Management: The memory agent automatically decides what to record, modify, or archive. Think of it as having a personal librarian who knows exactly how to organize your thoughts.
*   **Link:** Interconnected Knowledge Graph: Our system automatically creates meaningful connections between related memories, building a rich network of hyperlinked documents and transforming memory discovery from search into effortless recall.
*   **Evolve:** Continuous Self-Improvement: Generates new insights by analyzing existing memories, identifies patterns, and creates summary documents through self-reflection.
*   **Never Forget:** Adaptive Forgetting Mechanism: Automatically prioritizes information based on usage patterns, evolving with your needs.

---

## 😺 Advantages

*   **Higher Memory Accuracy:**  Achieves a leading 92.09% average accuracy on the Locomo dataset.  (Technical Report coming soon!)
    ![Memory Accuracy Comparison](assets/benchmark.png)
    *   (1) Single-hop questions; (2) Multi-hop questions; (3) Temporal reasoning; (4) Open-domain knowledge questions.
*   **Fast Retrieval:**  Categorizes information for quick access to relevant content.
*   **Low Cost:** Processes multiple turns efficiently to save on token usage.

---

## 🎓 **Use Cases**

| | | | |
|:---:|:---:|:---:|:---:|
| <img src="assets/usecase/ai_companion-0000.jpg" width="150" height="200"><br>**AI Companion** | <img src="assets/usecase/ai_role_play-0000.jpg" width="150" height="200"><br>**AI Role Play** | <img src="assets/usecase/ai_ip-0000.png" width="150" height="200"><br>**AI IP Characters** | <img src="assets/usecase/ai_edu-0000.jpg" width="150" height="200"><br>**AI Education** |
| <img src="assets/usecase/ai_therapy-0000.jpg" width="150" height="200"><br>**AI Therapy** | <img src="assets/usecase/ai_robot-0000.jpg" width="150" height="200"><br>**AI Robot** | <img src="assets/usecase/ai_creation-0000.jpg" width="150" height="200"><br>**AI Creation** | More...|

---

## 🤝 Contributing

We welcome your contributions!

*   **Contribute:** Explore our GitHub issues and projects.
*   **Guide:** [Read our detailed Contributing Guide →](CONTRIBUTING.md)

### **📄 License**

Contributions are licensed under the **Apache License 2.0**.

---

## 🌍 Community

*   **GitHub Issues:** Report bugs, request features: [Submit an issue](https://github.com/NevaMind-AI/memU/issues)
*   **Discord:** Get support and connect: [Join us](https://discord.com/invite/hQZntfGsbJ)
*   **X (Twitter):** Stay updated: [Follow us](https://x.com/memU_ai)

---

## 🤝 Ecosystem

We partner with amazing organizations:

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

*Interested in partnering? Contact us at [contact@nevamind.ai](mailto:contact@nevamind.ai)*

---

## 📱 Join Our WeChat Community

<div align="center">
  <img src="assets/qrcode.png" alt="MemU WeChat and discord QR Code" width="480" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px;">
  <br>
  *Scan the QR code to join our WeChat community.*
</div>

---

## Questionnaire

Help us improve MemU - share your feedback and get a free quota: https://forms.gle/H2ZuZVHv72xbqjvd7
```
Key improvements and SEO considerations:

*   **Clear Hook:**  The opening sentence is a strong and engaging hook, highlighting the core value proposition.
*   **Keyword Optimization:** Used relevant keywords like "AI companion," "memory framework," "AI," "retrieval," "accuracy," "cost," etc.
*   **Structured Headings:** Organized content with clear headings (H2, H3) for readability and SEO benefit.
*   **Bulleted Lists:** Used bullet points to highlight key features and benefits, making information easy to scan.
*   **Concise Language:**  Simplified and streamlined the text for better comprehension.
*   **Call to Actions:** Incorporated calls to action like "Get Started," "Join Us," and direct links.
*   **Partner Logos and Links:**  Showcasing the ecosystem and providing relevant links improves SEO.
*   **Mobile-Friendly:** Maintained good formatting for readability on different devices.
*   **Alt Text for Images:** Added descriptive alt text to images for accessibility and SEO.
*   **Focus on Value Proposition:** Emphasized the benefits (accuracy, speed, cost) throughout.
*   **Removed Redundancy:** Streamlined text while retaining important information.
*   **Clear Integration Steps:**  Made the "Getting Started" section more user-friendly with clear steps and code examples.
*   **Use Case Examples:** Expanded use case section for relevance.
*   **Contact Information:** Added contact details for potential users.
*   **Strong Community Emphasis:** Highlighted community involvement and support.