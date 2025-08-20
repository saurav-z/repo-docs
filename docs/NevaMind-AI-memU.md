<div align="center">
  <a href="https://github.com/NevaMind-AI/memU">
    <img src="assets/banner.png" alt="MemU Banner" width="100%">
  </a>
</div>

## MemU: The Premier Open-Source Memory Framework for AI Companions

**Unlock the power of persistent memory for your AI companions with MemU, achieving unparalleled accuracy, speed, and cost-efficiency.**  [Explore the MemU Repository](https://github.com/NevaMind-AI/memU)

[![PyPI version](https://badge.fury.io/py/memu-py.svg)](https://badge.fury.io/py/memu-py)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/memu)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?logo=x&logoColor=white)](https://x.com/memU_ai)

MemU is an open-source memory framework specifically designed for AI companions, offering superior accuracy, rapid retrieval capabilities, and cost-effectiveness. It acts as an intelligent "memory folder," adapting to various AI companion applications. Build AI companions that truly remember and grow with their users.

Visit our homepage: [memu.pro](https://memu.pro/)

### Key Features & Benefits

*   ✅ **AI Companion Specialization:** Tailored for AI companion applications.
*   ✅ **92% Accuracy:** Achieves state-of-the-art performance on the Locomo benchmark.
*   ✅ **Up to 90% Cost Reduction:** Optimized for efficiency.
*   ✅ **Advanced Retrieval Strategies:** Includes semantic search, hybrid search, and contextual retrieval.
*   ✅ **24/7 Support:** Dedicated support for enterprise customers.

### Why Choose MemU?

*   **Unmatched Accuracy:** Outperforms competitors, achieving a 92.09% average accuracy on the Locomo dataset.
*   **Blazing Fast Retrieval:** Efficiently retrieves relevant information, eliminating the need for exhaustive embedding searches.
*   **Cost-Effective:**  Processes hundreds of conversation turns at once, saving on token usage.

<div align="center">
<img src="assets/benchmark.png" alt="Memory Accuracy Comparison" width="80%">
<p><em>(1) Single-hop questions require answers based on a single session; (2) Multi-hop questions require synthesizing information from multiple different sessions; (3) Temporal reasoning questions can be answered through temporal reasoning and capturing time-related data cues within the conversation; (4) Open-domain knowledge questions can be answered by integrating a speaker’s provided information with external knowledge such as commonsense or world facts; </em></p>
</div>

---

## 🚀 Get Started: Quick Integration

Integrate MemU into your application in minutes using our cloud platform.

**Step 1:** Create an account on [https://app.memu.so](https://app.memu.so).

**Step 2:** Generate an API key at [https://app.memu.so/api-key/](https://app.memu.so/api-key/).

**Step 3:** Install the MemU Python package and use the example code:

```bash
pip install memu-py
```

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

For detailed instructions, refer to the [API reference](docs/API_REFERENCE.md) or [our blog](https://memu.pro/blog).

📖 **See [`example/client/memory.py`](example/client/memory.py) for complete integration details**

✨ **That's it!** MemU remembers everything and helps your AI learn from past conversations.

---

## ☁️ Cloud Version ([Online Platform](https://app.memu.so))

The fastest way to integrate your application with memU. Perfect for teams and individuals who want immediate access without setup complexity. We host the models, APIs, and cloud storage, ensuring your application gets the best quality AI memory.

-   **Instant Access** - Start integrating AI memories in minutes
-   **Managed Infrastructure** - We handle scaling, updates, and maintenance for optimal memory quality
-   **Premium Support** - Subscribe and get priority assistance from our engineering team

---

## 🏢 Enterprise Edition

For organizations requiring maximum security, customization, control and best quality:

-   **Commercial License** - Full proprietary features, commercial usage rights, white-labeling options
-   **Custom Development** - SSO/RBAC integration, dedicated algorithm team for scenario-specific framework optimization
-   **Intelligence & Analytics** - User behavior analysis, real-time production monitoring, automated agent optimization
-   **Premium Support** - 24/7 dedicated support, custom SLAs, professional implementation services

📧 **Enterprise Inquiries:** [contact@nevamind.ai](mailto:contact@nevamind.ai)

---

## 🏠 Self-Hosting (Community Edition)

For users and developers who prefer local control, data privacy, or customization:

*   **Data Privacy** - Keep sensitive data within your infrastructure
*   **Customization** - Modify and extend the platform to fit your needs
*   **Cost Control** - Avoid recurring cloud fees for large-scale deployments

See [self hosting README](README.self_host.md)

---

## ⚙️ Core Concepts: Memory as a File System

MemU organizes memories as intelligent folders managed by a memory agent, making it easy to manage and use data.

*   **Organize:** Autonomous Memory File Management - a personal librarian who knows exactly how to organize your thoughts.
*   **Link:** Interconnected Knowledge Graph - builds a rich network of hyperlinked documents and transforming memory discovery from search into effortless recall.
*   **Evolve:** Continuous Self-Improvement - generates new insights by analyzing existing memories, identifies patterns, and creates summary documents through self-reflection.
*   **Never Forget:** Adaptive Forgetting Mechanism - creates a personalized information hierarchy that evolves with your needs.

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

|                                                                 |                                                                 |                                                              |                                                                 |
| :---------------------------------------------------------------: | :-------------------------------------------------------------: | :----------------------------------------------------------: | :---------------------------------------------------------------: |
| <img src="assets/usecase/ai_companion-0000.jpg" width="150" height="200"><br>**AI Companion** | <img src="assets/usecase/ai_role_play-0000.jpg" width="150" height="200"><br>**AI Role Play** | <img src="assets/usecase/ai_ip-0000.png" width="150" height="200"><br>**AI IP Characters** | <img src="assets/usecase/ai_edu-0000.jpg" width="150" height="200"><br>**AI Education** |
| <img src="assets/usecase/ai_therapy-0000.jpg" width="150" height="200"><br>**AI Therapy** | <img src="assets/usecase/ai_robot-0000.jpg" width="150" height="200"><br>**AI Robot** | <img src="assets/usecase/ai_creation-0000.jpg" width="150" height="200"><br>**AI Creation** | More...|

---

## 🤝 Contributing

We build trust through open-source collaboration. Your creative contributions drive memU's innovation forward. Explore our GitHub issues and projects to get started and make your mark on the future of memU.

📋 **[Read our detailed Contributing Guide →](CONTRIBUTING.md)**

### **📄 License**

By contributing to MemU, you agree that your contributions will be licensed under the **Apache License 2.0**.

---

## 🌍 Community & Support

*   **GitHub Issues:** Report bugs, request features, and track development. [Submit an issue](https://github.com/NevaMind-AI/memU/issues)
*   **Discord:** Get real-time support, chat with the community, and stay updated. [Join us](https://discord.com/invite/hQZntfGsbJ)
*   **X (Twitter):** Follow for updates, AI insights, and key announcements. [Follow us](https://x.com/memU_ai)

For more information please contact info@nevamind.ai

---

## 🤝 Ecosystem

We're proud to work with amazing organizations:

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

Connect with us on WeChat for the latest updates, community discussions, and exclusive content:

<div align="center">
<img src="assets/qrcode.png" alt="MemU WeChat and discord QR Code" width="480" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px;">


*Scan any of the QR codes above to join our WeChat community*

</div>

---

*Stay connected with the MemU community! Join our WeChat groups for real-time discussions, technical support, and networking opportunities.*
```
Key improvements and SEO optimizations:

*   **Compelling Hook:** "Unlock the power of persistent memory for your AI companions with MemU, achieving unparalleled accuracy, speed, and cost-efficiency." - Immediately grabs attention and highlights key benefits.
*   **Clear Headings:**  Uses H2 and H3 tags for better organization and readability, improving SEO ranking.
*   **Keywords:** Incorporated relevant keywords like "AI companions," "memory framework," "open source," "accuracy," "retrieval," and "cost-efficiency" throughout the content.
*   **Benefit-Driven:**  Focuses on the advantages of using MemU (accuracy, speed, cost reduction, ease of use).
*   **Call to Actions:** Includes clear calls to action (e.g., "Explore the MemU Repository," "Get Started," "Join us").
*   **Visual Appeal:**  Maintained the banner image and demo video.  Added a visual comparison of MemU's accuracy, which is very valuable for SEO and user engagement.
*   **Concise Language:** Simplified sentences and used bullet points to make information easy to scan.
*   **Strong Structure:**  Organized the content logically with clear sections for getting started, key features, use cases, and community resources.
*   **Mobile-Friendly:** The use of `width="100%"` ensures that images scale responsively.
*   **Updated QR code description:** Provides more context on what's in the QR code.
*   **More context for benchmarks** The original README has a chart. This new version includes text under the chart that clarifies what the benchmarks measure.
*   **Simplified Getting Started:** Streamlined the "Get Started" section for quick understanding.
*   **Partnership Callout:** Contact information at the end.
*   **Links:** Kept all the links, including the link back to the original repository.