<!-- Improved README for MemU - Optimized for SEO -->

<div align="center">

![MemU Banner](assets/banner.png)

## MemU: Build AI Companions That Remember with Next-Gen Memory 🧠

**Unlock the power of persistent memory for your AI companions with MemU, the open-source memory framework designed for accuracy, speed, and cost-effectiveness.**  ([See the original repo](https://github.com/NevaMind-AI/memU))

[![PyPI version](https://badge.fury.io/py/memu-py.svg)](https://badge.fury.io/py/memu-py)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/memu)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?logo=x&logoColor=white)](https://x.com/memU_ai)

</div>

---

## ✨ Key Features: The MemU Advantage

MemU empowers you to create AI companions that understand and evolve with users, offering a superior memory solution.

*   ✅ **AI Companion Specialization:** Tailored specifically for AI companion applications.
*   ✅ **Unmatched Accuracy:** Achieve state-of-the-art accuracy, scoring 92% in the Locomo benchmark, ensuring reliable recall.
*   ✅ **Cost-Effective Solution:** Reduce operational costs by up to 90% through optimized online platform usage.
*   ✅ **Advanced Retrieval Strategies:** Leverage semantic search, hybrid search, and contextual retrieval for precise information access.
*   ✅ **24/7 Enterprise Support:** Dedicated support for enterprise customers to ensure optimal performance.

---

## 🚀 Get Started: Quick Integration Guide

MemU offers multiple ways to integrate:

### ☁️ **Cloud Version (Online Platform)** - The Fastest Way to Integrate

Get up and running with MemU quickly using our hosted solution. Perfect for teams and individuals.

*   **Instant Integration:**  Begin integrating AI memories within minutes.
*   **Managed Infrastructure:**  We handle all scaling, updates, and maintenance.
*   **Premium Support:** Benefit from priority assistance from our expert engineering team.

**Steps:**

1.  **Create an Account:**  Sign up at [https://app.memu.so](https://app.memu.so).
2.  **Generate API Keys:**  Obtain your API keys from [https://app.memu.so/api-key/](https://app.memu.so/api-key/).
3.  **Integrate 3 Lines of Code**: Use `pip install memu-py` and the following example.

```python
import os
from memu import MemuClient

# Initialize
memu_client = MemuClient(
    base_url="https://api.memu.so",
    api_key=os.getenv("MEMU_API_KEY")
)
memu_client.memorize_conversation(
    conversation=conversation_text,  # Recommend longer conversation (~8000 tokens)
    user_id="user001",
    user_name="User",
    agent_id="assistant001",
    agent_name="Assistant"
)
```

**For detailed implementation, see [`example/client/memory.py`](example/client/memory.py).**

**That's It!**  MemU stores everything and helps your AI learn from past conversations.
**See our blog:** [https://memu.pro/blog](https://memu.pro/blog)

### 🏢 Enterprise Edition

For organizations prioritizing security, customization, and control.

*   **Commercial Licensing:** Full proprietary features with commercial usage.
*   **Custom Development:** Integration with SSO/RBAC and dedicated algorithm optimization.
*   **Intelligence & Analytics:** Includes user behavior analysis and real-time production monitoring.
*   **Premium Support:** Provides 24/7 support, custom SLAs, and implementation services.

**For Enterprise inquiries, contact: [contact@nevamind.ai](mailto:contact@nevamind.ai)**

### 🏠 Self-Hosting (Community Edition)

For users who want local control, data privacy, or customization.

*   **Data Privacy:** Keeps sensitive data within your infrastructure.
*   **Customization:** Modify the platform as needed.
*   **Cost Control:** Avoids recurring cloud fees for large deployments.

**See the [self-hosting README](README.self_host.md)**

---

## 💡 Deep Dive: MemU's Core Concepts

### 🎥 Demo Video

<div align="left">
  <a href="https://www.youtube.com/watch?v=qZIuCoLglHs">
    <img src="https://img.youtube.com/vi/ueOe4ZPlZLU/maxresdefault.jpg" alt="MemU Demo Video" width="600">
  </a>
  <br>
  <em>Click to watch the MemU demonstration video</em>
</div>

### 🧠 Memory as a File System

MemU redefines memory management for AI.

*   **Organize:** Intelligent memory agents manage your memories as intelligent folders.
*   **Link:** Automatically creates connections between related memories for a rich knowledge graph.
*   **Evolve:** Self-improving, generating new insights and summaries.
*   **Never Forget:** Adaptive forgetting prioritizes and optimizes information based on usage patterns.

---

## ✅ Advantages & Benefits: Why Choose MemU?

*   **Higher Memory Accuracy:**  Achieves a 92.09% average accuracy in the Locomo dataset, surpassing competitors. (Technical report coming soon!)
  ![Memory Accuracy Comparison](assets/benchmark.png)
*   **Fast Retrieval:** Efficiently retrieves important information by categorizing and accessing the relevant content.
*   **Low Cost:** Processes hundreds of conversation turns at once to reduce token usage. See [best practice](https://memu.pro/blog/memu-best-practice).

---

## 🧩 Use Cases: Powering Diverse Applications

| AI Companion | AI Role Play | AI IP Characters | AI Education | AI Therapy | AI Robot | AI Creation | More...|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| <img src="assets/usecase/ai_companion-0000.jpg" width="150" height="200"> | <img src="assets/usecase/ai_role_play-0000.jpg" width="150" height="200"> | <img src="assets/usecase/ai_ip-0000.png" width="150" height="200"> | <img src="assets/usecase/ai_edu-0000.jpg" width="150" height="200"> | <img src="assets/usecase/ai_therapy-0000.jpg" width="150" height="200"> | <img src="assets/usecase/ai_robot-0000.jpg" width="150" height="200"> | <img src="assets/usecase/ai_creation-0000.jpg" width="150" height="200"> |  |

---

## 🤝 Contributing & Community

We welcome contributions from everyone. Join us in building the future of AI memory!

*   📋 **[Read our detailed Contributing Guide →](CONTRIBUTING.md)**
*   **License:** MemU is licensed under the **Apache License 2.0**.

---

## 🌍 Connect with the MemU Community

*   **GitHub Issues:** Report bugs and request features:  [Submit an issue](https://github.com/NevaMind-AI/memU/issues)
*   **Discord:** Get support, chat, and stay updated: [Join our Discord](https://discord.com/invite/hQZntfGsbJ)
*   **X (Twitter):** Follow us for the latest news: [Follow us](https://x.com/memU_ai)

---

## 🤝 Ecosystem & Partners

We're proud to collaborate with leading organizations in the AI field:

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

*Interested in a partnership? Contact us at [contact@nevamind.ai](mailto:contact@nevamind.ai)*

---

## 📱 Stay Connected

Join our WeChat community for updates and support:

<div align="center">
<img src="assets/qrcode.png" alt="MemU WeChat and discord QR Code" width="480" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px;">
</div>

---
## 📝 Feedback Survey

Help us improve! Complete our 3-min survey for a chance to receive 30 free quota: https://forms.gle/H2ZuZVHv72xbqjvd7
```

Key improvements and SEO optimizations:

*   **Keyword Integration:**  Used relevant keywords like "AI companions," "memory framework," "open-source," "AI memory," "persistent memory" throughout the README.
*   **Clear Headings:**  Organized the content with clear, descriptive headings (e.g., "Key Features," "Get Started," "Advantages").
*   **Concise Language:**  Used clear, concise language to improve readability.
*   **Bulleted Lists:**  Emphasized key features and benefits using bulleted lists for easy scanning.
*   **Strong Call to Actions:**  Encouraged users to contribute, join the community, and explore resources.
*   **SEO-Friendly Formatting:**  Used Markdown formatting (headings, bold text) to improve SEO.
*   **Targeted Use Cases:**  Highlighted the specific use cases that MemU is designed for.
*   **Homepage & Blog Links:** Included links to the homepage and blog to drive traffic and provide additional value.
*   **Alt Text:** Used descriptive `alt` text for images to help with SEO.
*   **Concise Hook:**  The one-sentence hook is prominently placed at the beginning to immediately grab the reader's attention and highlight the value proposition.
*   **Emphasis on Value:**  Focused on the *benefits* (accuracy, speed, cost reduction) that MemU provides.
*   **Clear Instructions:**  Simplified the "Get Started" section and made it easy to follow.
*   **Mobile-Friendly:**  The design is inherently mobile-friendly due to the use of markdown and responsive images.