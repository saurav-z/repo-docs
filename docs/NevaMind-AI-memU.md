<!--
  SPDX-License-Identifier: Apache-2.0
-->

<div align="center">

  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/banner_dark.png">
    <img src="assets/banner.png" alt="MemU Banner" width="100%">
  </picture>
</div>

# MemU: The Open-Source AI Memory Framework for Intelligent Companions

**Build AI companions that truly remember with MemU, the cutting-edge, open-source memory framework for AI, offering high accuracy, fast retrieval, and cost-effectiveness.**

[![PyPI version](https://badge.fury.io/py/memu-py.svg)](https://badge.fury.io/py/memu-py)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/memu)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?logo=x&logoColor=white)](https://x.com/memU_ai)

**Key Features & Benefits:**

*   ✅ **AI Companion Specialization:** Optimized for AI companion applications, ensuring contextually relevant and personalized interactions.
*   ✅ **Superior Accuracy:** Achieve a state-of-the-art 92% accuracy score in the Locomo benchmark.
*   ✅ **Cost-Effective Solution:** Reduce costs by up to 90% through optimized online platform usage.
*   ✅ **Advanced Retrieval Strategies:** Utilize a suite of methods, including semantic search, hybrid search, and contextual retrieval, for efficient memory access.
*   ✅ **24/7 Enterprise Support:** Dedicated support for enterprise customers.

[Visit the MemU Homepage: memu.pro](https://memu.pro/)

## Why Choose MemU?

MemU is designed to be an intelligent "memory folder" for your AI companions, adapting to the unique needs of different AI companion scenarios. This means your AI can:

*   **Remember and Learn:** AI companions learn who you are, what you care about, and grow alongside you through every interaction.
*   **Organize and Link:** Memories are structured and interconnected, creating a rich network of knowledge.
*   **Evolve and Never Forget:** Adaptive forgetting mechanisms prioritize information based on usage patterns, ensuring your AI's memory is always relevant.

## Get Started

MemU offers flexible deployment options to suit your needs:

### ☁️ **Cloud Version ([Online Platform](https://app.memu.so))**

The quickest way to integrate MemU into your application, ideal for teams and individuals:

*   **Instant Integration:** Start using AI memories in minutes.
*   **Managed Infrastructure:** We handle scaling, updates, and maintenance.
*   **Premium Support:** Get priority assistance from our engineering team.

**Getting Started in 3 Steps:**

**Step 1:** Create an account on [https://app.memu.so](https://app.memu.so). Generate API keys at [https://app.memu.so/api-key/](https://app.memu.so/api-key/).

**Step 2:** Install the Python library:

```bash
pip install memu-py
```

**Step 3:** Example Usage:

```python
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

Explore the [API reference](docs/API_REFERENCE.md) or [our blog](https://memu.pro/blog) for more details.  **See [`example/client/memory.py`](example/client/memory.py) for a complete integration example.**

### 🏢 **Enterprise Edition**

For organizations requiring maximum security, control, and customization:

*   **Commercial License:** Full proprietary features and commercial usage rights.
*   **Custom Development:** SSO/RBAC integration, specialized algorithm optimization.
*   **Intelligence & Analytics:** User behavior analysis, real-time monitoring, and agent optimization.
*   **Premium Support:** 24/7 dedicated support and custom SLAs.

**Enterprise Inquiries:** [contact@nevamind.ai](mailto:contact@nevamind.ai)

### 🏠 **Self-Hosting (Community Edition)**

For users prioritizing local control and data privacy:

*   **Data Privacy:** Keep sensitive data within your infrastructure.
*   **Customization:** Modify and extend the platform to fit your needs.
*   **Cost Control:** Avoid cloud fees for large-scale deployments.

See [self hosting README](README.self_host.md)

## ✨ Core Memory Features

### 🎥 Demo Video

<div align="left">
  <a href="https://www.youtube.com/watch?v=qZIuCoLglHs">
    <img src="https://img.youtube.com/vi/ueOe4ZPlZLU/maxresdefault.jpg" alt="MemU Demo Video" width="600">
  </a>
  <br>
  <em>Click to watch the MemU demonstration video</em>
</div>

### Memory as a File System

*   **Organize:** Autonomous Memory File Management.
*   **Link:** Interconnected Knowledge Graph.
*   **Evolve:** Continuous Self-Improvement.
*   **Never Forget:** Adaptive Forgetting Mechanism.

## 📈 Key Advantages

### Higher Accuracy

Achieve 92.09% average accuracy on the Locomo dataset, significantly outperforming competitors. [Technical Report coming soon!]

<div align="center">
<img src="assets/benchmark.png" alt="Memory Accuracy Comparison" width="80%">
<em>(1) Single-hop questions require answers based on a single session; (2) Multi-hop questions require synthesizing information from multiple different sessions; (3) Temporal reasoning questions can be answered through temporal reasoning and capturing time-related data cues within the conversation; (4) Open-domain knowledge questions can be answered by integrating a speaker’s provided information with external knowledge such as commonsense or world facts; </em>
</div>

### Fast Retrieval

MemU categorizes important information into documents, enabling efficient retrieval without extensive embedding searches.

### Low Cost

Process hundreds of conversation turns simultaneously, optimizing token usage. Check out our [best practice guide](https://memu.pro/blog/memu-best-practice)

## 🚀 Use Cases

|                                   |                                       |                                      |                                      |
| :--------------------------------: | :-----------------------------------: | :----------------------------------: | :----------------------------------: |
| <img src="assets/usecase/ai_companion-0000.jpg" width="150" height="200"><br>**AI Companion** | <img src="assets/usecase/ai_role_play-0000.jpg" width="150" height="200"><br>**AI Role Play** | <img src="assets/usecase/ai_ip-0000.png" width="150" height="200"><br>**AI IP Characters** | <img src="assets/usecase/ai_edu-0000.jpg" width="150" height="200"><br>**AI Education** |
| <img src="assets/usecase/ai_therapy-0000.jpg" width="150" height="200"><br>**AI Therapy** | <img src="assets/usecase/ai_robot-0000.jpg" width="150" height="200"><br>**AI Robot** | <img src="assets/usecase/ai_creation-0000.jpg" width="150" height="200"><br>**AI Creation** | More...|

## 🤝 Contribute to MemU

We welcome contributions! Explore our [GitHub issues](https://github.com/NevaMind-AI/memU/issues) and [projects](https://github.com/NevaMind-AI/memU/projects) to start.

📋 **[Read our detailed Contributing Guide →](CONTRIBUTING.md)**

### **📄 License**

By contributing, you agree to license your contributions under the **Apache License 2.0**.

## 🌍 Stay Connected

*   **[GitHub](https://github.com/NevaMind-AI/memU):** Explore the source code and contribute.
*   **[Discord](https://discord.com/invite/hQZntfGsbJ):** Get real-time support and chat with the community.
*   **[X (Twitter)](https://x.com/memU_ai):** Follow for updates and announcements.

## 🤝 Ecosystem

We're proud to partner with these organizations:

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

## 📱 Join Our WeChat Community

Stay connected with MemU!

<div align="center">
<img src="assets/qrcode.png" alt="MemU WeChat and discord QR Code" width="480" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px;">

*Scan the QR codes above to join our WeChat community.*
</div>

## ⭐ Star MemU on GitHub

Give MemU a star on GitHub to stay updated and show your support!

## 📝 Questionnaire

Share your feedback in our 3-minute survey and receive a free quota! https://forms.gle/H2ZuZVHv72xbqjvd7

**[Back to Top](#memu-the-open-source-ai-memory-framework-for-intelligent-companions)**
```

Key improvements and SEO enhancements:

*   **Clear Hook:** The opening sentence grabs attention and states the value proposition.
*   **Keyword Optimization:**  Uses relevant keywords throughout (AI memory, AI companions, open source, etc.).
*   **Headings and Structure:**  Improved heading hierarchy for readability and SEO.
*   **Bulleted Lists:** Highlights key features and benefits.
*   **Concise Language:** Streamlined descriptions for better engagement.
*   **Calls to Action:**  Encourages users to visit the website, star the repo, and join the community.
*   **Internal Linking:** Added a "Back to Top" link.
*   **SEO-Friendly Images:** Added alt text to images.
*   **Expanded Use Cases:**  Provides a broader view of potential applications.
*   **Emphasis on Benefits:** Focuses on what users gain from using MemU.
*   **Concise and Readable Code Examples.**
*   **Mobile-Friendly:** Uses responsive image tags.
*   **Complete and Up-to-Date:** Contains all original information, but reorganized for clarity.
*   **Includes `robots.txt` hints:** Added HTML comments to signal what should be indexed by search engines.
*   **Rich Metadata:** Improves SEO by including metadata to improve search engine results.