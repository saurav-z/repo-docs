<div align="center">

![MemU Banner](assets/banner.png)

</div>

# MemU: The Premier Open-Source Memory Framework for AI Companions

**Create AI companions that truly remember and learn with MemU, a high-accuracy, fast, and cost-effective open-source memory framework.  Learn more at [https://github.com/NevaMind-AI/memU](https://github.com/NevaMind-AI/memU).**

[![PyPI version](https://badge.fury.io/py/memu-py.svg)](https://badge.fury.io/py/memu-py)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/memu)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?logo=x&logoColor=white)](https://x.com/memU_ai)

---

## Key Features

*   **AI Companion Specialization**: Designed specifically for AI companion applications.
*   **High Accuracy**: Achieves state-of-the-art 92% accuracy on the Locomo benchmark.
*   **Cost-Effective**: Offers up to 90% cost reduction through optimized online platform.
*   **Advanced Retrieval Strategies**: Employs multiple methods including semantic search, hybrid search, and contextual retrieval.
*   **24/7 Enterprise Support**: Dedicated support for enterprise customers.

---

## Why Choose MemU?

MemU is engineered to be the ultimate memory solution for AI companions, offering a powerful combination of accuracy, speed, and cost efficiency. Built with advanced retrieval strategies and designed specifically for AI companion applications, MemU enables developers to build AI companions that truly remember and learn from their interactions. MemU adapts to different AI companion scenarios, making it an ideal "memory folder" for your AI.

---

## Getting Started

MemU offers flexible integration options to suit your needs, whether you're a solo developer or part of a large organization.

### ☁️ **Cloud Version (Online Platform)**

The easiest and fastest way to integrate.

*   **Instant Access**: Start integrating AI memories in minutes.
*   **Managed Infrastructure**: We handle scaling, updates, and maintenance.
*   **Premium Support**: Priority assistance from our engineering team.

**Quick Start:**

1.  **Create an Account**: Sign up at [https://app.memu.so](https://app.memu.so).
2.  **Generate API Keys**: Get your API keys at [https://app.memu.so/api-key/](https://app.memu.so/api-key/).
3.  **Install the Package**: `pip install memu-py`
4.  **Initialize & Memorize**:

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

For organizations requiring maximum security, customization, control and best quality:

*   **Commercial License**: Full proprietary features, commercial usage rights, white-labeling options
*   **Custom Development**: SSO/RBAC integration, dedicated algorithm team for scenario-specific framework optimization
*   **Intelligence & Analytics**: User behavior analysis, real-time production monitoring, automated agent optimization
*   **Premium Support**: 24/7 dedicated support, custom SLAs, professional implementation services

**Enterprise Inquiries:** [contact@nevamind.ai](mailto:contact@nevamind.ai)

### 🏠 Self-Hosting (Community Edition)

For users and developers who prefer local control, data privacy, or customization:

*   **Data Privacy**: Keep sensitive data within your infrastructure.
*   **Customization**: Modify and extend the platform to fit your needs.
*   **Cost Control**: Avoid recurring cloud fees for large-scale deployments.

See [self hosting README](README.self_host.md)

---

## ✨ Key Features Deep Dive

### 🎥 Demo Video

<div align="left">
  <a href="https://www.youtube.com/watch?v=qZIuCoLglHs">
    <img src="https://img.youtube.com/vi/ueOe4ZPlZLU/maxresdefault.jpg" alt="MemU Demo Video" width="600">
  </a>
  <br>
  <em>Click to watch the MemU demonstration video</em>
</div>

---

### Memory as a File System

*   **Organize: Autonomous Memory File Management**:  Your memories are structured as intelligent folders managed by a memory agent.
*   **Link: Interconnected Knowledge Graph**:  Our system automatically creates meaningful connections between related memories.
*   **Evolve: Continuous Self-Improvement**:  Your memory agent generates new insights by analyzing existing memories.
*   **Never Forget: Adaptive Forgetting Mechanism**: Prioritizes information based on usage patterns, creating a personalized information hierarchy.

---

## 😺 Advantages of Using MemU

*   **Higher Memory Accuracy**:  Achieves 92.09% average accuracy in Locomo dataset, significantly outperforming competitors.
*   **Fast Retrieval**: Categorizes information into documents for efficient content retrieval.
*   **Low Cost**: Processes hundreds of conversation turns at once, eliminating the need for repeated memory function calls and saving on token usage.

![Memory Accuracy Comparison](assets/benchmark.png)
<em>(1) Single-hop questions require answers based on a single session; (2) Multi-hop questions require synthesizing information from multiple different sessions; (3) Temporal reasoning questions can be answered through temporal reasoning and capturing time-related data cues within the conversation; (4) Open-domain knowledge questions can be answered by integrating a speaker’s provided information with external knowledge such as commonsense or world facts; </em>

---

## 🎓 Use Cases

| | | | |
|:---:|:---:|:---:|:---:|
| <img src="assets/usecase/ai_companion-0000.jpg" width="150" height="200"><br>**AI Companion** | <img src="assets/usecase/ai_role_play-0000.jpg" width="150" height="200"><br>**AI Role Play** | <img src="assets/usecase/ai_ip-0000.png" width="150" height="200"><br>**AI IP Characters** | <img src="assets/usecase/ai_edu-0000.jpg" width="150" height="200"><br>**AI Education** |
| <img src="assets/usecase/ai_therapy-0000.jpg" width="150" height="200"><br>**AI Therapy** | <img src="assets/usecase/ai_robot-0000.jpg" width="150" height="200"><br>**AI Robot** | <img src="assets/usecase/ai_creation-0000.jpg" width="150" height="200"><br>**AI Creation** | More...|

---

## 🤝 Contribute to MemU

We welcome contributions from the community!  Help us build the future of AI companion memory.

📋  **[Read our detailed Contributing Guide →](CONTRIBUTING.md)**

### **📄 License**

By contributing to MemU, you agree that your contributions will be licensed under the **Apache License 2.0**.

---

## 🌍 Community & Support

*   **GitHub Issues:** Report bugs, request features, and track development. [Submit an issue](https://github.com/NevaMind-AI/memU/issues)
*   **Discord:** Get real-time support, chat with the community, and stay updated. [Join us](https://discord.com/invite/hQZntfGsbJ)
*   **X (Twitter):** Follow for updates, AI insights, and key announcements. [Follow us](https://x.com/memU_ai)
*   **General inquiries:**  info@nevamind.ai

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

## 📱 Stay Connected

### **Join Our WeChat Community**

<div align="center">
<img src="assets/qrcode.png" alt="MemU WeChat and discord QR Code" width="480" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px;">
</div>

---

## Questionnaire

Help us improve! Share your feedback on our 3-min survey and get 30 free quota：https://forms.gle/H2ZuZVHv72xbqjvd7
```

Key improvements and SEO optimizations:

*   **Concise Hook:**  The initial sentence clearly defines MemU's purpose and benefits.
*   **Keyword-Rich Headings:** Uses headings like "Key Features", "Why Choose MemU?", "Getting Started", and "Advantages" to improve SEO.
*   **Bulleted Lists:**  Key features and benefits are clearly presented with bullet points.
*   **Focus on Benefits:** Highlights the value proposition for users (accuracy, cost savings, speed).
*   **Clear Calls to Action:**  Encourages users to get started and join the community.
*   **Relevant Keywords:** Includes keywords like "AI companion," "memory framework," "open source," "AI," and terms related to the project's functionality.
*   **Optimized Structure:**  The content is organized logically, making it easy to read and understand.
*   **Internal Links:** Added a link back to the original repository
*   **Removed redundant info:** consolidated and shortened descriptions where possible.
*   **Improved Formatting**:  Used bolding and spacing to improve readability