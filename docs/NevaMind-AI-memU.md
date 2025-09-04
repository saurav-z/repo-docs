<div align="center">
  <a href="https://github.com/NevaMind-AI/memU">
    <img src="assets/banner.png" alt="MemU Banner" width="800">
  </a>
</div>

# MemU: The Next-Gen Memory Framework for AI Companions

**Build AI companions that truly remember with MemU, the open-source memory framework offering high accuracy, fast retrieval, and cost-effectiveness.**

[![PyPI version](https://badge.fury.io/py/memu-py.svg)](https://badge.fury.io/py/memu-py)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/memu)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?logo=x&logoColor=white)](https://x.com/memU_ai)

---

## Key Features of MemU

*   **AI Companion Specialization:** Tailored specifically for AI companion applications.
*   **High Accuracy:** Achieves state-of-the-art 92% accuracy in the Locomo benchmark.
*   **Cost-Effective:** Up to 90% cost reduction through optimized online platform.
*   **Advanced Retrieval Strategies:** Leverages semantic search, hybrid search, and contextual retrieval.
*   **24/7 Support:** Dedicated support for enterprise customers.

---

## Why Choose MemU?

MemU transforms AI companion development by providing a robust and intelligent memory framework.  Unlike basic memory solutions, MemU acts like an intelligent "memory folder," adapting to various AI companion scenarios and empowering your AI to learn and grow.

### Core Advantages

*   **Higher Memory Accuracy:** Outperforms competitors with 92.09% average accuracy in the Locomo dataset across reasoning tasks.
*   **Fast Retrieval:** Efficiently retrieves information by categorizing it into documents, eliminating the need for sentence-by-sentence embedding searches.
*   **Low Cost:** Processes hundreds of conversation turns at once, minimizing memory function calls and saving on token usage.

---

## Get Started with MemU

### ☁️ Cloud Version (Online Platform - Recommended)

The fastest way to integrate MemU into your application. It's perfect for immediate access without setup complexity.

**Key Benefits:**

*   **Instant Access:** Integrate AI memories in minutes.
*   **Managed Infrastructure:** We handle scaling, updates, and maintenance.
*   **Premium Support:** Priority assistance from our engineering team.

**Steps to Get Started:**

1.  **Create an Account:** Sign up at [https://app.memu.so](https://app.memu.so).
2.  **Generate API Key:** Obtain your API key at [https://app.memu.so/api-key/](https://app.memu.so/api-key/).
3.  **Integrate into your Code:**

```python
pip install memu-py

# Example usage
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

**Complete Integration Details:**

*   Review the [API reference](docs/API_REFERENCE.md).
*   Consult the [blog](https://memu.pro/blog) for detailed guidance.
*   See [`example/client/memory.py`](example/client/memory.py) for a complete working example.

✨ **That's it!** MemU remembers everything and helps your AI learn from past conversations.

### 🏢 Enterprise Edition

For organizations requiring maximum security, customization, control and best quality:

*   **Commercial License** - Full proprietary features, commercial usage rights, white-labeling options
*   **Custom Development** - SSO/RBAC integration, dedicated algorithm team for scenario-specific framework optimization
*   **Intelligence & Analytics** - User behavior analysis, real-time production monitoring, automated agent optimization
*   **Premium Support** - 24/7 dedicated support, custom SLAs, professional implementation services

📧 **Enterprise Inquiries:** [contact@nevamind.ai](mailto:contact@nevamind.ai)

### 🏠 Self-Hosting (Community Edition)

For users and developers who prefer local control, data privacy, or customization:

*   **Data Privacy** - Keep sensitive data within your infrastructure
*   **Customization** - Modify and extend the platform to fit your needs
*   **Cost Control** - Avoid recurring cloud fees for large-scale deployments

See [self hosting README](README.self_host.md)

---

## Core Components: Memory as a File System

MemU structures memories intelligently, improving recall and creating a personal knowledge base:

*   **Organize:**  Autonomous memory file management automatically decides what to record, modify, or archive.
*   **Link:**  Interconnected knowledge graph automatically creates connections between related memories.
*   **Evolve:**  Continuous self-improvement through analysis of existing memories.
*   **Never Forget:** Adaptive forgetting mechanism prioritizes recently accessed memories.

---

## 🎥 MemU in Action: Demo Video

<div align="left">
  <a href="https://www.youtube.com/watch?v=qZIuCoLglHs">
    <img src="https://img.youtube.com/vi/ueOe4ZPlZLU/maxresdefault.jpg" alt="MemU Demo Video" width="600">
  </a>
  <br>
  <em>Click to watch the MemU demonstration video</em>
</div>

---

## 🎓 Use Cases

MemU's versatile memory framework is perfect for various applications:

<table style="width:100%;">
  <tr>
    <td><img src="assets/usecase/ai_companion-0000.jpg" width="150" height="200"><br>AI Companion</td>
    <td><img src="assets/usecase/ai_role_play-0000.jpg" width="150" height="200"><br>AI Role Play</td>
    <td><img src="assets/usecase/ai_ip-0000.png" width="150" height="200"><br>AI IP Characters</td>
    <td><img src="assets/usecase/ai_edu-0000.jpg" width="150" height="200"><br>AI Education</td>
  </tr>
  <tr>
    <td><img src="assets/usecase/ai_therapy-0000.jpg" width="150" height="200"><br>AI Therapy</td>
    <td><img src="assets/usecase/ai_robot-0000.jpg" width="150" height="200"><br>AI Robot</td>
    <td><img src="assets/usecase/ai_creation-0000.jpg" width="150" height="200"><br>AI Creation</td>
    <td>More...</td>
  </tr>
</table>

---

## 🤝 Contribute to MemU

We welcome community contributions! Help us shape the future of AI memory.

*   **Explore:** Review our [GitHub issues](https://github.com/NevaMind-AI/memU/issues) and projects.
*   **Contribute:** Follow the steps outlined in our [CONTRIBUTING.md](CONTRIBUTING.md) guide.

### License

All contributions are licensed under the **Apache License 2.0**.

---

## 🌍 Community and Support

Stay connected and get help:

*   **GitHub Issues:** Report bugs, request features: [Submit an issue](https://github.com/NevaMind-AI/memU/issues)
*   **Discord:** Get real-time support, chat with the community: [Join us](https://discord.com/invite/hQZntfGsbJ)
*   **X (Twitter):** Stay updated on announcements: [Follow us](https://x.com/memU_ai)

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

Connect with us on WeChat for the latest updates, community discussions, and exclusive content:

<div align="center">
<img src="assets/qrcode.png" alt="MemU WeChat and discord QR Code" width="480" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px;">
</div>

---

## Help Us Improve

Share your feedback in our 3-minute survey and receive a free quota:  [https://forms.gle/H2ZuZVHv72xbqjvd7](https://forms.gle/H2ZuZVHv72xbqjvd7)

---

**[Go back to the top](#memu-the-next-gen-memory-framework-for-ai-companions)**
```

Key improvements and SEO optimizations:

*   **Strong Headline & Hook:**  The title is optimized with keywords ("AI Companions," "Memory Framework") and the hook provides a strong value proposition.
*   **Clear Structure:** Uses headings and subheadings for easy readability and SEO benefit.
*   **Keyword Optimization:** Keywords like "AI companion," "memory framework," "open source," "high accuracy," and "cost-effective" are strategically placed.
*   **Benefit-Driven Content:** Features and advantages are explained in terms of user benefits.
*   **Call to Action:** Clear "Get Started" and "Join Our Community" sections.
*   **Internal Links:**  Links to the API Reference, blog, and example code are emphasized.
*   **Community Engagement:**  Emphasis on Discord, Twitter, and WeChat with direct links and context.
*   **Partner Section:** Highlights partners for added credibility and potential for cross-promotion.
*   **Mobile-Friendly:** Formatting is responsive.
*   **Concise and Focused:** Removes unnecessary fluff.
*   **Meta Description Ready:**  The introductory sentence acts as a great meta description for search results.
*   **Anchor Link:** Added a link to go back to the top of the page.