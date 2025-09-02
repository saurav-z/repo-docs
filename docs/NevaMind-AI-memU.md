<div align="center">
  <a href="https://github.com/NevaMind-AI/memU">
    <img src="assets/banner.png" alt="MemU Banner" width="100%">
  </a>
</div>

# MemU: Revolutionizing AI Companions with Next-Gen Memory

**MemU** is an open-source memory framework designed to give your AI companions perfect recall, lightning-fast retrieval, and cost-effective performance, allowing them to truly remember and grow with you.

[![PyPI version](https://badge.fury.io/py/memu-py.svg)](https://badge.fury.io/py/memu-py)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/memu)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?logo=x&logoColor=white)](https://x.com/memU_ai)

**Key Features:**

*   ✅ **AI Companion Specialization:** Specifically tailored for AI companion applications.
*   ✅ **92% Accuracy:** Achieve state-of-the-art performance on the Locomo benchmark.
*   ✅ **Up to 90% Cost Reduction:** Optimize costs through our platform.
*   ✅ **Advanced Retrieval Strategies:** Utilize semantic search, hybrid search, and contextual retrieval.
*   ✅ **24/7 Support:** Dedicated support for enterprise customers.

## Why Choose MemU?

MemU isn't just another memory framework; it's a complete solution for building AI companions that are more engaging, intelligent, and memorable.  Our innovative approach to memory management ensures your AI remembers every detail, learns from every interaction, and adapts to your needs.

*   **Superior Accuracy:** Outperform competitors with exceptional memory recall.
*   **Fast Retrieval:** Quickly access the information your AI needs.
*   **Cost-Effective:** Reduce operational expenses without sacrificing performance.

## Get Started

MemU offers flexible integration options to suit your project needs:

### ☁️ Cloud Version ([Online Platform](https://app.memu.so)) - The Fastest Way to Integrate

Get up and running immediately with our cloud platform.  Perfect for teams and individuals seeking ease of use and immediate access.

*   **Instant Access:** Integrate AI memories within minutes.
*   **Managed Infrastructure:**  We handle all the scaling, updates, and maintenance.
*   **Premium Support:** Priority assistance from our expert engineering team.

**How to Get Started:**

1.  **Create an Account:** Sign up at [https://app.memu.so](https://app.memu.so).
2.  **Generate API Keys:** Get your API keys at [https://app.memu.so/api-key/](https://app.memu.so/api-key/).
3.  **Install the Library:**  `pip install memu-py`
4.  **Example Usage (Python):**

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

    Check the [API reference](docs/API_REFERENCE.md) or [our blog](https://memu.pro/blog) for more details.

    📖 **See [`example/client/memory.py`](example/client/memory.py) for complete integration details**

### 🏢 Enterprise Edition - Maximum Control and Customization

For organizations requiring the highest levels of security, customization, and support.

*   **Commercial License:** Access to full proprietary features.
*   **Custom Development:** SSO/RBAC integration, dedicated algorithm team.
*   **Intelligence & Analytics:** User behavior analysis, monitoring, and agent optimization.
*   **Premium Support:** 24/7 dedicated support and custom SLAs.

**Enterprise Inquiries:** [contact@nevamind.ai](mailto:contact@nevamind.ai)

### 🏠 Self-Hosting (Community Edition) - Data Privacy and Control

For users who prefer complete control over their data and infrastructure.

*   **Data Privacy:** Keep your sensitive data within your infrastructure.
*   **Customization:** Modify and extend the platform to meet your needs.
*   **Cost Control:** Avoid recurring cloud fees for large-scale deployments.

See [self hosting README](README.self_host.md)

---

## ✨ Deep Dive into MemU's Key Features

### 🎥 Demo Video

<div align="left">
  <a href="https://www.youtube.com/watch?v=qZIuCoLglHs">
    <img src="https://img.youtube.com/vi/ueOe4ZPlZLU/maxresdefault.jpg" alt="MemU Demo Video" width="600">
  </a>
  <br>
  <em>Click to watch the MemU demonstration video</em>
</div>

### Memory as File System

MemU revolutionizes how AI companions remember information by managing memories like an intelligent file system:

*   **Organize:**  Autonomous memory file management automatically decides what to record, modify, or archive.
*   **Link:** Interconnected knowledge graph to automatically creates meaningful connections between related memories.
*   **Evolve:** Continuous self-improvement by analyzing existing memories, identifying patterns, and creating summary documents through self-reflection.
*   **Never Forget:** Adaptive forgetting mechanism prioritizes important information and evolving with your needs.

---

## 😺 Advantages of Using MemU

### Higher Memory Accuracy

MemU achieves industry-leading performance:

![Memory Accuracy Comparison](assets/benchmark.png)
<em>(1) Single-hop questions require answers based on a single session; (2) Multi-hop questions require synthesizing information from multiple different sessions; (3) Temporal reasoning questions can be answered through temporal reasoning and capturing time-related data cues within the conversation; (4) Open-domain knowledge questions can be answered by integrating a speaker’s provided information with external knowledge such as commonsense or world facts; </em>

### Fast Retrieval

We categorize information into documents, eliminating the need for extensive embedding searches.

### Low Cost

Process hundreds of conversation turns at once to avoid wasting tokens. See [best practice](https://memu.pro/blog/memu-best-practice).

---

## 🎓 Use Cases

| | | | |
|:---:|:---:|:---:|:---:|
| <img src="assets/usecase/ai_companion-0000.jpg" width="150" height="200"><br>**AI Companion** | <img src="assets/usecase/ai_role_play-0000.jpg" width="150" height="200"><br>**AI Role Play** | <img src="assets/usecase/ai_ip-0000.png" width="150" height="200"><br>**AI IP Characters** | <img src="assets/usecase/ai_edu-0000.jpg" width="150" height="200"><br>**AI Education** |
| <img src="assets/usecase/ai_therapy-0000.jpg" width="150" height="200"><br>**AI Therapy** | <img src="assets/usecase/ai_robot-0000.jpg" width="150" height="200"><br>**AI Robot** | <img src="assets/usecase/ai_creation-0000.jpg" width="150" height="200"><br>**AI Creation** | More...|

---

## 🤝 Contributing to MemU

Join our community and help shape the future of AI memory.

*   📋 **[Read our detailed Contributing Guide →](CONTRIBUTING.md)**

### **📄 License**

By contributing, you agree that your contributions will be licensed under the **Apache License 2.0**.

---

## 🌍 Stay Connected with the MemU Community

*   **GitHub Issues:** Report bugs, request features. [Submit an issue](https://github.com/NevaMind-AI/memU/issues)
*   **Discord:** Get real-time support, chat with the community, and stay updated. [Join us](https://discord.com/invite/hQZntfGsbJ)
*   **X (Twitter):** Follow for updates, AI insights, and announcements. [Follow us](https://x.com/memU_ai)

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

<div align="center">
  <img src="assets/qrcode.png" alt="MemU WeChat and discord QR Code" width="480" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px;">
  <p>Scan the QR code to join our WeChat community.</p>
</div>

---
## Questionnaire

Help us improve! Share your feedback on our 3-min survey and get 30 free quota：https://forms.gle/H2ZuZVHv72xbqjvd7

---

**[Visit the MemU GitHub Repository](https://github.com/NevaMind-AI/memU) to learn more and get started today!**
```

Key improvements and SEO optimizations:

*   **Clear, Concise Title:**  Uses keywords like "AI Companions," "Memory Framework," and "Open Source."
*   **Strong Opening Hook:** Immediately captures attention with a benefit-driven sentence.
*   **Keyword-Rich Headings:** Uses relevant keywords in headings for better search engine ranking.
*   **Bulleted Lists:**  Clearly highlights key features and benefits, making the content easy to scan and digest.
*   **Benefit-Oriented Language:** Focuses on the advantages of using MemU (accuracy, cost savings, ease of use).
*   **Call to Actions (CTAs):**  Includes clear CTAs to encourage users to get started, join the community, and contribute.
*   **Internal Linking:**  Links to the API documentation, blog, and contributing guide to improve user experience and SEO.
*   **External Linking:** Links to partner websites and relevant external resources to provide more value and context.
*   **Image Alt Text:** Added descriptive alt text to images to improve accessibility and SEO.
*   **Concise and Focused:** Streamlined the text to be more direct and impactful, removing unnecessary fluff.
*   **Use Case Section:** Added Use Cases with images.
*   **Community Section:** Improved and reformatted with clear links and CTAs to connect with the community
*   **WeChat QR Code:** Kept and reorganized the information about WeChat for better display.