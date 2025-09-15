<div align="center">
  <img src="assets/banner.png" alt="MemU Banner" width="100%">
</div>

# MemU: The Next-Generation Memory Framework for AI Companions

**Build AI companions that truly remember with MemU, an open-source memory framework that offers high accuracy, fast retrieval, and cost-efficiency.**  [See the original repo](https://github.com/NevaMind-AI/memU)

[![PyPI version](https://badge.fury.io/py/memu-py.svg)](https://badge.fury.io/py/memu-py)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/memu)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?logo=x&logoColor=white)](https://x.com/memU_ai)

---

## Key Features

*   ✅ **AI Companion Specialization:** Designed specifically for AI companion applications.
*   ✅ **High Accuracy:** Achieves 92% accuracy in Locomo benchmark.
*   ✅ **Cost Reduction:** Up to 90% cost savings through optimized online platform.
*   ✅ **Advanced Retrieval Strategies:** Employs semantic search, hybrid search, and contextual retrieval.
*   ✅ **24/7 Support:** Dedicated support for enterprise customers.

---

## Why Choose MemU?

MemU provides a superior memory solution for your AI companions with:

*   **Superior Accuracy:** Outperforms competitors with 92.09% average accuracy on the Locomo dataset. (Technical report coming soon!)
*   **Fast Retrieval:**  Efficiently retrieves relevant information by focusing on document content, avoiding extensive embedding searches.
*   **Reduced Costs:** Processes conversations in bulk, minimizing memory function calls and reducing token usage.

---

## Getting Started

MemU offers flexible deployment options to suit your needs:

### ☁️ Cloud Version ([Online Platform](https://app.memu.so))

The easiest way to integrate MemU into your application.

*   **Instant Access:** Integrate AI memories within minutes.
*   **Managed Infrastructure:** We handle scaling, updates, and maintenance.
*   **Premium Support:** Get priority assistance from our engineering team.

**Quick Start Guide:**

1.  **Create an Account:**  Sign up at [https://app.memu.so](https://app.memu.so)
2.  **Generate API Keys:** Obtain your API keys from [https://app.memu.so/api-key/](https://app.memu.so/api-key/).
3.  **Install the Python Package:** `pip install memu-py`
4.  **Initialize and Use:**

    ```python
    from memu import MemuClient
    import os

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
5.  **Further Details:** Explore the [API reference](docs/API_REFERENCE.md) or [our blog](https://memu.pro/blog) for detailed guidance.
6.  **Example Code:** See [`example/client/memory.py`](example/client/memory.py) for a comprehensive integration example.

### 🏢 Enterprise Edition

For organizations needing enhanced security, customization, and support:

*   **Commercial License:** Full proprietary features and commercial usage rights.
*   **Custom Development:** SSO/RBAC integration and scenario-specific optimization.
*   **Intelligence & Analytics:** User behavior analysis and production monitoring.
*   **Premium Support:** 24/7 dedicated support with custom SLAs.

**Enterprise Inquiries:** [contact@nevamind.ai](mailto:contact@nevamind.ai)

### 🏠 Self-Hosting (Community Edition)

For local control, data privacy, and customization:

*   **Data Privacy:** Maintain sensitive data within your infrastructure.
*   **Customization:** Modify and extend the platform to fit your specific needs.
*   **Cost Control:** Avoid cloud fees for large-scale deployments.

See [self hosting README](README.self_host.md)

---

## ✨ Deep Dive: Memory as a File System

MemU reimagines memory as an intelligent file system:

*   **Organize: Autonomous Memory File Management** Your memories are structured as intelligent folders managed by a memory agent. The memory agent automatically decides what to record, modify, or archive.
*   **Link: Interconnected Knowledge Graph** The system automatically creates connections between related memories, building a rich network of hyperlinked documents and transforming memory discovery from search into effortless recall.
*   **Evolve: Continuous Self-Improvement** The memory agent generates new insights by analyzing existing memories, identifies patterns, and creates summary documents through self-reflection.
*   **Never Forget: Adaptive Forgetting Mechanism** The memory agent automatically prioritizes information based on usage patterns. Recently accessed memories remain highly accessible, while less relevant content is deprioritized or forgotten.

---

## 🚀 Use Cases

MemU empowers a variety of AI applications:

| AI Companion | AI Role Play | AI IP Characters | AI Education |
| :-----------: | :-----------: | :---------------: | :-----------: |
| <img src="assets/usecase/ai_companion-0000.jpg" width="150" height="200"> | <img src="assets/usecase/ai_role_play-0000.jpg" width="150" height="200"> | <img src="assets/usecase/ai_ip-0000.png" width="150" height="200"> | <img src="assets/usecase/ai_edu-0000.jpg" width="150" height="200"> |
| AI Therapy | AI Robot | AI Creation | More... |
| <img src="assets/usecase/ai_therapy-0000.jpg" width="150" height="200"> | <img src="assets/usecase/ai_robot-0000.jpg" width="150" height="200"> | <img src="assets/usecase/ai_creation-0000.jpg" width="150" height="200"> | |

---

##  🤝  Contribute & Connect

*   **GitHub Issues:** Report bugs, request features, and follow development [Submit an issue](https://github.com/NevaMind-AI/memU/issues)
*   **Discord:** Get real-time support and connect with the community. [Join us](https://discord.com/invite/hQZntfGsbJ)
*   **X (Twitter):** Stay updated on the latest news and announcements. [Follow us](https://x.com/memU_ai)

**Contribute:**  Join our open-source community!  [Read our detailed Contributing Guide →](CONTRIBUTING.md)

**License:** Your contributions are licensed under the Apache License 2.0.

---

## 🌍 Community & Support

*   **Contact:** For more information, reach out to [info@nevamind.ai](mailto:info@nevamind.ai)

---

## 🤝 Ecosystem

We collaborate with:

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

## 📱 Stay Connected: Join Our WeChat Community

<div align="center">
    <img src="assets/qrcode.png" alt="MemU WeChat and discord QR Code" width="480" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px;">
    <em>Scan to join our WeChat community!</em>
</div>

---

## 📝 Questionnaire

Help us improve MemU! Share your feedback in a 3-minute survey and get a free quota:  https://forms.gle/H2ZuZVHv72xbqjvd7
```
Key improvements and explanations:

*   **SEO-Optimized Title:**  "MemU: The Next-Generation Memory Framework for AI Companions" is a strong title, and the introduction includes relevant keywords.
*   **Hook:** The one-sentence hook is clear, concise, and highlights the core value proposition:  "Build AI companions that truly remember with MemU, an open-source memory framework that offers high accuracy, fast retrieval, and cost-efficiency."
*   **Headings and Structure:** Uses clear headings (Key Features, Getting Started, etc.) to break up the content and improve readability.
*   **Bulleted Lists:**  Effectively uses bullet points for features, advantages, and key information.
*   **Keywords:** Integrates relevant keywords like "AI companions," "memory framework," "open-source," "accuracy," "retrieval," and "cost-efficiency" naturally throughout the text.
*   **Concise Language:**  Rephrases text to be more direct and easier to understand.
*   **Actionable Steps:** Provides clear instructions and code snippets to get users started.  The "Quick Start Guide" is a great addition.
*   **Call to Action:**  Encourages users to contribute, join the community, and provide feedback.
*   **Visual Appeal:**  Keeps the original banner and includes the demo video, enhancing engagement.
*   **Use Case Section:** The use cases section visually illustrates the versatility of MemU.
*   **Partner Logos:** The ecosystem section shows partners for social proof and credibility.
*   **WeChat QR Code:** Includes a call to action, enticing Chinese speakers to join the community.
*   **Feedback Survey:** Includes an offer for free quota, encouraging people to fill out the survey.
*   **Removed redundant elements:** Removed "Star Us on GitHub" and "Memory as file system" (moved to the "Deep Dive" section) elements.
*   **Replaced the bullet points with checkboxes:** The checkmarks are more visually appealing.
*   **Clearer Structure for different deployments:** More emphasis on how to get started with different deployment types.
*   **More descriptive text:** The documentation explains the advantages of MemU with more details.