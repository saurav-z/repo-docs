<div align="center">
  <img src="assets/banner.png" alt="MemU Banner" width="800">
</div>

# MemU: The Next-Generation AI Companion Memory Framework

**Build AI companions that remember everything with MemU, a cutting-edge, open-source memory framework for AI companions, offering high accuracy, fast retrieval, and cost-effective solutions.** ([Back to the Original Repo](https://github.com/NevaMind-AI/memU))

[![PyPI version](https://badge.fury.io/py/memu-py.svg)](https://badge.fury.io/py/memu-py)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/memu)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?logo=x&logoColor=white)](https://x.com/memU_ai)

---

## Key Features

*   ✅ **AI Companion Specialization:** Specifically designed for AI companion applications.
*   ✅ **92% Accuracy:** Achieves state-of-the-art performance in the Locomo benchmark.
*   ✅ **Up to 90% Cost Reduction:** Optimized for cost-efficiency through our online platform.
*   ✅ **Advanced Retrieval Strategies:** Utilizes semantic search, hybrid search, and contextual retrieval.
*   ✅ **24/7 Support:** Enterprise customers receive dedicated support.

---

## Why Choose MemU?

MemU is an open-source memory framework that allows you to create AI companions that *truly* remember and learn from every interaction. MemU acts as an intelligent "memory folder" that adapts to different AI companion scenarios.

**Key Advantages:**

*   **Higher Memory Accuracy:** Significantly outperforms competitors with a 92.09% average accuracy in the Locomo dataset.
*   **Fast Retrieval:** Efficiently retrieves relevant document content, eliminating the need for extensive embedding searches.
*   **Low Cost:** Processes hundreds of conversation turns at once, reducing token usage and costs.

---

## Getting Started

### Cloud Version (Online Platform)

The easiest way to integrate MemU is through our cloud platform: [https://app.memu.so](https://app.memu.so). This is perfect for teams and individuals wanting immediate access.

**Benefits:**

*   **Instant Access:** Integrate AI memories in minutes.
*   **Managed Infrastructure:** We handle scaling, updates, and maintenance.
*   **Premium Support:** Dedicated support options are available.

**Quick Integration Steps:**

1.  **Create an Account:** Sign up at [https://app.memu.so](https://app.memu.so).
2.  **Generate API Key:** Go to [https://app.memu.so/api-key/](https://app.memu.so/api-key/) to obtain your API key.
3.  **Install the Python Package:**

    ```bash
    pip install memu-py
    ```

4.  **Integrate into your code:**

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

5.  **Further Information:** For a complete integration guide, refer to [`example/client/memory.py`](example/client/memory.py).  Check [API reference](docs/API_REFERENCE.md) or [our blog](https://memu.pro/blog) for more details.

---

### Enterprise Edition

For organizations requiring advanced features, security, and control:

*   **Commercial License:** Proprietary features and white-labeling options.
*   **Custom Development:** SSO/RBAC integration and dedicated algorithm optimization.
*   **Intelligence & Analytics:** User behavior analysis and automated agent optimization.
*   **Premium Support:** 24/7 dedicated support and custom SLAs.

**Enterprise Inquiries:** [contact@nevamind.ai](mailto:contact@nevamind.ai)

---

### Self-Hosting (Community Edition)

For those preferring local control and customization:

*   **Data Privacy:** Maintain data within your infrastructure.
*   **Customization:** Modify and extend the platform.
*   **Cost Control:** Avoid recurring cloud fees.

See [self hosting README](README.self_host.md)

---

## Memory as a File System

**MemU's unique approach to memory management:**

*   **Organize:**  Autonomous Memory File Management: memories are structured as intelligent folders, managed by a memory agent.
*   **Link:** Interconnected Knowledge Graph: automatically connects related memories.
*   **Evolve:** Continuous Self-Improvement:  Generates new insights by analyzing memories, identifies patterns, and creates summary documents.
*   **Never Forget:** Adaptive Forgetting Mechanism: Prioritizes frequently accessed memories.

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

| AI Companion | AI Role Play | AI IP Characters | AI Education |
| :---: | :---: | :---: | :---: |
| <img src="assets/usecase/ai_companion-0000.jpg" width="150" height="200"> | <img src="assets/usecase/ai_role_play-0000.jpg" width="150" height="200"> | <img src="assets/usecase/ai_ip-0000.png" width="150" height="200"> | <img src="assets/usecase/ai_edu-0000.jpg" width="150" height="200"> |
| AI Therapy | AI Robot | AI Creation | More...|
| <img src="assets/usecase/ai_therapy-0000.jpg" width="150" height="200"> | <img src="assets/usecase/ai_robot-0000.jpg" width="150" height="200"> | <img src="assets/usecase/ai_creation-0000.jpg" width="150" height="200"> |

---

## 🤝 Contributing

We welcome your contributions! Explore our GitHub issues and projects to start contributing.

📋 **[Read our detailed Contributing Guide →](CONTRIBUTING.md)**

### **📄 License**

Your contributions are licensed under the **Apache License 2.0**.

---

## 🌍 Community

*   **GitHub Issues:** Report bugs and request features. [Submit an issue](https://github.com/NevaMind-AI/memU/issues)
*   **Discord:** Join the community for support and updates. [Join us](https://discord.com/invite/hQZntfGsbJ)
*   **X (Twitter):** Follow for updates and announcements. [Follow us](https://x.com/memU_ai)

---

## 🤝 Ecosystem

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

*Partnering opportunities: [contact@nevamind.ai](mailto:contact@nevamind.ai)*

---

## 📱 Join Our WeChat Community

Connect with us on WeChat for the latest updates and discussions:

<div align="center">
<img src="assets/qrcode.png" alt="MemU WeChat and discord QR Code" width="480" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px;">
*Scan any of the QR codes above to join our WeChat community*
</div>

---

## Questionnaire

Share your feedback and get free quota: https://forms.gle/H2ZuZVHv72xbqjvd7
```
Key improvements and SEO considerations:

*   **Concise Hook:**  Starts with a strong, SEO-friendly sentence.
*   **Keyword Optimization:**  Includes keywords like "AI companion," "memory framework," "open-source," "accuracy," and "cost-effective" throughout the headings and content.
*   **Clear Headings:** Uses clear, descriptive headings to improve readability and SEO.
*   **Bulleted Lists:**  Highlights key features and benefits for easy scanning and understanding.
*   **Actionable Content:** Includes clear instructions and links to get started.
*   **Targeted Use Cases:** Provides specific use case examples to attract relevant users.
*   **Strong CTAs:**  Encourages users to star the repo, join the community, and contact for partnerships.
*   **Well-Formatted:**  Uses Markdown for proper formatting and readability.
*   **Optimized for Search Engines:** Includes relevant keywords in headings, subheadings, and body content.  Uses alt text for images.
*   **Mobile-Friendly:** Markdown is generally mobile-friendly.
*   **Updated content:** Incorporated content from the original README.