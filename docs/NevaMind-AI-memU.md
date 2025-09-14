<div align="center">

![MemU Banner](assets/banner.png)

</div>

# MemU: The Next-Generation Memory Framework for AI Companions

**Build AI companions that truly remember with MemU, an open-source memory framework for high accuracy, fast retrieval, and low cost.** ([Original Repository](https://github.com/NevaMind-AI/memU))

[![PyPI version](https://badge.fury.io/py/memu-py.svg)](https://badge.fury.io/py/memu-py)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/memu)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?logo=x&logoColor=white)](https://x.com/memU_ai)

## Key Features of MemU:

*   ✅ **AI Companion Specialization:** Optimized for AI companion applications.
*   ✅ **92% Accuracy:** Achieve state-of-the-art performance in the Locomo benchmark.
*   ✅ **Up to 90% Cost Reduction:** Optimized for cost-efficiency.
*   ✅ **Advanced Retrieval Strategies:** Utilize semantic search, hybrid search, and contextual retrieval for optimal results.
*   ✅ **24/7 Support:** Dedicated support for enterprise customers.

## Why Choose MemU?

MemU is designed to be the intelligent "memory folder" for your AI companions, adapting to the unique needs of various AI companion scenarios. With MemU, your AI companions will learn and grow with every interaction.

## Core Advantages:

*   **Higher Memory Accuracy:** MemU achieves 92.09% average accuracy in the Locomo dataset.
*   **Fast Retrieval:** Efficiently retrieves relevant information by categorizing and organizing information into documents.
*   **Low Cost:** Process hundreds of conversation turns at once, minimizing token usage.

## Get Started with MemU:

### ☁️ Cloud Version ([Online Platform](https://app.memu.so))

Get up and running quickly with the cloud version, perfect for immediate access.

*   **Instant Access:** Integrate AI memories in minutes.
*   **Managed Infrastructure:** We handle scaling, updates, and maintenance.
*   **Premium Support:** Priority assistance from our engineering team (with subscription).

**How to Get Started:**

1.  **Create an Account:** Create an account at [https://app.memu.so](https://app.memu.so).
2.  **Generate API Keys:** Go to [https://app.memu.so/api-key/](https://app.memu.so/api-key/) to generate your API keys.
3.  **Install the Python Package:**

    ```bash
    pip install memu-py
    ```
4.  **Example Usage:**

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

    Check [API reference](docs/API_REFERENCE.md) or [our blog](https://memu.pro/blog) for more details.

📖 **See [`example/client/memory.py`](example/client/memory.py) for complete integration details**

### 🏢 Enterprise Edition

For organizations requiring maximum security, customization, control, and the best quality.

*   **Commercial License:** Full proprietary features and commercial usage rights.
*   **Custom Development:** SSO/RBAC integration, dedicated algorithm team.
*   **Intelligence & Analytics:** User behavior analysis, real-time monitoring, and agent optimization.
*   **Premium Support:** 24/7 dedicated support.

📧 **Enterprise Inquiries:** [contact@nevamind.ai](mailto:contact@nevamind.ai)

### 🏠 Self-Hosting (Community Edition)

For users who prefer local control and data privacy.

*   **Data Privacy:** Keep sensitive data within your infrastructure.
*   **Customization:** Modify and extend the platform to fit your needs.
*   **Cost Control:** Avoid recurring cloud fees.

See [self hosting README](README.self_host.md)

---

## ✨ Key Features in Detail

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

*   **Organize:** Autonomous Memory File Management
*   **Link:** Interconnected Knowledge Graph
*   **Evolve:** Continuous Self-Improvement
*   **Never Forget:** Adaptive Forgetting Mechanism

---

## 🎓 Use Cases

MemU is versatile and can be applied to numerous AI companion-related applications.

|  <img src="assets/usecase/ai_companion-0000.jpg" width="150" height="200"><br>**AI Companion** | <img src="assets/usecase/ai_role_play-0000.jpg" width="150" height="200"><br>**AI Role Play** | <img src="assets/usecase/ai_ip-0000.png" width="150" height="200"><br>**AI IP Characters** | <img src="assets/usecase/ai_edu-0000.jpg" width="150" height="200"><br>**AI Education** |
|---|---|---|---|
| <img src="assets/usecase/ai_therapy-0000.jpg" width="150" height="200"><br>**AI Therapy** | <img src="assets/usecase/ai_robot-0000.jpg" width="150" height="200"><br>**AI Robot** | <img src="assets/usecase/ai_creation-0000.jpg" width="150" height="200"><br>**AI Creation** | More...|

---

## 🤝 Contributing

Contribute to the future of MemU. Your contributions are welcomed!

📋 **[Read our detailed Contributing Guide →](CONTRIBUTING.md)**

### **📄 License**

By contributing to MemU, you agree that your contributions will be licensed under the **Apache License 2.0**.

---

## 🌍 Community and Support

*   **GitHub Issues:** Report bugs, request features. [Submit an issue](https://github.com/NevaMind-AI/memU/issues)
*   **Discord:** Get real-time support and connect with the community. [Join us](https://discord.com/invite/hQZntfGsbJ)
*   **X (Twitter):** Stay updated on news and announcements. [Follow us](https://x.com/memU_ai)
*   **Contact:** [info@nevamind.ai](mailto:info@nevamind.ai)

---

## 🤝 Ecosystem

### Development Tools
<div align="center">

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

### WeChat Community

<div align="center">
<img src="assets/qrcode.png" alt="MemU WeChat and discord QR Code" width="480" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px;">
</div>

*Scan the QR code above to join our WeChat community.*

---

## Feedback

Help us improve MemU! Share your feedback in our 3-minute survey: https://forms.gle/H2ZuZVHv72xbqjvd7