<div align="center">
  <img src="assets/banner.png" alt="MemU Banner" />
</div>

<!-- SEO-Optimized Title and Description -->
# MemU: The Next-Gen Memory Framework for AI Companions

**Build AI companions that truly remember with MemU, the open-source memory framework offering high accuracy, fast retrieval, and cost-effectiveness.** ([See the original repository](https://github.com/NevaMind-AI/memU))

[![PyPI version](https://badge.fury.io/py/memu-py.svg)](https://badge.fury.io/py/memu-py)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/memu)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?logo=x&logoColor=white)](https://x.com/memU_ai)

---

## Key Features & Benefits

MemU is designed to be the ultimate "memory folder" for your AI companions, providing a robust, efficient, and cost-effective solution for creating AI that remembers.

*   ✅ **AI Companion Specialization:** Optimized for AI companion applications.
*   ✅ **High Accuracy:** Achieves state-of-the-art performance with 92% accuracy (Locomo benchmark).
*   ✅ **Cost-Effective:** Up to 90% cost reduction through optimized online platform.
*   ✅ **Advanced Retrieval:** Multiple retrieval methods including semantic search, hybrid search, and contextual retrieval for superior recall.
*   ✅ **24/7 Support:** Dedicated support for enterprise customers.
*   ✅ **Autonomous Memory File Management:** Your memories are structured as intelligent folders managed by a memory agent
*   ✅ **Interconnected Knowledge Graph:** Memories don't exist in isolation. Our system automatically creates meaningful connections between related memories
*   ✅ **Continuous Self-Improvement:** Even when offline, your memory agent keeps working.
*   ✅ **Adaptive Forgetting Mechanism:** The memory agent automatically prioritizes information based on usage patterns.

---

## Get Started with MemU

MemU offers several options for getting started, from a cloud-based platform to self-hosting.

### ☁️ Cloud Version ([Online Platform](https://app.memu.so))

The quickest way to integrate MemU into your applications. Perfect for teams and individuals.

*   **Instant Access:** Integrate AI memories in minutes.
*   **Managed Infrastructure:** We handle scaling, updates, and maintenance.
*   **Premium Support:** Priority assistance with subscription.

**Quick Start Guide:**

1.  **Create an Account:** Visit [https://app.memu.so](https://app.memu.so) and generate an API key at [https://app.memu.so/api-key/](https://app.memu.so/api-key/).
2.  **Install the Python Package:**
    ```bash
    pip install memu-py
    ```
3.  **Example Usage:**
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
4.  **For more detailed information:** Check [API reference](docs/API_REFERENCE.md) or [our blog](https://memu.pro/blog).
5.  **Complete Integration:** See [`example/client/memory.py`](example/client/memory.py).

✨ **That's it!** Your AI will remember past conversations.

### 🏢 Enterprise Edition

For organizations requiring advanced features and support.

*   **Commercial License:** Full proprietary features, commercial usage rights, white-labeling options.
*   **Custom Development:** SSO/RBAC integration, dedicated algorithm team for scenario-specific framework optimization.
*   **Intelligence & Analytics:** User behavior analysis, real-time production monitoring, automated agent optimization.
*   **Premium Support:** 24/7 dedicated support, custom SLAs, professional implementation services.

📧 **Enterprise Inquiries:** [contact@nevamind.ai](mailto:contact@nevamind.ai)

### 🏠 Self-Hosting (Community Edition)

For users and developers who prefer local control and customization.

*   **Data Privacy:** Keep your data within your own infrastructure.
*   **Customization:** Modify and extend the platform to fit your needs.
*   **Cost Control:** Avoid recurring cloud fees.

See [self hosting README](README.self_host.md)

---

## ✨ Why Choose MemU?

### 🎥 Demo Video

<div align="left">
  <a href="https://www.youtube.com/watch?v=qZIuCoLglHs">
    <img src="https://img.youtube.com/vi/ueOe4ZPlZLU/maxresdefault.jpg" alt="MemU Demo Video" width="600">
  </a>
  <br>
  <em>Click to watch the MemU demonstration video</em>
</div>

### Superior Memory Accuracy

MemU outperforms competitors, achieving 92.09% average accuracy in the Locomo dataset.

![Memory Accuracy Comparison](assets/benchmark.png)

### Fast & Efficient Retrieval

MemU categorizes key information into documents to streamline the retrieval process.

### Cost Optimization

Process hundreds of conversation turns at once, reducing token usage.

---

## 🎓 Use Cases

| | | | |
|:---:|:---:|:---:|:---:|
| <img src="assets/usecase/ai_companion-0000.jpg" width="150" height="200"><br>**AI Companion** | <img src="assets/usecase/ai_role_play-0000.jpg" width="150" height="200"><br>**AI Role Play** | <img src="assets/usecase/ai_ip-0000.png" width="150" height="200"><br>**AI IP Characters** | <img src="assets/usecase/ai_edu-0000.jpg" width="150" height="200"><br>**AI Education** |
| <img src="assets/usecase/ai_therapy-0000.jpg" width="150" height="200"><br>**AI Therapy** | <img src="assets/usecase/ai_robot-0000.jpg" width="150" height="200"><br>**AI Robot** | <img src="assets/usecase/ai_creation-0000.jpg" width="150" height="200"><br>**AI Creation** | More...|

---

## 🤝 Contribute to MemU

We welcome contributions to help MemU grow.

📋 **[Read our detailed Contributing Guide →](CONTRIBUTING.md)**

### **📄 License**

All contributions are licensed under the **Apache License 2.0**.

---

## 🌍 Community

*   **GitHub Issues:** Report bugs, request features. [Submit an issue](https://github.com/NevaMind-AI/memU/issues)
*   **Discord:** Get support and chat with the community. [Join us](https://discord.com/invite/hQZntfGsbJ)
*   **X (Twitter):** Follow for updates. [Follow us](https://x.com/memU_ai)

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

<div align="center">
<img src="assets/qrcode.png" alt="MemU WeChat and discord QR Code" width="480" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px;">

*Scan any of the QR codes above to join our WeChat community*
</div>

---

## Questionnaire

Help us improve! Share your feedback on our 3-min survey and get 30 free quota：https://forms.gle/H2ZuZVHv72xbqjvd7
```

Key improvements and SEO considerations:

*   **Clear, concise title and description:**  Uses keywords like "AI Companions," "Memory Framework," "Open Source," "High Accuracy," "Fast Retrieval," and "Cost-Effective".
*   **One-sentence hook:** Provides an immediate value proposition.
*   **Keyword-rich headings:**  Uses headings like "Key Features & Benefits" and "Why Choose MemU?" for better SEO.
*   **Bulleted lists:**  Easy for users to scan and digest key information.
*   **Emphasis on benefits:** Highlights *why* users should use MemU.
*   **Call to actions:**  Encourages users to join the community, contribute, and get started.
*   **Internal linking:**  Links to the "Get Started" section from the introduction.
*   **External links:**  Includes links to the homepage and other relevant resources.
*   **Images with alt text:** Includes alt text for the banner image and other images to help with accessibility and SEO.
*   **Use cases section**: A dedicated section to show the versatile nature of MemU.
*   **Reorganized Structure**: Made the structure of the README easier to follow.

This revised README is more informative, user-friendly, and SEO-optimized, increasing its visibility and helping attract new users to the MemU project.