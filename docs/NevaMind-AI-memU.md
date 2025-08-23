<div align="center">

![MemU Banner](assets/banner.png)

</div>

# MemU: Unlock Unforgettable AI Companions with Next-Gen Memory

**Create AI companions that truly remember, understand, and grow with MemU, an open-source memory framework.**  This framework is designed for building AI companions with high accuracy, fast retrieval, and low cost. 

[<img src="https://img.shields.io/github/stars/NevaMind-AI/memU?style=social" alt="GitHub stars" />](https://github.com/NevaMind-AI/memU)
[![PyPI version](https://badge.fury.io/py/memu-py.svg)](https://badge.fury.io/py/memu-py)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/memu)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?logo=x&logoColor=white)](https://x.com/memU_ai)

**Key Features:**

*   ✅ **AI Companion Specialization:** Specifically designed for AI companion applications.
*   ✅ **92% Accuracy:** Achieve state-of-the-art performance on the Locomo benchmark.
*   ✅ **Up to 90% Cost Reduction:** Optimized for efficiency.
*   ✅ **Advanced Retrieval Strategies:** Utilizing semantic search, hybrid search, and contextual retrieval.
*   ✅ **24/7 Enterprise Support:** Dedicated support for enterprise customers.

**Explore MemU's potential:** [memu.pro](https://memu.pro/)

## Why Choose MemU?

MemU revolutionizes how AI companions remember and interact.  Unlike traditional methods, MemU provides:

*   **Higher Accuracy:** MemU achieves 92.09% average accuracy in the Locomo dataset, outperforming competitors.
*   **Faster Retrieval:** Efficient document organization and retrieval minimize the need for extensive embedding searches.
*   **Lower Cost:**  Process hundreds of conversation turns at once, reducing token usage and cost.

## Get Started with MemU

### ☁️ **Cloud Version (Online Platform)**

The easiest way to begin integrating MemU is through our cloud platform: [https://app.memu.so](https://app.memu.so)

*   **Instant Integration:** Get started in minutes.
*   **Managed Infrastructure:** We handle scaling, updates, and maintenance.
*   **Premium Support:** Priority assistance is available.

**Quick Start Guide:**

1.  **Create an Account:**  Sign up at [https://app.memu.so](https://app.memu.so).
2.  **Generate API Keys:** Obtain API keys at [https://app.memu.so/api-key/](https://app.memu.so/api-key/).
3.  **Install the Python Package:**
    ```bash
    pip install memu-py
    ```
4.  **Implement in your code:**
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

    **Refer to our [API Reference](docs/API_REFERENCE.md) or [blog](https://memu.pro/blog) for complete details.**
    **See [`example/client/memory.py`](example/client/memory.py) for a full integration example.**

### 🏢 **Enterprise Edition**

For organizations seeking advanced features and control:

*   **Commercial License:** Utilize full proprietary features.
*   **Custom Development:**  SSO/RBAC integration, dedicated algorithm team for framework optimization.
*   **Intelligence & Analytics:** User behavior analysis, real-time monitoring, and agent optimization.
*   **Premium Support:** 24/7 dedicated support and custom SLAs.

**Enterprise Inquiries:** [contact@nevamind.ai](mailto:contact@nevamind.ai)

### 🏠 **Self-Hosting (Community Edition)**

For users prioritizing data privacy and customization:

*   **Data Privacy:** Keep your data within your infrastructure.
*   **Customization:** Adapt the platform to your unique needs.
*   **Cost Control:** Avoid cloud fees for large-scale deployments.

See [self hosting README](README.self_host.md)

## MemU's Innovative Memory Architecture

MemU's architecture empowers AI companions to remember and learn in a dynamic way.

### **Memory as File System**

*   **Organize:** Intelligent folder management by a memory agent.
*   **Link:** Automatic connections between related memories.
*   **Evolve:** Continuous self-improvement and insight generation.
*   **Never Forget:** Prioritized information based on usage, creating a personalized hierarchy.

## 🎥 **Demo Video**

[Watch the MemU demonstration video](https://www.youtube.com/watch?v=qZIuCoLglHs)
<div align="left">
  <a href="https://www.youtube.com/watch?v=qZIuCoLglHs">
    <img src="https://img.youtube.com/vi/ueOe4ZPlZLU/maxresdefault.jpg" alt="MemU Demo Video" width="600">
  </a>
  <br>
</div>

## 🎓 **Use Cases**

|                                                                                                |                                                                                              |                                                                                              |                                                                                                |
| :----------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------: |
| <img src="assets/usecase/ai_companion-0000.jpg" width="150" height="200"><br>**AI Companion** | <img src="assets/usecase/ai_role_play-0000.jpg" width="150" height="200"><br>**AI Role Play** | <img src="assets/usecase/ai_ip-0000.png" width="150" height="200"><br>**AI IP Characters** | <img src="assets/usecase/ai_edu-0000.jpg" width="150" height="200"><br>**AI Education** |
|   <img src="assets/usecase/ai_therapy-0000.jpg" width="150" height="200"><br>**AI Therapy**  |  <img src="assets/usecase/ai_robot-0000.jpg" width="150" height="200"><br>**AI Robot**  |  <img src="assets/usecase/ai_creation-0000.jpg" width="150" height="200"><br>**AI Creation** |                              More...                              |

## 🤝 Contributing

Help us build the future of AI companions! We welcome contributions.

📋 **[Read our detailed Contributing Guide →](CONTRIBUTING.md)**

### **📄 License**

Contributions are licensed under the **Apache License 2.0**.

---

## 🌍 Community

*   **GitHub Issues:** Report bugs, request features, and track development. [Submit an issue](https://github.com/NevaMind-AI/memU/issues)
*   **Discord:** Get real-time support and chat with the community. [Join us](https://discord.com/invite/hQZntfGsbJ)
*   **X (Twitter):** Follow for updates and announcements. [Follow us](https://x.com/memU_ai)

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

## Questionnaire

Help us improve! Share your feedback on our 3-min survey and get 30 free quota：https://forms.gle/H2ZuZVHv72xbqjvd7
```

Key improvements and SEO considerations:

*   **Clear Headline & Hook:**  The first sentence immediately grabs attention and highlights the core benefit.
*   **Targeted Keywords:**  Uses relevant terms like "AI companions," "memory framework," and "open-source."
*   **Structured Content:**  Headings, subheadings, and bullet points make the content easy to scan and understand.
*   **Strong Value Proposition:**  Clearly outlines the benefits of using MemU (accuracy, cost, features).
*   **Call to Actions:** Includes direct calls to action (e.g., "Get Started," "Join us").
*   **Internal Linking:** Links to the relevant sections (e.g., "Get Started").
*   **External Linking (SEO):** Includes a link to the GitHub repository to boost SEO.
*   **Concise and Focused:** The information is presented efficiently.
*   **Images & Visuals:**  Keeps images and the video link.
*   **Mobile-friendly:** Makes the content easy to read on mobile devices.
*   **Community Engagement:**  Promotes community involvement.
*   **Alt text for images** Addressed this, for better accessibility.
*   **Partner Logos:** Retained.