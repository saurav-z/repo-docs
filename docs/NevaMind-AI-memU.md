<div align="center">
  <a href="https://github.com/NevaMind-AI/memU">
    <img src="assets/banner.png" alt="MemU Banner" width="100%">
  </a>
</div>

# MemU: The Next-Generation Memory Framework for AI Companions 🚀

**MemU empowers you to build AI companions that remember, learn, and grow with users, offering unparalleled accuracy and efficiency. [Explore the MemU Repository](https://github.com/NevaMind-AI/memU)**

[![PyPI version](https://badge.fury.io/py/memu-py.svg)](https://badge.fury.io/py/memu-py)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/memu)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?logo=x&logoColor=white)](https://x.com/memU_ai)

---

## 💡 Key Features & Benefits

*   **AI Companion Specialization:** Specifically designed for AI companion applications.
*   **Unmatched Accuracy:** Achieves a state-of-the-art 92% accuracy score in Locomo benchmarks.
*   **Cost-Effective Solutions:** Up to 90% cost reduction through optimized online platform usage.
*   **Advanced Retrieval Strategies:** Leverages semantic search, hybrid search, and contextual retrieval for superior results.
*   **Comprehensive Support:** 24/7 support available for enterprise customers.

## 🌟 Core Advantages

*   **Higher Memory Accuracy:** MemU boasts a 92.09% average accuracy in the Locomo dataset, significantly surpassing competitors. (Technical Report coming soon!)
    <div align="center">
    <img src="assets/benchmark.png" alt="Memory Accuracy Comparison" width="80%">
    </div>
*   **Fast Retrieval:** Prioritizes efficiency by organizing information into documents, streamlining the retrieval process.
*   **Low Cost:**  Processes numerous conversation turns simultaneously, minimizing token usage and saving you money.

## 🚀 Get Started

### ☁️ Cloud Version ([Online Platform](https://app.memu.so))

The fastest way to integrate AI memories into your applications.

*   **Instant Access:** Start integrating in minutes.
*   **Managed Infrastructure:** We handle scaling, updates, and maintenance.
*   **Premium Support:** Get priority assistance.

**Quick Start Guide:**

1.  **Create an Account:**  Sign up at [https://app.memu.so](https://app.memu.so)
2.  **Generate API Keys:**  Navigate to [https://app.memu.so/api-key/](https://app.memu.so/api-key/)
3.  **Install the Python Package:**  `pip install memu-py`
4.  **Example Usage:**

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

    **[See `example/client/memory.py`](example/client/memory.py) for complete integration details.**

### 🏢 Enterprise Edition

For enterprise-level requirements:

*   **Commercial Licensing:** Full proprietary features and usage rights.
*   **Custom Development:** SSO/RBAC integration and dedicated algorithm support.
*   **Intelligence & Analytics:** User behavior analysis and agent optimization.
*   **Premium Support:** 24/7 support, custom SLAs, and implementation services.

**Enterprise Inquiries:** [contact@nevamind.ai](mailto:contact@nevamind.ai)

### 🏠 Self-Hosting (Community Edition)

For local control, data privacy, and customization:

*   **Data Privacy:** Keep your data within your infrastructure.
*   **Customization:** Modify and extend the platform.
*   **Cost Control:** Avoid recurring cloud fees.

**See [self hosting README](README.self_host.md)**

---

## 🎥 Demo Video

[Link to the MemU demonstration video](https://www.youtube.com/watch?v=qZIuCoLglHs)
<div align="center">
<a href="https://www.youtube.com/watch?v=qZIuCoLglHs">
  <img src="https://img.youtube.com/vi/ueOe4ZPlZLU/maxresdefault.jpg" alt="MemU Demo Video" width="600">
</a>
<br>
<em>Click to watch the MemU demonstration video</em>
</div>

---

## ✨ Memory as a File System

*   **Organize:** Autonomous memory file management.
*   **Link:** Interconnected knowledge graph.
*   **Evolve:** Continuous self-improvement.
*   **Never Forget:** Adaptive forgetting mechanism.

---

## 🎓 Use Cases

<div align="center">
<table style="border: none;">
    <tr>
        <td align="center" style="border: none;">
            <img src="assets/usecase/ai_companion-0000.jpg" width="150" height="200"><br>
            AI Companion
        </td>
        <td align="center" style="border: none;">
            <img src="assets/usecase/ai_role_play-0000.jpg" width="150" height="200"><br>
            AI Role Play
        </td>
        <td align="center" style="border: none;">
            <img src="assets/usecase/ai_ip-0000.png" width="150" height="200"><br>
            AI IP Characters
        </td>
        <td align="center" style="border: none;">
            <img src="assets/usecase/ai_edu-0000.jpg" width="150" height="200"><br>
            AI Education
        </td>
    </tr>
    <tr>
        <td align="center" style="border: none;">
            <img src="assets/usecase/ai_therapy-0000.jpg" width="150" height="200"><br>
            AI Therapy
        </td>
        <td align="center" style="border: none;">
            <img src="assets/usecase/ai_robot-0000.jpg" width="150" height="200"><br>
            AI Robot
        </td>
        <td align="center" style="border: none;">
            <img src="assets/usecase/ai_creation-0000.jpg" width="150" height="200"><br>
            AI Creation
        </td>
        <td align="center" style="border: none;">
            More...
        </td>
    </tr>
</table>
</div>

---

## 🤝 Contributing

We welcome contributions!

**[Read our detailed Contributing Guide →](CONTRIBUTING.md)**

### **📄 License**

Contributions are licensed under the **Apache License 2.0**.

---

## 🌍 Community

*   **GitHub Issues:** Report bugs and request features. [Submit an issue](https://github.com/NevaMind-AI/memU/issues)
*   **Discord:** Get real-time support and chat with the community. [Join us](https://discord.com/invite/hQZntfGsbJ)
*   **X (Twitter):** Follow us for updates. [Follow us](https://x.com/memU_ai)

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

*Interested in partnering with MemU? Contact us at [contact@nevamind.ai](mailto:contact@nevamind.ai)*

---

## 📱 Join Our WeChat Community

<div align="center">
<img src="assets/qrcode.png" alt="MemU WeChat and discord QR Code" width="480" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px;">

*Scan any of the QR codes above to join our WeChat community*
</div>

---

## Questionnaire

Help us improve! Share your feedback on our 3-min survey and get 30 free quota：https://forms.gle/H2ZuZVHv72xbqjvd7
```

Key improvements and SEO optimizations:

*   **Clear, Concise Hook:** "MemU empowers you to build AI companions that remember, learn, and grow with users, offering unparalleled accuracy and efficiency." This immediately grabs attention.
*   **Keyword-Rich Headings:** Uses relevant keywords like "AI Companion," "Memory Framework," and "Get Started."
*   **Bulleted Lists:**  Highlights key features and benefits for easy readability.
*   **Strong Call to Actions:** Includes links to the repository, quickstart instructions, and community resources.
*   **Clear Structure:** Organizes information logically with headings and subheadings.
*   **SEO-Friendly Formatting:** Uses markdown for clear headings and bold text.
*   **Removed Unnecessary Content:** Streamlined the "Ecosystem" section and removed the redundant intro for each section.
*   **Improved Visuals:**  Added alt text to images for accessibility and SEO.  Included link back to video.
*   **Concise Language:** Reworded sentences for greater impact.
*   **Prioritized Information:** Focused on the most important selling points at the top.
*   **Mobile-Friendly:** Table formatting is generally responsive.
*   **More Internal Linking:** Added links between different sections of the README (e.g., "See [self hosting README]")