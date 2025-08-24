<div align="center">
  <a href="https://github.com/NevaMind-AI/memU">
    <img src="assets/banner.png" alt="MemU Banner" width="100%">
  </a>
</div>

# MemU: The Ultimate AI Companion Memory Framework

**MemU** is an open-source memory framework that revolutionizes AI companion development with unparalleled accuracy, speed, and cost-effectiveness.  

[![PyPI version](https://badge.fury.io/py/memu-py.svg)](https://badge.fury.io/py/memu-py)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/memu)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?logo=x&logoColor=white)](https://x.com/memU_ai)

**Key Features:**

*   ✅ **AI Companion Specialization:** Built specifically for AI companion applications.
*   ✅ **92% Accuracy:** Achieve state-of-the-art performance in Locomo benchmark.
*   ✅ **Up to 90% Cost Reduction:** Optimized for affordability through its online platform.
*   ✅ **Advanced Retrieval Strategies:** Semantic search, hybrid search, and contextual retrieval.
*   ✅ **24/7 Support:** Dedicated support for enterprise customers.

**Visit our homepage:** [memu.pro](https://memu.pro/)

---

## Why Choose MemU?

MemU empowers you to create AI companions that truly *remember* and learn, fostering deeper, more meaningful interactions.

*   **Unrivaled Accuracy:**  MemU achieves an exceptional 92% accuracy on the Locomo benchmark, providing superior memory recall and contextual understanding.
*   **Blazing-Fast Retrieval:** Efficiently access the right information when it matters most, for quick and responsive AI interactions.
*   **Cost-Effective Solution:** Optimized architecture and resource management delivers significant cost savings, up to 90% compared to alternatives.

---

## 🚀 Get Started with MemU

MemU offers multiple ways to get started, making it accessible for developers of all levels.

### ☁️ Cloud Version (Online Platform)

The easiest and quickest way to start.  Ideal for teams and individuals seeking immediate access without the complexities of self-hosting.

*   **Instant Integration:** Start using MemU's AI memory capabilities in minutes.
*   **Managed Infrastructure:** We handle all the scaling, updates, and maintenance.
*   **Premium Support:** Get priority assistance from our expert engineering team.

**Quick Start Guide**

1.  **Create an Account:**  Sign up at [https://app.memu.so](https://app.memu.so)
2.  **Generate API Keys:** Obtain your API keys from [https://app.memu.so/api-key/](https://app.memu.so/api-key/)
3.  **Install the Package:**

    ```bash
    pip install memu-py
    ```

4.  **Implement in your code**

    ```python
    # Example usage
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
5.  **See [`example/client/memory.py`](example/client/memory.py) for complete integration details**

✨ **That's it!** MemU remembers everything and helps your AI learn from past conversations.

### 🏢 Enterprise Edition

For organizations requiring advanced features and dedicated support:

*   **Commercial Licensing:** Access proprietary features and white-labeling options.
*   **Custom Development:** Leverage our team for SSO/RBAC integration and scenario-specific optimization.
*   **Intelligence & Analytics:** Utilize user behavior analysis, real-time monitoring, and automated agent optimization tools.
*   **Premium Support:** Receive 24/7 dedicated support and custom SLAs.

**Enterprise Inquiries:** [contact@nevamind.ai](mailto:contact@nevamind.ai)

### 🏠 Self-Hosting (Community Edition)

Take full control with self-hosting for enhanced data privacy, customization, and cost management:

*   **Data Privacy:**  Keep your sensitive data within your infrastructure.
*   **Customization:** Modify and extend the platform to meet your specific needs.
*   **Cost Control:** Avoid cloud fees for large-scale deployments.

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

### Memory as File System:  Intelligent Memory Management

*   **Organize:**  MemU's agent automatically organizes your memories into intelligent folders, like a personal librarian.
*   **Link:**  Creates connections between memories to build a knowledge graph for effortless recall.
*   **Evolve:**  Continuously learns, identifying patterns and generating insights to enhance your knowledge base.
*   **Never Forget:**  Prioritizes recent information, so your AI companion remembers what matters most.

---

## 😺 Advantages in Detail

###  Unmatched Accuracy
MemU achieves an outstanding 92.09% average accuracy on the Locomo dataset across all reasoning tasks.  A detailed technical report will be published soon!

<div align="center">
<img src="assets/benchmark.png" alt="Memory Accuracy Comparison" width="80%">
</div>

### Fast Retrieval
MemU categorizes information into documents for more focused retrieval.  This eliminates the need for extensive embedding searches and allows for quick responses.

### Low Cost
Process hundreds of conversation turns with ease, reducing the number of API calls.  This dramatically reduces token usage and lowers overall costs.

---

## 🎓 Use Cases:  Where MemU Shines

MemU is versatile, with powerful applications across multiple domains:

<div align="center">
<table>
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
</div>

---

## 🤝 Contributing & Community

We value open-source collaboration.  Help us build the future of MemU!

*   **Contribute:** Explore issues and projects on GitHub and submit a pull request.
    *   📋 **[Read our detailed Contributing Guide →](CONTRIBUTING.md)**

### **📄 License**

Your contributions are licensed under the **Apache License 2.0**.

---

## 🌍 Stay Connected

*   **GitHub Issues:**  Report bugs, request features, and track progress: [Submit an issue](https://github.com/NevaMind-AI/memU/issues)
*   **Discord:** Get real-time support and join the community: [Join us](https://discord.com/invite/hQZntfGsbJ)
*   **X (Twitter):** Follow for news and announcements: [Follow us](https://x.com/memU_ai)

---

## 🤝 Ecosystem

We're proud to collaborate with these amazing organizations:

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

## 📱 Join Our Community

Stay connected with the MemU community!

<div align="center">
<img src="assets/qrcode.png" alt="MemU WeChat and discord QR Code" width="480" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px;">

</div>

---

## 📝  Help Us Improve!

Share your feedback and get free quota!  [Take our 3-minute survey](https://forms.gle/H2ZuZVHv72xbqjvd7)