<div align="center">
  <a href="https://github.com/NevaMind-AI/memU">
    <img src="assets/banner.png" alt="MemU Banner" width="100%">
  </a>
</div>

# MemU: The Ultimate Memory Framework for Intelligent AI Companions

**MemU empowers you to build AI companions that remember everything, offering superior accuracy, speed, and cost-effectiveness.** Explore the [MemU GitHub Repository](https://github.com/NevaMind-AI/memU) to get started!

[![PyPI version](https://badge.fury.io/py/memu-py.svg)](https://badge.fury.io/py/memu-py)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/memu)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?logo=x&logoColor=white)](https://x.com/memU_ai)

---

## Key Features & Benefits

*   **AI Companion Specialization:** Built specifically for AI companion applications.
*   **Exceptional Accuracy:** Achieves a state-of-the-art 92% accuracy score in the Locomo benchmark.
*   **Cost-Effective:** Up to 90% cost reduction through optimized online platform.
*   **Advanced Retrieval Strategies:** Employs semantic, hybrid, and contextual search methods.
*   **24/7 Support:** Dedicated support for enterprise customers.

---

## Why Choose MemU?

MemU stands out as the next-generation memory framework, offering a powerful and efficient solution for building AI companions. Key advantages include:

*   **Superior Memory Accuracy:**  MemU's advanced architecture provides significantly higher accuracy in retrieving and utilizing information compared to other memory solutions.
*   **Blazing Fast Retrieval:** MemU's optimized retrieval processes allow for quicker access to relevant information, enhancing the responsiveness of your AI companion.
*   **Reduced Costs:**  Through efficient design and optimized platform utilization, MemU helps to minimize operational expenses, providing a cost-effective solution.

---

## Getting Started

MemU offers several ways to integrate and utilize its capabilities:

### ☁️ **Cloud Version (Online Platform)**

The fastest way to integrate with memU, ideal for teams and individuals.

*   **Instant Integration:** Integrate AI memories in minutes.
*   **Managed Infrastructure:** We handle scaling, updates, and maintenance.
*   **Premium Support:** Priority assistance available.

**Quick Start Guide:**

1.  **Create an Account:** Visit [https://app.memu.so](https://app.memu.so).
2.  **Generate API Keys:** Go to [https://app.memu.so/api-key/](https://app.memu.so/api-key/) to obtain your API keys.
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
5.  **API Reference:** Consult the [API Reference](docs/API_REFERENCE.md) and [blog](https://memu.pro/blog) for more details.
6.  **Complete Integration:**  See [`example/client/memory.py`](example/client/memory.py) for a detailed example.

### 🏢 **Enterprise Edition**

For organizations requiring advanced features and support:

*   Commercial License: Full proprietary features and usage rights.
*   Custom Development: SSO/RBAC integration and scenario-specific optimization.
*   Intelligence & Analytics: User behavior analysis and automated agent optimization.
*   Premium Support: 24/7 dedicated support and custom SLAs.

**Contact:** [contact@nevamind.ai](mailto:contact@nevamind.ai)

### 🏠 **Self-Hosting (Community Edition)**

For users preferring local control and data privacy:

*   **Data Privacy:** Keep your data within your own infrastructure.
*   **Customization:** Modify and extend the platform to fit your needs.
*   **Cost Control:** Avoid recurring cloud fees.

See [self hosting README](README.self_host.md)

---

## ✨ **Memory System Overview**

MemU employs an innovative approach to memory management, offering powerful features:

### **Organize**

*   Autonomous Memory File Management: Memories are organized as intelligent folders by the memory agent, ensuring effective structure and management.

### **Link**

*   Interconnected Knowledge Graph: The system automatically creates connections between related memories, forming a rich network of interconnected knowledge.

### **Evolve**

*   Continuous Self-Improvement: The memory agent analyzes existing memories to generate insights and improve knowledge over time.

### **Never Forget**

*   Adaptive Forgetting Mechanism: Prioritizes information based on usage patterns, personalizing the information hierarchy to meet evolving needs.

---

## 🎥 **Demo Video**

<div align="center">
  <a href="https://www.youtube.com/watch?v=qZIuCoLglHs">
    <img src="https://img.youtube.com/vi/ueOe4ZPlZLU/maxresdefault.jpg" alt="MemU Demo Video" width="600">
  </a>
  <br>
  <em>Click to watch the MemU demonstration video</em>
</div>

---

## 🎓 **Use Cases**

MemU empowers a variety of applications, including:

| | | | |
|:---:|:---:|:---:|:---:|
| <img src="assets/usecase/ai_companion-0000.jpg" width="150" height="200"><br>**AI Companion** | <img src="assets/usecase/ai_role_play-0000.jpg" width="150" height="200"><br>**AI Role Play** | <img src="assets/usecase/ai_ip-0000.png" width="150" height="200"><br>**AI IP Characters** | <img src="assets/usecase/ai_edu-0000.jpg" width="150" height="200"><br>**AI Education** |
| <img src="assets/usecase/ai_therapy-0000.jpg" width="150" height="200"><br>**AI Therapy** | <img src="assets/usecase/ai_robot-0000.jpg" width="150" height="200"><br>**AI Robot** | <img src="assets/usecase/ai_creation-0000.jpg" width="150" height="200"><br>**AI Creation** | More...|

---

## 🤝 Contributing

We welcome contributions from the community!  Explore our [GitHub Issues](https://github.com/NevaMind-AI/memU/issues) and [Projects](https://github.com/NevaMind-AI/memU/projects) to get involved.

*   **Contributing Guide:** [CONTRIBUTING.md](CONTRIBUTING.md)
*   **License:**  All contributions are licensed under the Apache License 2.0.

---

## 🌍 Community & Support

*   **GitHub Issues:** Report bugs, request features, and track development:  [Submit an issue](https://github.com/NevaMind-AI/memU/issues)
*   **Discord:** Get real-time support and chat with the community: [Join us](https://discord.com/invite/hQZntfGsbJ)
*   **X (Twitter):** Follow for updates and announcements: [Follow us](https://x.com/memU_ai)
*   **For more information please contact info@nevamind.ai**

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
<br>
*Scan any of the QR codes above to join our WeChat community*

</div>

---

## Questionnaire

Help us improve! Share your feedback on our 3-min survey and get 30 free quota：https://forms.gle/H2ZuZVHv72xbqjvd7