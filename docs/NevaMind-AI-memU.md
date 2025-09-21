<div align="center">

![MemU Banner](assets/banner.png)

</div>

# MemU: The Next-Generation Memory Framework for AI Companions

**Unlock the power of persistent memory for your AI companions with MemU, a cutting-edge open-source framework designed for accuracy, speed, and cost-effectiveness.** ([View on GitHub](https://github.com/NevaMind-AI/memU))

[![PyPI version](https://badge.fury.io/py/memu-py.svg)](https://badge.fury.io/py/memu-py)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/memu)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?logo=x&logoColor=white)](https://x.com/memU_ai)

---

## Key Features

*   ✅ **AI Companion Specialization:** Optimized for AI companion applications.
*   ✅ **92% Accuracy:** Achieves state-of-the-art performance in the Locomo benchmark.
*   ✅ **Up to 90% Cost Reduction:** Through an optimized online platform.
*   ✅ **Advanced Retrieval Strategies:** Includes semantic search, hybrid search, and contextual retrieval.
*   ✅ **24/7 Support:** Available for enterprise customers.
*   **Autonomous Memory File Management:** Organize memories as intelligent folders managed by a memory agent.
*   **Interconnected Knowledge Graph:** Automatically creates connections between related memories.
*   **Continuous Self-Improvement:**  The memory agent generates new insights by analyzing existing memories.
*   **Adaptive Forgetting Mechanism:** The memory agent automatically prioritizes information based on usage patterns.

---

## Why Choose MemU?

MemU empowers you to build AI companions that truly remember and learn from every interaction, fostering deeper connections and personalized experiences.

### Advantages:

*   **Higher Memory Accuracy:** Significantly outperforms competitors with 92.09% average accuracy in Locomo dataset.
*   **Fast Retrieval:** Efficiently retrieves relevant document content, avoiding costly embedding searches.
*   **Low Cost:** Processes hundreds of conversation turns at once, saving on token usage.

---

## 🚀 Get Started

MemU offers flexible deployment options to suit your needs:

### ☁️ Cloud Version ([Online Platform](https://app.memu.so))

Get up and running quickly with our cloud platform.

*   **Instant Access:** Integrate AI memories in minutes.
*   **Managed Infrastructure:** We handle scaling, updates, and maintenance.
*   **Premium Support:** Get priority assistance from our engineering team.

**Quick Start:**

1.  Create an account at [https://app.memu.so](https://app.memu.so) and generate an API key at [https://app.memu.so/api-key/](https://app.memu.so/api-key/).
2.  Install the Python package: `pip install memu-py`
3.  Use the following code example:

```python
# Example usage
import os
from memu import MemuClient

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

📖 See [`example/client/memory.py`](example/client/memory.py) for complete integration details.

### 🏢 Enterprise Edition

For organizations requiring maximum security, customization, and control.

*   **Commercial License:** Full proprietary features and usage rights.
*   **Custom Development:** SSO/RBAC integration and scenario-specific framework optimization.
*   **Intelligence & Analytics:** User behavior analysis and automated agent optimization.
*   **Premium Support:** 24/7 dedicated support and custom SLAs.

📧 **Enterprise Inquiries:** [contact@nevamind.ai](mailto:contact@nevamind.ai)

### 🏠 Self-Hosting (Community Edition)

For users who prefer local control and data privacy.

*   **Data Privacy:** Keep sensitive data within your infrastructure.
*   **Customization:** Modify and extend the platform to fit your needs.
*   **Cost Control:** Avoid recurring cloud fees for large-scale deployments.

See [self hosting README](README.self_host.md)

---

## ✨ Demo Video

[<img src="https://img.youtube.com/vi/ueOe4ZPlZLU/maxresdefault.jpg" alt="MemU Demo Video" width="600">](https://www.youtube.com/watch?v=qZIuCoLglHs)
*Click to watch the MemU demonstration video*

---

## 🎓 Use Cases

| | | | |
|:---:|:---:|:---:|:---:|
| <img src="assets/usecase/ai_companion-0000.jpg" width="150" height="200"><br>**AI Companion** | <img src="assets/usecase/ai_role_play-0000.jpg" width="150" height="200"><br>**AI Role Play** | <img src="assets/usecase/ai_ip-0000.png" width="150" height="200"><br>**AI IP Characters** | <img src="assets/usecase/ai_edu-0000.jpg" width="150" height="200"><br>**AI Education** |
| <img src="assets/usecase/ai_therapy-0000.jpg" width="150" height="200"><br>**AI Therapy** | <img src="assets/usecase/ai_robot-0000.jpg" width="150" height="200"><br>**AI Robot** | <img src="assets/usecase/ai_creation-0000.jpg" width="150" height="200"><br>**AI Creation** | More...|
---

## 🤝 Contributing

We welcome contributions! Explore our GitHub issues and projects to get involved.

📋 **[Read our detailed Contributing Guide →](CONTRIBUTING.md)**

### **📄 License**

Contributions are licensed under the **Apache License 2.0**.

---

## 🌍 Community

*   **GitHub Issues:** Report bugs, request features. [Submit an issue](https://github.com/NevaMind-AI/memU/issues)
*   **Discord:** Get real-time support and chat with the community. [Join us](https://discord.com/invite/hQZntfGsbJ)
*   **X (Twitter):** Stay updated on the latest news. [Follow us](https://x.com/memU_ai)

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

<div align="center">
<img src="assets/qrcode.png" alt="MemU WeChat and discord QR Code" width="480" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px;">
*Scan any of the QR codes above to join our WeChat community*
</div>

---

## Questionnaire

Help us improve! Share your feedback on our 3-min survey and get 30 free quota：https://forms.gle/H2ZuZVHv72xbqjvd7
```

Key improvements:

*   **SEO-Optimized Title:**  The title now includes the keyword "AI Companions" which is important for search visibility.
*   **Concise Hook:** The first sentence directly explains what MemU is and highlights a key benefit.
*   **Clear Headings and Formatting:**  Uses clear headings (H2, H3) and formatting (bold, bullet points) for readability and SEO.
*   **Keyword Integration:**  Strategically includes relevant keywords throughout the text (e.g., "AI companion," "memory framework," "open-source").
*   **Benefit-Driven Content:**  Focuses on the benefits of using MemU for the user (e.g., "Unlock the power," "deeper connections").
*   **Call to Actions:** Includes multiple clear CTAs (e.g., "Get Started," "Join us," "Submit an issue").
*   **Improved Summary of Key Features:** Uses bullet points to highlight core features for easy skimming.
*   **Organized Sections:**  The content is divided into logical sections to improve navigation and user experience.
*   **Simplified "Get Started" Section:** Made the quick start guide more concise and actionable.
*   **Community Focus:**  Highlights ways to connect with the MemU community.
*   **Partner Integration:** Includes logos of ecosystem partners.
*   **Removed Redundancy:** Streamlined and removed unnecessary or repetitive information.
*   **Direct Links to Documentation & Examples:** Included clear links to relevant resources.
*   **Focus on User Value:**  Highlights what the user gains by using MemU.