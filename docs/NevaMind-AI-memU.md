<div align="center">

![MemU Banner](assets/banner.png)

</div>

# MemU: The Ultimate Memory Framework for Intelligent AI Companions

**Build AI companions that truly remember with MemU, the open-source memory framework for high accuracy, fast retrieval, and cost-effective AI.** ([Original Repo](https://github.com/NevaMind-AI/memU))

[![PyPI version](https://badge.fury.io/py/memu-py.svg)](https://badge.fury.io/py/memu-py)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/memu)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?logo=x&logoColor=white)](https://x.com/memU_ai)

---

## Key Features of MemU

MemU provides a comprehensive memory solution for AI companions, offering superior performance and cost efficiency:

*   ✅ **AI Companion Specialization:** Specifically designed for AI companion applications.
*   ✅ **92% Accuracy:** Achieves state-of-the-art accuracy on the Locomo benchmark.
*   ✅ **Up to 90% Cost Reduction:** Optimized for cost-effective operation on online platforms.
*   ✅ **Advanced Retrieval Strategies:** Employs multiple methods including semantic search, hybrid search, and contextual retrieval for optimal results.
*   ✅ **24/7 Support (Enterprise):** Dedicated support for enterprise customers.
*   **Autonomous Memory File Management**: Organized intelligently and managed automatically by the memory agent.
*   **Interconnected Knowledge Graph**: Creates connections between memories to build a network of information.
*   **Continuous Self-Improvement**: Learns and evolves to create new insights by analyzing existing memories.
*   **Adaptive Forgetting Mechanism**: Prioritizes information based on usage patterns.

---

## Why Choose MemU?

*   **Superior Accuracy:** MemU excels with a 92.09% average accuracy on the Locomo dataset.
*   **Fast Retrieval:** Efficiently retrieves relevant information by focusing on document content, avoiding extensive embedding searches.
*   **Cost-Effective:** Designed to reduce operational costs, processing large amounts of data efficiently.

---

## Get Started with MemU

Choose the deployment option that fits your needs:

### ☁️ Cloud Version ([Online Platform](https://app.memu.so))

Quickly integrate AI memories with our cloud platform.

*   **Instant Access:** Integrate AI memories in minutes.
*   **Managed Infrastructure:** We handle scaling, updates, and maintenance.
*   **Premium Support:** Get priority assistance from our engineering team.

**Getting Started:**

1.  **Create an Account:** Sign up at [https://app.memu.so](https://app.memu.so).
2.  **Generate API Keys:** Go to [https://app.memu.so/api-key/](https://app.memu.so/api-key/) to get your API keys.
3.  **Integrate with Code:**

    ```python
    pip install memu-py

    # Example Usage
    from memu import MemuClient
    import os

    # Initialize
    memu_client = MemuClient(
        base_url="https://api.memu.so",
        api_key=os.getenv("MEMU_API_KEY") # Set your API key
    )

    conversation_text = "Your conversation text here" # Replace with actual conversation text

    memu_client.memorize_conversation(
        conversation=conversation_text, # Recommend longer conversation (~8000 tokens), see https://memu.pro/blog/memu-best-practice for details
        user_id="user001",
        user_name="User",
        agent_id="assistant001",
        agent_name="Assistant"
    )
    ```

    **Note:**  Check the [API reference](docs/API_REFERENCE.md) or [our blog](https://memu.pro/blog) for more details.

4.  **Complete Integration:** See [`example/client/memory.py`](example/client/memory.py) for detailed integration examples.

### 🏢 Enterprise Edition

For organizations requiring enhanced security, customization, and control:

*   **Commercial License:** Full proprietary features and commercial usage rights.
*   **Custom Development:** Includes SSO/RBAC integration and scenario-specific framework optimization.
*   **Intelligence & Analytics:** Provides user behavior analysis, real-time monitoring, and automated agent optimization.
*   **Premium Support:** Offers 24/7 dedicated support and professional implementation services.

**Contact:** [contact@nevamind.ai](mailto:contact@nevamind.ai)

### 🏠 Self-Hosting (Community Edition)

For users who prefer local control and data privacy:

*   **Data Privacy:** Keep sensitive data within your infrastructure.
*   **Customization:** Modify and extend the platform to fit your needs.
*   **Cost Control:** Avoid cloud fees for large-scale deployments.

See [self hosting README](README.self_host.md)

---

##  🚀 Demo Video

Watch MemU in action:

<div align="left">
  <a href="https://www.youtube.com/watch?v=qZIuCoLglHs">
    <img src="https://img.youtube.com/vi/ueOe4ZPlZLU/maxresdefault.jpg" alt="MemU Demo Video" width="600">
  </a>
  <br>
  <em>Click to watch the MemU demonstration video</em>
</div>

---

##  ✨ Advantages in Detail

### Higher Memory Accuracy

MemU's superior accuracy is demonstrated with a 92.09% average score on the Locomo dataset across all reasoning tasks, significantly outperforming competitors.  [Technical Report will be published soon!].

![Memory Accuracy Comparison](assets/benchmark.png)

### Fast Retrieval

We categorize important information into documents, and during retrieval, we only need to find the relevant document content, eliminating the need for extensive embedding searches for fragmented sentences.

### Low cost

We can process hundreds of conversation turns at once, eliminating the need for developers to repeatedly call memory functions, thus saving users from wasting tokens on multiple memory operations. See [best practice](https://memu.pro/blog/memu-best-practice).

---

## 🎓 Use Cases

MemU enhances various AI applications:

| AI Companion | AI Role Play | AI IP Characters | AI Education |
| :---: | :---: | :---: | :---: |
| <img src="assets/usecase/ai_companion-0000.jpg" width="150" height="200"> | <img src="assets/usecase/ai_role_play-0000.jpg" width="150" height="200"> | <img src="assets/usecase/ai_ip-0000.png" width="150" height="200"> | <img src="assets/usecase/ai_edu-0000.jpg" width="150" height="200"> |
| AI Therapy | AI Robot | AI Creation | More... |
| <img src="assets/usecase/ai_therapy-0000.jpg" width="150" height="200"> | <img src="assets/usecase/ai_robot-0000.jpg" width="150" height="200"> | <img src="assets/usecase/ai_creation-0000.jpg" width="150" height="200"> |  |

---

## 🤝 Contribute to MemU

Join our open-source community and help shape the future of MemU!

📋 **[Read our detailed Contributing Guide →](CONTRIBUTING.md)**

### **📄 License**

All contributions are licensed under the **Apache License 2.0**.

---

## 🌍 Community & Support

*   **GitHub Issues:** Report bugs, request features: [Submit an issue](https://github.com/NevaMind-AI/memU/issues)
*   **Discord:** Get real-time support and connect with the community: [Join us](https://discord.com/invite/hQZntfGsbJ)
*   **X (Twitter):** Follow for updates and announcements: [Follow us](https://x.com/memU_ai)
*   **General Inquiries:**  info@nevamind.ai

---

## 🤝 Ecosystem

We are proud to partner with:

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

Stay updated and connect with the MemU community on WeChat:

<div align="center">
<img src="assets/qrcode.png" alt="MemU WeChat and discord QR Code" width="480" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px;">
</div>

---

## Questionnaire

Help us improve MemU! Share your feedback and get a reward: https://forms.gle/H2ZuZVHv72xbqjvd7