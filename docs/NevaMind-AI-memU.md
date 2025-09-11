<div align="center">

![MemU Banner](assets/banner.png)
</div>

# MemU: The Ultimate AI Memory Framework for Lifelike Companions 🧠

**Empower your AI with unparalleled memory capabilities!** MemU is an open-source memory framework designed to give AI companions the ability to truly remember and learn from their interactions, offering high accuracy, fast retrieval, and cost-effectiveness. [Explore MemU on GitHub!](https://github.com/NevaMind-AI/memU)

[![PyPI version](https://badge.fury.io/py/memu-py.svg)](https://badge.fury.io/py/memu-py)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/memu)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?logo=x&logoColor=white)](https://x.com/memU_ai)

## Key Features of MemU

*   ✅ **AI Companion Specialization:** Tailored for AI companion applications.
*   ✅ **Unmatched Accuracy:** Achieve state-of-the-art performance, with 92% accuracy in the Locomo benchmark.
*   ✅ **Cost-Effective:** Reduce expenses by up to 90% with our optimized online platform.
*   ✅ **Advanced Retrieval Strategies:** Leverage multiple methods, including semantic, hybrid, and contextual search.
*   ✅ **Comprehensive Support:** 24/7 support for enterprise customers.

## Why Choose MemU?

MemU provides a superior memory solution, enabling AI companions to:

*   **Personalize Interactions:** Remember users, their interests, and evolving preferences.
*   **Learn and Grow:** Develop alongside users, building a rich understanding through every interaction.
*   **Enhance Engagement:** Create more meaningful and memorable experiences.

## Get Started

MemU offers flexible integration options to suit your needs.

### ☁️ **Cloud Version (Online Platform)**

The quickest way to integrate MemU into your application. Ideal for teams and individuals seeking immediate access without setup complexities.

*   **Instant Access:** Begin integrating AI memories in minutes.
*   **Managed Infrastructure:** We handle scaling, updates, and maintenance for optimal memory quality.
*   **Premium Support:** Get priority assistance from our engineering team.

**How to Get Started with Cloud Version**

1.  **Create an Account:** Sign up at [https://app.memu.so](https://app.memu.so).
2.  **Generate API Keys:** Obtain your API keys from [https://app.memu.so/api-key/](https://app.memu.so/api-key/).
3.  **Install the Package:**
    ```bash
    pip install memu-py
    ```
4.  **Implement in your Code:**
    ```python
    # Example usage
    from memu import MemuClient
    import os  # Import the os module

    # Initialize
    memu_client = MemuClient(
        base_url="https://api.memu.so",
        api_key=os.getenv("MEMU_API_KEY")
    )

    conversation_text = "User: Hi, what is your name? Assistant: I am an AI companion. User: That's cool."  # Example conversation
    memu_client.memorize_conversation(
        conversation=conversation_text,  # Recommend longer conversation (~8000 tokens), see https://memu.pro/blog/memu-best-practice for details
        user_id="user001",
        user_name="User",
        agent_id="assistant001",
        agent_name="Assistant"
    )
    ```

**For Complete Integration Details:**

*   📖 See [`example/client/memory.py`](example/client/memory.py) for complete integration details
*   Check [API reference](docs/API_REFERENCE.md) or [our blog](https://memu.pro/blog) for more details.

✨ **That's it!** MemU will remember everything and help your AI learn from past conversations.

### 🏢 **Enterprise Edition**

Designed for organizations requiring maximum security, customization, and control:

*   **Commercial License:** Access full proprietary features and white-labeling.
*   **Custom Development:** Leverage SSO/RBAC integration and dedicated algorithm teams.
*   **Intelligence & Analytics:** Benefit from user behavior analysis and real-time monitoring.
*   **Premium Support:** Enjoy 24/7 dedicated support and custom SLAs.

📧 **Enterprise Inquiries:** [contact@nevamind.ai](mailto:contact@nevamind.ai)

### 🏠 **Self-Hosting (Community Edition)**

For users who prefer local control, data privacy, and customization:

*   **Data Privacy:** Keep sensitive data within your infrastructure.
*   **Customization:** Modify and extend the platform to fit your needs.
*   **Cost Control:** Avoid recurring cloud fees for large-scale deployments.

See [self hosting README](README.self_host.md)

## ✨ Core MemU Technology: Memory as a File System

MemU's innovative approach to memory management enables unparalleled capabilities:

### **Organize: Autonomous Memory File Management**

*   Your memories are structured as intelligent folders managed by a memory agent. The memory agent automatically decides what to record, modify, or archive.

### **Link: Interconnected Knowledge Graph**

*   Memories automatically connect, building a network of hyperlinked documents for effortless recall.

### **Evolve: Continuous Self-Improvement**

*   Your memory agent generates insights, identifies patterns, and creates summaries to improve over time.

### **Never Forget: Adaptive Forgetting Mechanism**

*   The memory agent prioritizes information based on usage patterns for a personalized information hierarchy.

## 😺 Advantages

*   **Higher Memory Accuracy:** Achieve 92.09% average accuracy in Locomo dataset, surpassing competitors. [Technical Report will be published soon!]
    ![Memory Accuracy Comparison](assets/benchmark.png)
    *   (1) Single-hop questions; (2) Multi-hop questions; (3) Temporal reasoning questions; (4) Open-domain knowledge questions

*   **Fast Retrieval:** Quickly find relevant document content without needing extensive embedding searches.
*   **Low Cost:** Process hundreds of conversation turns at once.

## 🎓 Use Cases

MemU is perfect for a wide range of applications:

| | | | |
|:---:|:---:|:---:|:---:|
| <img src="assets/usecase/ai_companion-0000.jpg" width="150" height="200"><br>**AI Companion** | <img src="assets/usecase/ai_role_play-0000.jpg" width="150" height="200"><br>**AI Role Play** | <img src="assets/usecase/ai_ip-0000.png" width="150" height="200"><br>**AI IP Characters** | <img src="assets/usecase/ai_edu-0000.jpg" width="150" height="200"><br>**AI Education** |
| <img src="assets/usecase/ai_therapy-0000.jpg" width="150" height="200"><br>**AI Therapy** | <img src="assets/usecase/ai_robot-0000.jpg" width="150" height="200"><br>**AI Robot** | <img src="assets/usecase/ai_creation-0000.jpg" width="150" height="200"><br>**AI Creation** | More...|

## 🤝 Contribute to MemU

We welcome contributions to the MemU project!

*   📋 **[Read our detailed Contributing Guide →](CONTRIBUTING.md)**

### **📄 License**

All contributions are licensed under the **Apache License 2.0**.

## 🌍 Community

*   **GitHub Issues:** Report bugs, request features, and track development. [Submit an issue](https://github.com/NevaMind-AI/memU/issues)
*   **Discord:** Get real-time support and chat with the community. [Join us](https://discord.com/invite/hQZntfGsbJ)
*   **X (Twitter):** Follow for updates and announcements. [Follow us](https://x.com/memU_ai)

## 🤝 Ecosystem

We're proud to work with these development tools & partners:

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

Stay connected with the MemU community by joining our WeChat groups:

<div align="center">
<img src="assets/qrcode.png" alt="MemU WeChat and discord QR Code" width="480" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px;">
</div>

---

*Scan the QR code above to join our WeChat community.*

## Questionnaire

Help us improve! Share your feedback on our 3-min survey and get 30 free quota：https://forms.gle/H2ZuZVHv72xbqjvd7
```

Key improvements and SEO optimizations:

*   **Clear Hook:**  Immediately states the core benefit.
*   **Targeted Keywords:**  Includes terms like "AI memory framework", "AI companions", "open-source", and related phrases throughout the document.
*   **Structured Headings:** Organizes the information for readability and SEO ranking.
*   **Bulleted Lists:**  Highlights key features and benefits.
*   **Concise Language:**  Uses direct language to convey value.
*   **Calls to Action:** Encourages users to explore, contribute, and connect with the community.
*   **Internal and External Linking:** Links to relevant resources within the document and the GitHub repository.
*   **Use Case Emphasis:**  Highlights the various applications of MemU with visual cues.
*   **Community Focus:**  Promotes community engagement.
*   **Mobile-Friendly:**  Uses proper image dimensions.
*   **Alt Text:** Provides descriptive alt text for images.
*   **Contact Info:** Added contact information for partnerships.
*   **Clear "Getting Started" Section:** Made the setup steps easier to follow.
*   **Included partner logos**