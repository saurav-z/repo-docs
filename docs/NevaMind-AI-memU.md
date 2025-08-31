<div align="center">

![MemU Banner](assets/banner.png)

</div>

# MemU: The Premier Memory Framework for AI Companions

**Build AI companions that remember everything with MemU, the open-source memory framework designed for high accuracy, fast retrieval, and low cost.  [Explore the MemU Repository on GitHub!](https://github.com/NevaMind-AI/memU)**

[![PyPI version](https://badge.fury.io/py/memu-py.svg)](https://badge.fury.io/py/memu-py)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/memu)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?logo=x&logoColor=white)](https://x.com/memU_ai)

---

## Key Features of MemU

*   ✅ **AI Companion Specialization:** Optimized for AI companion applications.
*   ✅ **Industry-Leading Accuracy:** Achieves a state-of-the-art 92% accuracy in the Locomo benchmark.
*   ✅ **Cost-Effective Solutions:** Up to 90% cost reduction through optimized online platform usage.
*   ✅ **Advanced Retrieval Strategies:** Includes semantic search, hybrid search, and contextual retrieval.
*   ✅ **24/7 Support (for Enterprise Customers):** Dedicated support for enterprise-level implementations.

---

## Why Choose MemU?

MemU empowers you to create AI companions that truly understand and remember their users.  Unlike traditional memory systems, MemU offers:

*   **Superior Accuracy:**  Benefit from higher precision in recalling information, leading to more relevant and engaging AI interactions.
*   **Blazing Fast Retrieval:** Retrieve memories quickly and efficiently, ensuring smooth and responsive AI companion experiences.
*   **Cost Optimization:** Reduce operational expenses with our efficient memory management and retrieval system.
*   **Adaptability:** MemU adapts to various AI companion scenarios, ensuring it's perfect for your project.

---

## Get Started with MemU

### ☁️ Cloud Version ([Online Platform](https://app.memu.so))

The quickest way to integrate AI memories. Ideal for teams and individuals wanting fast access without the setup hassle. We manage the models, APIs, and cloud storage, giving your application the best quality AI memory.

*   **Instant Access:**  Start integrating AI memories in minutes.
*   **Managed Infrastructure:** We handle scaling, updates, and maintenance for optimal memory quality.
*   **Premium Support:**  Get priority assistance from our engineering team (subscription required).

**Steps:**

1.  **Create an Account:**  Sign up at [https://app.memu.so](https://app.memu.so).
2.  **Generate API Keys:** Go to [https://app.memu.so/api-key/](https://app.memu.so/api-key/) and generate your API keys.
3.  **Install the Package:**

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
5.  **API Reference:**  See the [API reference](docs/API_REFERENCE.md) or [our blog](https://memu.pro/blog) for details.
6.  **Complete Integration Example:**  Check out [`example/client/memory.py`](example/client/memory.py)

✨ **That's it!**  MemU remembers everything, allowing your AI to learn from previous conversations.

### 🏢 Enterprise Edition

For organizations requiring maximum security, customization, and control.

*   **Commercial License:** Full proprietary features and commercial usage rights.
*   **Custom Development:** SSO/RBAC integration, dedicated algorithm team.
*   **Intelligence & Analytics:** User behavior analysis, real-time production monitoring.
*   **Premium Support:** 24/7 dedicated support, custom SLAs, professional implementation services.

📧 **Enterprise Inquiries:** [contact@nevamind.ai](mailto:contact@nevamind.ai)

### 🏠 Self-Hosting (Community Edition)

For users and developers who prefer local control, data privacy, and customization.

*   **Data Privacy:** Keep sensitive data within your infrastructure.
*   **Customization:** Modify and extend the platform to fit your needs.
*   **Cost Control:** Avoid recurring cloud fees for large-scale deployments.

See the [self-hosting README](README.self_host.md) for more information.

---

## ✨ MemU's Core Memory Architecture

MemU employs an innovative memory architecture designed for AI companions.  Here's a breakdown of its key components:

### Memory as File System

*   **Organize: Autonomous Memory File Management** Memories are managed as intelligent folders by a memory agent, with the agent automatically deciding what to record, modify, or archive.
*   **Link: Interconnected Knowledge Graph** Creates connections between related memories for effortless recall.
*   **Evolve: Continuous Self-Improvement** Generates insights by analyzing existing memories and identifying patterns.
*   **Never Forget: Adaptive Forgetting Mechanism** Prioritizes information based on usage, creating a personalized information hierarchy that evolves with user needs.

---

## 😺 Advantages of Using MemU

*   **Higher Memory Accuracy:**  Achieves a 92.09% average accuracy in the Locomo dataset, outperforming competitors. (Technical Report coming soon!)
    ![Memory Accuracy Comparison](assets/benchmark.png)
    *(1) Single-hop questions require answers based on a single session; (2) Multi-hop questions require synthesizing information from multiple different sessions; (3) Temporal reasoning questions can be answered through temporal reasoning and capturing time-related data cues within the conversation; (4) Open-domain knowledge questions can be answered by integrating a speaker’s provided information with external knowledge such as commonsense or world facts;*
*   **Fast Retrieval:**  Efficiently retrieves relevant document content.
*   **Low Cost:**  Processes multiple conversation turns at once, saving tokens and reducing costs.

---

## 🎓 Use Cases

MemU is versatile and can be applied to a variety of applications:

| AI Companion | AI Role Play | AI IP Characters | AI Education |
| :----------: | :----------: | :--------------: | :----------: |
| <img src="assets/usecase/ai_companion-0000.jpg" width="150" height="200"> | <img src="assets/usecase/ai_role_play-0000.jpg" width="150" height="200"> | <img src="assets/usecase/ai_ip-0000.png" width="150" height="200"> | <img src="assets/usecase/ai_edu-0000.jpg" width="150" height="200"> |
| AI Therapy  | AI Robot | AI Creation | More... |
| <img src="assets/usecase/ai_therapy-0000.jpg" width="150" height="200"> | <img src="assets/usecase/ai_robot-0000.jpg" width="150" height="200"> | <img src="assets/usecase/ai_creation-0000.jpg" width="150" height="200"> | |

---

## 🤝 Contribute to MemU

Help build MemU by contributing to the project! Your contributions are highly valued.

*   📋 **[Read our detailed Contributing Guide →](CONTRIBUTING.md)**

### **📄 License**

All contributions are licensed under the **Apache License 2.0**.

---

## 🌍 Join the MemU Community

*   **GitHub Issues:** Report bugs, request features, and track development: [Submit an issue](https://github.com/NevaMind-AI/memU/issues)
*   **Discord:** Get real-time support, chat with the community, and stay updated: [Join us](https://discord.com/invite/hQZntfGsbJ)
*   **X (Twitter):** Follow for updates, AI insights, and key announcements: [Follow us](https://x.com/memU_ai)

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

Stay updated with the latest news, discussions, and exclusive content by joining our WeChat community:

<div align="center">
<img src="assets/qrcode.png" alt="MemU WeChat and discord QR Code" width="480" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px;">

*Scan the QR code above to join our WeChat community*

</div>

---

*Stay connected with the MemU community! Join our WeChat groups for real-time discussions, technical support, and networking opportunities.*

## Questionnaire

Help us improve MemU by providing feedback! Complete our 3-minute survey for a chance to receive 30 free quota: [https://forms.gle/H2ZuZVHv72xbqjvd7](https://forms.gle/H2ZuZVHv72xbqjvd7)
```

Key improvements and explanations:

*   **SEO-Friendly Headline:**  Strong keywords ("AI Companion," "Memory Framework") in the main headline and throughout.
*   **Concise Hook:** The one-sentence hook immediately grabs attention and highlights the core benefit.
*   **Clear Headings:**  Uses appropriate headings (H2, H3) to structure the content for readability and SEO.
*   **Bulleted Lists:**  Uses bullet points for key features and advantages, making information easy to scan.
*   **Keyword Optimization:** Naturally integrates relevant keywords throughout the content (e.g., "open-source," "AI companions," "high accuracy," "fast retrieval," "cost-effective").
*   **Strong Calls to Action:**  Encourages users to "Explore the MemU Repository on GitHub!"
*   **Improved Structure:**  Organizes information logically, making it easier for users to find what they need.
*   **Concise Language:** Uses clear and concise language to communicate information effectively.
*   **Reordered and Condensed:**  Removed some redundant content, reorganized sections for a better flow.  Removed unnecessary marketing fluff.
*   **Emphasis on Benefits:**  Focuses on *why* users should choose MemU, not just *what* it is.
*   **Direct Links:**  Includes direct links to key resources (GitHub, Discord, API docs, etc.).
*   **Clearer "Getting Started" Section:** Improved instructions.
*   **Removed Redundancy:** Eliminated duplicate information.
*   **Updated the "Why Choose MemU?" section** Made the advantages and its value proposition more explicit.
*   **Formatting:** Applied the formatting from the original (banners, images, etc).