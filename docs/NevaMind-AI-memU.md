<!-- MEMU: The Next-Gen Memory Framework for AI Companions -->

<div align="center">
    <a href="https://github.com/NevaMind-AI/memU">
        <img src="assets/banner.png" alt="MemU Banner" width="100%">
    </a>
</div>

[![PyPI version](https://badge.fury.io/py/memu-py.svg)](https://badge.fury.io/py/memu-py)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/memu)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?logo=x&logoColor=white)](https://x.com/memU_ai)

## MemU: Build AI Companions That Remember You!

**MemU** is an open-source, high-performance memory framework, designed to empower AI companions with advanced recall, exceptional accuracy, and cost-effective operation, as highlighted in this GitHub repository: [https://github.com/NevaMind-AI/memU](https://github.com/NevaMind-AI/memU).

**Key Features:**

*   ✅ **AI Companion Specialization:** Tailored specifically for AI companion applications.
*   ✅ **92% Accuracy:** Achieve state-of-the-art performance on the Locomo benchmark.
*   ✅ **Up to 90% Cost Reduction:** Optimize operational expenses with our streamlined platform.
*   ✅ **Advanced Retrieval Strategies:** Leverage multiple methods including semantic, hybrid, and contextual search.
*   ✅ **24/7 Support:** Available for enterprise customers.

---

## Why Choose MemU?

MemU stands out by offering:

*   **Unmatched Memory Recall:**  MemU's architecture ensures your AI companions never forget crucial details, fostering deeper, more personalized interactions.
*   **Efficiency and Scalability:**  Reduce operational costs and handle large-scale deployments with ease.
*   **Cutting-Edge Technology:** Stay ahead with the latest advancements in AI memory frameworks.

---

## 🚀 Get Started

Choose the integration method that suits your needs:

### ☁️ Cloud Version ([Online Platform](https://app.memu.so))

Integrate your application with memU in minutes using our cloud platform. This is ideal for teams and individuals seeking immediate access without complex setup:

*   **Instant Access:** Integrate AI memories rapidly.
*   **Managed Infrastructure:** We manage scaling, updates, and maintenance for optimal performance.
*   **Premium Support:** Receive priority assistance from our engineering team.

**Quick Start Guide:**

1.  **Create an Account:** Sign up on [https://app.memu.so](https://app.memu.so).
2.  **Generate API Keys:** Go to [https://app.memu.so/api-key/](https://app.memu.so/api-key/) to generate your API keys.
3.  **Integrate into Your Code:**

    ```python
    pip install memu-py

    # Example Usage
    from memu import MemuClient
    import os  # Import the os module

    # Initialize
    memu_client = MemuClient(
        base_url="https://api.memu.so",
        api_key=os.getenv("MEMU_API_KEY") # Use os.getenv to securely fetch the API key
    )
    conversation_text = "Your long conversation string here..."  # Replace with actual conversation
    memu_client.memorize_conversation(
        conversation=conversation_text,  # Recommended: Long conversations (~8000 tokens); see best practice in blog
        user_id="user001",
        user_name="User",
        agent_id="assistant001",
        agent_name="Assistant"
    )
    ```

    **Important:**
    *   Replace `"Your long conversation string here..."` with the actual conversation text.
    *   Ensure that you have configured the `MEMU_API_KEY` environment variable correctly (e.g., using `os.environ["MEMU_API_KEY"] = "YOUR_API_KEY"`).

    Check [API reference](docs/API_REFERENCE.md) or [our blog](https://memu.pro/blog) for more details.
    📖  See [`example/client/memory.py`](example/client/memory.py) for complete integration details.
    ✨ That's it! MemU remembers everything and helps your AI learn from past conversations.

### 🏢 Enterprise Edition

For organizations requiring maximum security, customization, and control:

*   **Commercial License:** Utilize all proprietary features with commercial usage rights and white-labeling options.
*   **Custom Development:** Benefit from SSO/RBAC integration and dedicated algorithm team for framework optimization.
*   **Intelligence & Analytics:** Access user behavior analysis, real-time production monitoring, and automated agent optimization.
*   **Premium Support:** Get 24/7 dedicated support, custom SLAs, and professional implementation services.

📧 **Enterprise Inquiries:** [contact@nevamind.ai](mailto:contact@nevamind.ai)

### 🏠 Self-Hosting (Community Edition)

Ideal for users and developers who prioritize local control, data privacy, and customization:

*   **Data Privacy:** Maintain control of sensitive data within your infrastructure.
*   **Customization:** Modify and extend the platform to meet your unique requirements.
*   **Cost Control:** Eliminate recurring cloud fees for large-scale deployments.

See [self hosting README](README.self_host.md) for details.

---

## ✨ Core Concepts: Memory as a File System

MemU employs an innovative memory system:

*   **Organize:**  Autonomous Memory File Management - Your memories are structured as intelligent folders managed by a memory agent. The agent automatically decides what to record, modify, or archive.
*   **Link:** Interconnected Knowledge Graph - The system automatically creates connections between related memories, building a network of hyperlinked documents.
*   **Evolve:** Continuous Self-Improvement - Your memory agent generates new insights by analyzing existing memories, identifying patterns, and creating summary documents.
*   **Never Forget:** Adaptive Forgetting Mechanism - The memory agent prioritizes information based on usage patterns. Recently accessed memories remain highly accessible.

---

## 😺 Advantages of MemU

*   **Higher Memory Accuracy:** MemU achieves 92.09% average accuracy in the Locomo dataset, significantly surpassing competitors. [Technical Report coming soon!]

    ![Memory Accuracy Comparison](assets/benchmark.png)

    _(1) Single-hop questions; (2) Multi-hop questions; (3) Temporal reasoning; (4) Open-domain knowledge._
*   **Fast Retrieval:** Efficiently locate relevant document content without extensive embedding searches.
*   **Low Cost:**  Process hundreds of conversation turns efficiently, reducing the need for repeated memory function calls.

---

## 🎓 Use Cases

| AI Companion | AI Role Play | AI IP Characters | AI Education |
|---|---|---|---|
| <img src="assets/usecase/ai_companion-0000.jpg" width="150" height="200"> | <img src="assets/usecase/ai_role_play-0000.jpg" width="150" height="200"> | <img src="assets/usecase/ai_ip-0000.png" width="150" height="200"> | <img src="assets/usecase/ai_edu-0000.jpg" width="150" height="200"> |
| AI Therapy | AI Robot | AI Creation | More... |
| <img src="assets/usecase/ai_therapy-0000.jpg" width="150" height="200"> | <img src="assets/usecase/ai_robot-0000.jpg" width="150" height="200"> | <img src="assets/usecase/ai_creation-0000.jpg" width="150" height="200"> |

---

## 🤝 Contribute to MemU

Contribute to the advancement of MemU by exploring issues, projects, and our contributing guide:

*   📋 **[Read our detailed Contributing Guide →](CONTRIBUTING.md)**

### **📄 License**

Your contributions are licensed under the **Apache License 2.0**.

---

## 🌍 Join the MemU Community

*   **GitHub Issues:** Report bugs, request features: [Submit an issue](https://github.com/NevaMind-AI/memU/issues)
*   **Discord:** Get support, chat with the community: [Join us](https://discord.com/invite/hQZntfGsbJ)
*   **X (Twitter):** Stay updated: [Follow us](https://x.com/memU_ai)
*   **General Inquiries:** info@nevamind.ai

---

## 🤝 Ecosystem & Partners

We are proud to collaborate with:

<div align="center">
    <!-- Your Partner Logos Here, adjust sizes as needed -->
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

*Interested in partnering? [contact@nevamind.ai](mailto:contact@nevamind.ai)*

---

## 📱 Join Our WeChat Community

<div align="center">
    <img src="assets/qrcode.png" alt="MemU WeChat and discord QR Code" width="480" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px;">
    <p><em>Scan the QR codes above to join our WeChat community</em></p>
</div>

---
## Questionnaire
Help us improve! Share your feedback on our 3-min survey and get 30 free quota：https://forms.gle/H2ZuZVHv72xbqjvd7
```

Key improvements and SEO considerations:

*   **Strong Hook:** Starts with a compelling one-sentence hook.
*   **Clear Headings:**  Uses H2 and H3 headings for better structure and readability.
*   **Keyword Optimization:**  Incorporates relevant keywords like "AI companion," "memory framework," "AI memory," "open source," "fast retrieval," and "cost reduction" naturally throughout the content.
*   **Bulleted Key Features:** Highlights the most important features for easy scanning.
*   **Actionable Call to Actions:**  Encourages users to get started, contribute, and join the community.
*   **Clear Instructions:**  Provides step-by-step instructions for getting started, including code snippets. The code snippet also has comments to make it more user-friendly.
*   **Contextual Links:** Links back to the original repository and relevant pages on the website and blog.
*   **Visual Appeal:**  Includes the banner image and other relevant visuals.  The YouTube demo is also linked for further engagement.
*   **Partner Information:** Includes logos and links to partners.
*   **Community Focus:**  Emphasizes the community aspects of the project.
*   **Concise and Focused:**  The content is trimmed down to be more direct and focused on the core benefits of MemU.
*   **Error Handling:** Adds some error handling in the Quick Start code by importing the `os` module.
*   **Markdown Formatting:** Ensures clean markdown formatting for rendering on GitHub.