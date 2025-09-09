# MemU: The Next-Generation Memory Framework for AI Companions

[![PyPI version](https://badge.fury.io/py/memu-py.svg)](https://badge.fury.io/py/memu-py)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/memu)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?logo=x&logoColor=white)](https://x.com/memU_ai)

**Create AI companions that truly remember and understand with MemU, a cutting-edge, open-source memory framework.**  [Explore the original repository](https://github.com/NevaMind-AI/memU) for more details.

**Key Features:**

*   🧠 **AI Companion Specialization:** Designed specifically for AI companion applications.
*   🎯 **92% Accuracy:** Achieves state-of-the-art performance on the Locomo benchmark.
*   💰 **Up to 90% Cost Reduction:** Optimized for efficient and cost-effective memory management.
*   🔍 **Advanced Retrieval Strategies:** Leverages semantic, hybrid, and contextual search for optimal recall.
*   💬 **24/7 Enterprise Support:** Dedicated support for enterprise customers.

## Why Choose MemU?

MemU provides a robust and efficient solution for building AI companions. By offering high accuracy, fast retrieval, and cost-effectiveness, MemU empowers developers to create AI experiences that are both intelligent and engaging.

## Getting Started

### ☁️ **Cloud Version (Online Platform)**

The quickest way to integrate MemU:

*   **Instant Access:** Begin integrating AI memories within minutes.
*   **Managed Infrastructure:** We handle scaling, updates, and maintenance.
*   **Premium Support:** Priority assistance from our engineering team.

**Steps:**

1.  **Create an account:** [https://app.memu.so](https://app.memu.so)
2.  **Generate an API Key:** [https://app.memu.so/api-key/](https://app.memu.so/api-key/)
3.  **Install the package:**
    ```bash
    pip install memu-py
    ```
4.  **Integrate into your code:**
    ```python
    # Example usage
    from memu import MemuClient
    import os

    memu_client = MemuClient(
        base_url="https://api.memu.so",
        api_key=os.getenv("MEMU_API_KEY")
    )
    memu_client.memorize_conversation(
        conversation=conversation_text,  # Recommend longer conversation (~8000 tokens), see https://memu.pro/blog/memu-best-practice for details
        user_id="user001",
        user_name="User",
        agent_id="assistant001",
        agent_name="Assistant"
    )
    ```
    **See [`example/client/memory.py`](example/client/memory.py) for a complete integration example.**

### 🏢 **Enterprise Edition**

For organizations requiring advanced features and support:

*   **Commercial License:** Full features and usage rights.
*   **Custom Development:** SSO/RBAC, and scenario-specific optimization.
*   **Intelligence & Analytics:** User behavior analysis, real-time monitoring, and agent optimization.
*   **Premium Support:** Dedicated 24/7 support with custom SLAs.

*For enterprise inquiries: [contact@nevamind.ai](mailto:contact@nevamind.ai)*

### 🏠 **Self-Hosting (Community Edition)**

For local control and customization:

*   **Data Privacy:** Maintain data within your infrastructure.
*   **Customization:** Modify and extend the platform to fit your needs.
*   **Cost Control:** Avoid cloud fees for large deployments.

*See [self hosting README](README.self_host.md)*

## ✨ **Key Advantages of MemU**

### 📈 **Higher Memory Accuracy**

MemU achieves a 92.09% average accuracy on the Locomo dataset, surpassing the competition. (Technical Report coming soon!)

### 🚀 **Fast Retrieval**

MemU efficiently categorizes information for fast retrieval, removing the need for extensive embedding searches.

### 💰 **Low Cost**

Process hundreds of conversation turns simultaneously, eliminating the need for multiple memory operations, thus saving you tokens.

## 🎥 **Demo Video**

[Watch the MemU Demonstration Video](https://www.youtube.com/watch?v=qZIuCoLglHs)

## 🧠 **Memory as File System**

*   **Organize:** Autonomous Memory File Management. Intelligent folders managed by a memory agent.
*   **Link:** Interconnected Knowledge Graph. Builds meaningful connections between related memories.
*   **Evolve:** Continuous Self-Improvement. Generates insights and identifies patterns over time.
*   **Never Forget:** Adaptive Forgetting Mechanism. Prioritizes information based on usage.

## 💡 **Use Cases**

| AI Companion | AI Role Play | AI IP Characters | AI Education |
| :-----------: | :-----------: | :-------------: | :-----------: |
| <img src="assets/usecase/ai_companion-0000.jpg" width="150" height="200"> | <img src="assets/usecase/ai_role_play-0000.jpg" width="150" height="200"> | <img src="assets/usecase/ai_ip-0000.png" width="150" height="200"> | <img src="assets/usecase/ai_edu-0000.jpg" width="150" height="200"> |
| AI Therapy | AI Robot | AI Creation | More... |
| <img src="assets/usecase/ai_therapy-0000.jpg" width="150" height="200"> | <img src="assets/usecase/ai_robot-0000.jpg" width="150" height="200"> | <img src="assets/usecase/ai_creation-0000.jpg" width="150" height="200"> |  |

## 🤝 **Contribute to MemU**

We welcome your contributions!

*   **[Read our detailed Contributing Guide →](CONTRIBUTING.md)**

*   **License:** MemU contributions are licensed under the Apache License 2.0.

## 🌍 **Community & Support**

*   **GitHub Issues:** [Submit an issue](https://github.com/NevaMind-AI/memU/issues) for bug reports, feature requests, and project updates.
*   **Discord:** [Join our Discord](https://discord.com/invite/hQZntfGsbJ) for real-time support and community interaction.
*   **X (Twitter):** [Follow us on X](https://x.com/memU_ai) for updates and announcements.

## 🤝 **Ecosystem Partners**

<div align="center">

### Development Tools

[Image links to partner logos - see original markdown for details]

</div>

*Interested in partnering? Contact us at [contact@nevamind.ai](mailto:contact@nevamind.ai)*

## 📱 **Join Our WeChat Community**

<div align="center">
<img src="assets/qrcode.png" alt="MemU WeChat and discord QR Code" width="480" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px;">
</div>

*Scan the QR code to join our WeChat community.*

## 🙏 **Questionnaire**

Help us improve MemU! Take our 3-minute survey and get 30 free quota:  https://forms.gle/H2ZuZVHv72xbqjvd7
```
Key improvements and SEO optimizations:

*   **Concise Hook:**  The opening sentence is more direct and action-oriented.
*   **Clear Headings:**  Uses H1, H2, and H3 headings for structure and readability.
*   **Keyword Optimization:** Incorporated relevant keywords like "AI companion," "memory framework," "open source," "fast retrieval," etc.
*   **Bulleted Lists:** Uses bullet points for easy scanning and highlights key features.
*   **Call to Action:** Encourages users to get started, contribute, and join the community.
*   **Emphasis on Benefits:** Highlights the advantages of using MemU.
*   **Use Case Section:** Adds value by showing practical application.
*   **Community Engagement:**  Clear calls to action to join Discord, follow on Twitter, and contribute.
*   **Partner Section:**  Showcases the ecosystem, which can improve credibility and SEO.
*   **Mobile Optimization:** Includes a QR code to make it easier for mobile users to join WeChat and Discord
*   **Concise and Focused Language:** Avoids unnecessary jargon.
*   **Internal Links:**  Uses internal links for better navigation and user experience (e.g., contributing guide).
*   **Overall Flow:**  Organizes information logically, starting with the core offering and progressing through features, getting started guides, benefits, and community resources.