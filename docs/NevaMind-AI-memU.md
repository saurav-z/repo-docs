<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/banner.png">
  <source media="(prefers-color-scheme: light)" srcset="assets/banner.png">
  <img alt="MemU Banner" src="assets/banner.png" width="100%">
</picture>
</div>

<!-- SEO-optimized title and description -->
# MemU: The Cutting-Edge Memory Framework for AI Companions

**Create AI companions that truly remember with MemU, the open-source memory framework designed for high accuracy, fast retrieval, and cost-effectiveness, built by [NevaMind-AI](https://github.com/NevaMind-AI/memU).**

[![PyPI version](https://badge.fury.io/py/memu-py.svg)](https://badge.fury.io/py/memu-py)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/memu)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?logo=x&logoColor=white)](https://x.com/memU_ai)

---

## Key Features

*   ✅ **AI Companion Specialization**: Designed specifically for AI companion applications.
*   ✅ **High Accuracy**: Achieve state-of-the-art performance with 92% accuracy in Locomo benchmark tests.
*   ✅ **Cost-Effective**: Reduce operational costs by up to 90% through optimized online platforms.
*   ✅ **Advanced Retrieval Strategies**: Utilize semantic search, hybrid search, and contextual retrieval methods.
*   ✅ **Dedicated Support**: Benefit from 24/7 support for enterprise customers.
*   **Memory as a File System:** Organize, Link, Evolve, and Never Forget with the system.

---

## Why Choose MemU?

MemU transforms how AI companions learn and interact, enabling them to build deep, personalized memories.  It's not just a memory framework; it's a foundation for creating AI that truly understands and remembers.

*   **Superior Accuracy:**  MemU consistently outperforms competitors, achieving 92% average accuracy in the Locomo dataset.
*   **Blazing Fast Retrieval:** Optimized retrieval processes find the right information quickly and efficiently.
*   **Cost Efficiency:** Efficient design ensures optimal use of resources, significantly reducing costs.

---

## 🚀 Get Started

MemU offers flexible integration options to suit your needs:

### ☁️ **Cloud Version (Online Platform)**

The quickest way to integrate MemU into your application.

*   **Instant Access**: Start integrating AI memories within minutes.
*   **Managed Infrastructure**:  We handle scaling, updates, and maintenance.
*   **Premium Support**:  Get priority assistance from our engineering team (subscription-based).

**How to start**:

1.  **Create an Account:**  Sign up at [https://app.memu.so](https://app.memu.so).
2.  **Generate API Keys:** Get your API keys from [https://app.memu.so/api-key/](https://app.memu.so/api-key/).
3.  **Integrate with your code:**

    ```python
    pip install memu-py
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

    See [API reference](docs/API_REFERENCE.md) or [our blog](https://memu.pro/blog) for more details.

    📖 **Explore [`example/client/memory.py`](example/client/memory.py) for complete integration details.**

### 🏢 **Enterprise Edition**

For organizations requiring robust security, customization, and control.

*   **Commercial License**: Utilize full proprietary features and commercial usage rights.
*   **Custom Development**: Benefit from SSO/RBAC integration and dedicated algorithm optimization.
*   **Intelligence & Analytics**: Access user behavior analysis, production monitoring, and agent optimization.
*   **Premium Support**: Enjoy 24/7 dedicated support and custom SLAs.

📧 **Enterprise Inquiries:** [contact@nevamind.ai](mailto:contact@nevamind.ai)

### 🏠 **Self-Hosting (Community Edition)**

For developers seeking local control and data privacy.

*   **Data Privacy**: Maintain sensitive data within your infrastructure.
*   **Customization**:  Modify and extend the platform.
*   **Cost Control**:  Avoid recurring cloud fees for large deployments.

See [self hosting README](README.self_host.md)

---

## ✨ Key Advantages in Detail

### 🎥 **Demo Video**

<div align="left">
  <a href="https://www.youtube.com/watch?v=qZIuCoLglHs">
    <img src="https://img.youtube.com/vi/ueOe4ZPlZLU/maxresdefault.jpg" alt="MemU Demo Video" width="600">
  </a>
  <br>
  <em>Click to watch the MemU demonstration video</em>
</div>

### **Memory as a File System:  A Revolutionary Approach**

*   **Organize**: Intelligent folders managed by a memory agent.  The agent decides what to record, modify, or archive.
*   **Link**: Automatically creates connections between related memories, building a network.
*   **Evolve**: Generates insights, identifies patterns, and creates summaries.
*   **Never Forget**: Prioritizes information based on usage patterns, creating a personalized hierarchy.

---

## 🏆 **Advantages**

### **Higher Memory Accuracy**

MemU achieves 92.09% average accuracy in Locomo dataset, significantly outperforming competitors.

![Memory Accuracy Comparison](assets/benchmark.png)
<em>(1) Single-hop questions require answers based on a single session; (2) Multi-hop questions require synthesizing information from multiple different sessions; (3) Temporal reasoning questions can be answered through temporal reasoning and capturing time-related data cues within the conversation; (4) Open-domain knowledge questions can be answered by integrating a speaker’s provided information with external knowledge such as commonsense or world facts; </em>

### **Fast Retrieval**

By categorizing information into documents, MemU eliminates the need for extensive embedding searches, allowing for faster retrieval of relevant information.

### **Low Cost**

Processing conversations in batches reduces the need for repeated memory function calls, saving on token usage and reducing costs. Check out our [best practice](https://memu.pro/blog/memu-best-practice).

---

## 🎓 **Use Cases**

Explore the diverse applications of MemU:

| | | | |
|:---:|:---:|:---:|:---:|
| <img src="assets/usecase/ai_companion-0000.jpg" width="150" height="200"><br>**AI Companion** | <img src="assets/usecase/ai_role_play-0000.jpg" width="150" height="200"><br>**AI Role Play** | <img src="assets/usecase/ai_ip-0000.png" width="150" height="200"><br>**AI IP Characters** | <img src="assets/usecase/ai_edu-0000.jpg" width="150" height="200"><br>**AI Education** |
| <img src="assets/usecase/ai_therapy-0000.jpg" width="150" height="200"><br>**AI Therapy** | <img src="assets/usecase/ai_robot-0000.jpg" width="150" height="200"><br>**AI Robot** | <img src="assets/usecase/ai_creation-0000.jpg" width="150" height="200"><br>**AI Creation** | More...|

---

## 🤝 **Contribute to MemU**

Your contributions are welcome!  Help shape the future of AI companions.

📋 **[Read our detailed Contributing Guide →](CONTRIBUTING.md)**

### **📄 License**

MemU is licensed under the Apache License 2.0.

---

## 🌍 **Community**

*   **GitHub Issues:** Report bugs, request features, and track development.  [Submit an issue](https://github.com/NevaMind-AI/memU/issues)
*   **Discord:** Get real-time support and chat with the community. [Join us](https://discord.com/invite/hQZntfGsbJ)
*   **X (Twitter):** Follow for updates and announcements.  [Follow us](https://x.com/memU_ai)

For more information please contact info@nevamind.ai

---

## 🤝 **Ecosystem**

We collaborate with leading development tool providers:

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

## 📱 **Join Our WeChat Community**

Stay updated with MemU.

<div align="center">
<img src="assets/qrcode.png" alt="MemU WeChat and discord QR Code" width="480" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px;">
*Scan any of the QR codes above to join our WeChat community*
</div>

---

## Survey

Help us improve! Share your feedback on our 3-min survey and get 30 free quota：https://forms.gle/H2ZuZVHv72xbqjvd7

```
Key improvements and SEO optimizations:

*   **Concise Hook:** A compelling one-sentence introduction that clearly states the value proposition.
*   **Clear Headings:** Use of descriptive headings (e.g., "Key Features," "Why Choose MemU?," "Get Started") for better readability and SEO.
*   **Bulleted Lists:**  Using bullet points for key features, advantages, and getting started steps for improved scannability.
*   **Keyword Optimization:**  Incorporates relevant keywords like "AI companion," "memory framework," "open-source," "high accuracy," "fast retrieval," and "cost-effective" throughout the content.
*   **Clear Structure:** Organized content into logical sections for better user experience and SEO.
*   **Call to Action:**  Encourages users to explore the documentation, join the community, and contribute.
*   **Internal Links:**  Linking back to the original repo at the beginning of the document to make sure the reader knows where the original content is and the project's home.
*   **Partner Section:** Maintained and formatted the partner section to maintain community links.
*   **Alt Text:** Added alt text to images for accessibility and SEO.
*   **Clear Steps:**  Simplified the "Get Started" section with concise steps, highlighting key actions.
*   **WeChat community section**: Added a section to encourage WeChat Community members to join the discussion.