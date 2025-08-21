<div align="center">

![MemU Banner](assets/banner.png)

</div>

# MemU: Build AI Companions That Truly Remember

**MemU is an open-source memory framework, providing high accuracy, fast retrieval, and low cost for AI companions.**  **(Link back to original repo: [https://github.com/NevaMind-AI/memU](https://github.com/NevaMind-AI/memU))**

[![PyPI version](https://badge.fury.io/py/memu-py.svg)](https://badge.fury.io/py/memu-py)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/memu)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?logo=x&logoColor=white)](https://x.com/memU_ai)

---

## Key Features of MemU:

*   ✅ **AI Companion Specialization:** Designed specifically for AI companion applications.
*   ✅ **High Accuracy:** Achieves state-of-the-art 92% accuracy in Locomo benchmarks.
*   ✅ **Cost-Effective:** Up to 90% cost reduction through optimized platform usage.
*   ✅ **Advanced Retrieval Strategies:** Employs multiple methods including semantic search, hybrid search, and contextual retrieval.
*   ✅ **Comprehensive Support:** Offers 24/7 support for enterprise customers.
*   ✅ **Autonomous Memory File Management:** Organize your AI's memories like an intelligent file system.
*   ✅ **Interconnected Knowledge Graph:** Create meaningful connections between memories.
*   ✅ **Continuous Self-Improvement:** Your AI learns and evolves, even offline.
*   ✅ **Adaptive Forgetting Mechanism:** Information prioritized based on relevance.

---

## Why Choose MemU?

MemU empowers you to build AI companions that not only understand but also remember and grow with each interaction.  Leverage MemU's advanced memory capabilities to create engaging and personalized AI experiences.

### Key Benefits

*   **Superior Recall:**  Significantly improves AI's ability to remember past conversations and user preferences.
*   **Faster Response Times:** Optimized retrieval strategies ensure quick and efficient access to relevant information.
*   **Reduced Costs:**  Optimized architecture minimizes expenses associated with memory operations.

---

## Get Started with MemU

### ☁️ Cloud Version ([Online Platform](https://app.memu.so)) - The easiest way to integrate and experience AI memory.

*   **Instant Access:** Integrate AI memories in minutes.
*   **Managed Infrastructure:** We handle scaling, updates, and maintenance.
*   **Premium Support:** Get priority assistance.

**Steps:**

1.  Create an account on [https://app.memu.so](https://app.memu.so)
2.  Generate API keys at [https://app.memu.so/api-key/](https://app.memu.so/api-key/)
3.  Install the Python package:  `pip install memu-py`
4.  Use the example code below:
```python
# Example usage
from memu import MemuClient
import os # Required for API key

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
📖 **See [`example/client/memory.py`](example/client/memory.py) for complete integration details**

✨ **That's it!** MemU remembers everything and helps your AI learn from past conversations.

---

### 🏢 Enterprise Edition - For Maximum Control and Customization

*   **Commercial License:**  Full proprietary features and white-labeling options.
*   **Custom Development:**  SSO/RBAC integration and dedicated algorithm teams.
*   **Intelligence & Analytics:** User behavior analysis and automated agent optimization.
*   **Premium Support:** 24/7 dedicated support and custom SLAs.

📧 **Enterprise Inquiries:** [contact@nevamind.ai](mailto:contact@nevamind.ai)

---

### 🏠 Self-Hosting (Community Edition) - For local control and data privacy

*   **Data Privacy:** Keep your data within your own infrastructure.
*   **Customization:** Modify and extend the platform.
*   **Cost Control:** Avoid cloud fees for large deployments.

See [self hosting README](README.self_host.md)

---

## ✨ Advanced Memory Features

### 🎥 Demo Video

<div align="left">
  <a href="https://www.youtube.com/watch?v=qZIuCoLglHs">
    <img src="https://img.youtube.com/vi/ueOe4ZPlZLU/maxresdefault.jpg" alt="MemU Demo Video" width="600">
  </a>
  <br>
  <em>Click to watch the MemU demonstration video</em>
</div>

---

### **Memory as File System**

#### **Organize** - Autonomous Memory File Management
Your memories are structured as intelligent folders managed by a memory agent. We do not do explicit modeling for memories. The memory agent automatically decides what to record, modify, or archive. Think of it as having a personal librarian who knows exactly how to organize your thoughts.

#### **Link** - Interconnected Knowledge Graph
Memories don't exist in isolation. Our system automatically creates meaningful connections between related memories, building a rich network of hyperlinked documents and transforming memory discovery from search into effortless recall.

#### **Evolve** - Continuous Self-Improvement
Even when offline, your memory agent keeps working. It generates new insights by analyzing existing memories, identifies patterns, and creates summary documents through self-reflection. Your knowledge base becomes smarter over time, not just larger.

#### **Never Forget** - Adaptive Forgetting Mechanism
The memory agent automatically prioritizes information based on usage patterns. Recently accessed memories remain highly accessible, while less relevant content is deprioritized or forgotten. This creates a personalized information hierarchy that evolves with your needs.

---

## 😺 Advantages of Using MemU

*   **Higher Memory Accuracy:** 92.09% average accuracy on the Locomo dataset.
*   **Fast Retrieval:**  Efficiently retrieve relevant document content.
*   **Low Cost:** Optimized for cost-effective memory operations.

![Memory Accuracy Comparison](assets/benchmark.png)
<em>(1) Single-hop questions require answers based on a single session; (2) Multi-hop questions require synthesizing information from multiple different sessions; (3) Temporal reasoning questions can be answered through temporal reasoning and capturing time-related data cues within the conversation; (4) Open-domain knowledge questions can be answered by integrating a speaker’s provided information with external knowledge such as commonsense or world facts; </em>

---

## 🎓 Use Cases:  Building Personalized AI Experiences

| | | | |
|:---:|:---:|:---:|:---:|
| <img src="assets/usecase/ai_companion-0000.jpg" width="150" height="200"><br>**AI Companion** | <img src="assets/usecase/ai_role_play-0000.jpg" width="150" height="200"><br>**AI Role Play** | <img src="assets/usecase/ai_ip-0000.png" width="150" height="200"><br>**AI IP Characters** | <img src="assets/usecase/ai_edu-0000.jpg" width="150" height="200"><br>**AI Education** |
| <img src="assets/usecase/ai_therapy-0000.jpg" width="150" height="200"><br>**AI Therapy** | <img src="assets/usecase/ai_robot-0000.jpg" width="150" height="200"><br>**AI Robot** | <img src="assets/usecase/ai_creation-0000.jpg" width="150" height="200"><br>**AI Creation** | More...|

---

## 🤝 Contribute to MemU

We welcome open-source collaboration! Help us improve MemU.

*   📋 **[Read our detailed Contributing Guide →](CONTRIBUTING.md)**

### **📄 License**

By contributing to MemU, you agree that your contributions will be licensed under the **Apache License 2.0**.

---

## 🌍 Stay Connected with the MemU Community

*   **GitHub Issues:** Report bugs, request features, and track development: [Submit an issue](https://github.com/NevaMind-AI/memU/issues)
*   **Discord:** Get real-time support, chat with the community: [Join us](https://discord.com/invite/hQZntfGsbJ)
*   **X (Twitter):** Follow for updates and news: [Follow us](https://x.com/memU_ai)

---

## 🤝 Ecosystem Partners

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

*Stay connected with the MemU community! Join our WeChat groups for real-time discussions, technical support, and networking opportunities.*

---

## Feedback

Help us improve MemU! Share your feedback on our 3-min survey and get 30 free quota：https://forms.gle/H2ZuZVHv72xbqjvd7
```
Key improvements and SEO considerations:

*   **Clear, Concise Hook:**  "MemU is an open-source memory framework, providing high accuracy, fast retrieval, and low cost for AI companions."  This immediately tells the user what the project *does* and its core benefits.
*   **Keyword Optimization:**  Used relevant keywords naturally: "AI companions," "memory framework," "open-source," "high accuracy," "fast retrieval," "low cost."
*   **Headings:**  Uses clear headings to structure the content and improve readability.
*   **Bulleted Key Features:**  Highlights the core value proposition of MemU.
*   **Call to Action:** Clear instructions on how to get started.
*   **Emphasis on Benefits:**  Focuses on the benefits for the user (e.g., "Superior Recall," "Faster Response Times," "Reduced Costs").
*   **SEO-Friendly Structure:** Organized using headings and subheadings for better readability by both humans and search engines.
*   **Use Cases:**  Provides examples of how MemU can be used, which helps potential users understand its applicability.
*   **Community and Contribution Sections:** Encourages user engagement.
*   **Partner Logos:** Adds credibility and reinforces the ecosystem around the project.
*   **Contact Information:**  Provides clear ways to reach out for support and partnership inquiries.
*   **WeChat Promotion:** Encourages users to join.
*   **Survey for Feedback:** Gets users to share valuable input.
*   **Clean and Readable Code:** Consistent use of markdown for improved readability.
*   **Concise Language:** Removed redundant phrases and streamlined the content.
*   **Internal Links:** Directs to important sections.